from __future__ import annotations

import string
import webbrowser
import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional

import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from collect_data import collect_samples, sanitize_label
from connection import auto_detect_port, list_available_ports
from tts import GestureSpeaker
from train_model import load_dataset_file, sliding_windows, validate_trainable_labels
from utils import (
    ArduinoSerialStream,
    NoiseFilter,
    RollingSensorBuffer,
    SENSOR_COLUMNS,
    append_rows_to_csv,
    load_model_artifacts,
    majority_vote,
    prepare_window_features,
)


DATA_FILE = Path("data") / "simulated_gestures.csv"
MODEL_DIR = Path("models")
MODEL_FILE = MODEL_DIR / "gesture_model.pkl"
ENCODER_FILE = MODEL_DIR / "label_encoder.pkl"


def _preset_gestures() -> List[str]:
    return list(string.ascii_uppercase) + [str(i) for i in range(10)] + ["SPACE", "CUSTOM"]


def _selected_label(choice: str, custom_text: str) -> str:
    if choice == "SPACE":
        return "space"
    if choice == "CUSTOM":
        return sanitize_label(custom_text)
    return sanitize_label(choice)


def _train_model(data_file: Path, model_dir: Path, window_size: int, step_size: int, test_size: float) -> str:
    df = load_dataset_file(data_file)
    X, y_labels = sliding_windows(df, window_size=window_size, step_size=step_size)
    validate_trainable_labels(y_labels)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_labels)

    stratify = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

    model_dir.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(model, model_dir / "gesture_model.pkl")
    joblib.dump(label_encoder, model_dir / "label_encoder.pkl")

    return f"Accuracy: {accuracy:.4f}\n\n{report}"


def _predict_live(
    port: Optional[str],
    baudrate: int,
    seconds: int,
    speak: bool,
    min_confidence: float,
) -> List[str]:
    model, label_encoder = load_model_artifacts(MODEL_FILE, ENCODER_FILE)

    buffer = RollingSensorBuffer(maxlen=40)
    vote_history: Deque[str] = deque(maxlen=5)
    filter_ = NoiseFilter()
    speaker = GestureSpeaker()
    if speak:
        speaker.start()

    outputs: List[str] = []
    try:
        with ArduinoSerialStream(port=port, baudrate=baudrate) as stream:
            start = time.time()
            while time.time() - start < seconds:
                parsed = stream.read_parsed()
                if parsed is None:
                    continue

                sample = [parsed[col] for col in SENSOR_COLUMNS]
                buffer.append(sample)
                if not buffer.is_full():
                    continue

                window = buffer.as_list()
                if filter_.is_idle(window):
                    continue

                features = prepare_window_features(window).reshape(1, -1)
                pred_idx = int(model.predict(features)[0])
                pred_label = str(label_encoder.inverse_transform([pred_idx])[0])

                confidence = 1.0
                if hasattr(model, "predict_proba"):
                    confidence = float(np.max(model.predict_proba(features)[0]))

                if confidence < min_confidence:
                    continue

                vote_history.append(pred_label)
                smoothed = majority_vote(vote_history)
                outputs.append(f"{smoothed} ({confidence:.3f})")

                if speak:
                    speaker.speak(smoothed)
    finally:
        speaker.stop()

    return outputs


def _record_samples_with_progress(
    stream: ArduinoSerialStream,
    label: str,
    target_samples: int,
    progress_bar,
    status_box,
) -> List[dict]:
    rows: List[dict] = []
    start_time = time.time()
    while len(rows) < target_samples:
        parsed = stream.read_parsed()
        if parsed is None:
            status_box.warning(f"Waiting for valid sensor data... captured {len(rows)}/{target_samples}")
            continue

        parsed["label"] = label
        rows.append(parsed)
        progress_bar.progress(len(rows) / target_samples)
        elapsed = time.time() - start_time
        status_box.info(f"Recording '{label}'... {len(rows)}/{target_samples} rows captured in {elapsed:.1f}s")

    return rows


def main() -> None:
    st.set_page_config(page_title="Gesture Control", layout="centered")
    st.title("Gesture Control")
    st.write("A team project.")

    ports = list_available_ports()
    detected_port = auto_detect_port()

    st.subheader("Connection")
    port_mode = st.radio("Port", options=["Auto", "Manual"], horizontal=True)
    if port_mode == "Auto":
        selected_port = detected_port
        st.write(f"Detected port: {selected_port if selected_port else 'None'}")
    else:
        selected_port = st.selectbox("Choose COM port", options=ports) if ports else None
    baudrate = st.number_input("Baudrate", value=115200, step=1)

    st.divider()
    st.subheader("1. Record")
    gesture_choice = st.selectbox("Gesture", options=_preset_gestures())
    custom_text = st.text_input("Custom gesture text", value="")
    record_label = _selected_label(gesture_choice, custom_text)
    st.caption(f"Saving 5-value flex sensor rows to {DATA_FILE}")

    if st.button("Record 40 Readings", type="primary"):
        if not selected_port:
            st.error("No serial port available.")
        elif gesture_choice == "CUSTOM" and not custom_text.strip():
            st.error("Enter a custom gesture label.")
        else:
            try:
                progress_bar = st.progress(0)
                status_box = st.empty()
                status_box.info(f"Starting capture for '{record_label}'...")
                with ArduinoSerialStream(port=selected_port, baudrate=int(baudrate)) as stream:
                    st.caption("Reading the next 40 flex sensor rows from the serial port...")
                    rows = _record_samples_with_progress(
                        stream=stream,
                        label=record_label,
                        target_samples=40,
                        progress_bar=progress_bar,
                        status_box=status_box,
                    )
                    saved = append_rows_to_csv(DATA_FILE, rows)
                status_box.success(f"Recording complete. Saved {saved} rows for '{record_label}'.")
                st.success(f"Recorded and saved {saved} rows for '{record_label}'.")
            except Exception as exc:
                st.error(f"Record failed: {exc}")

    st.divider()
    st.subheader("2. Train Model")
    window_size = st.number_input("Window size", value=40, min_value=10, step=1)
    step_size = st.number_input("Step size", value=20, min_value=1, step=1)
    test_size = st.slider("Test split", min_value=0.1, max_value=0.4, value=0.2, step=0.05)

    if st.button("Train Model"):
        try:
            if not DATA_FILE.exists():
                st.error(f"Data file not found: {DATA_FILE}")
            else:
                with st.spinner("Training model..."):
                    result_text = _train_model(DATA_FILE, MODEL_DIR, int(window_size), int(step_size), float(test_size))
                st.success(f"Model saved to {MODEL_FILE} and {ENCODER_FILE}")
                st.text(result_text)
        except Exception as exc:
            st.error(f"Training failed: {exc}")

    st.divider()
    st.subheader("3. Start Model")
    st.write("Open the continuous runtime display in a new page with big words, letters, and speech.")
    st.code("python automate.py runtime-ui", language="bash")
    runtime_url = st.text_input("Runtime URL", value="http://localhost:8502")

    c_open, c_visit = st.columns(2)
    with c_open:
        if st.button("Open Runtime Page", type="primary", use_container_width=True):
            if not MODEL_FILE.exists() or not ENCODER_FILE.exists():
                st.error("Train the model first.")
            else:
                webbrowser.open_new_tab(runtime_url)
                st.success(f"Opened {runtime_url} in a new tab.")
    with c_visit:
        st.link_button("Visit Runtime URL", runtime_url, use_container_width=True)


if __name__ == "__main__":
    main()
