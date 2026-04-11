from __future__ import annotations

import queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional

import numpy as np
import streamlit as st

from connection import auto_detect_port, list_available_ports
from tts import GestureSpeaker
from utils import (
    ArduinoSerialStream,
    NoiseFilter,
    RollingSensorBuffer,
    SENSOR_COLUMNS,
    SensorSmoother,
    load_model_artifacts,
    majority_vote,
    prepare_window_features,
)

MODEL_PATH = Path("models") / "gesture_model.pkl"
ENCODER_PATH = Path("models") / "label_encoder.pkl"


def _prediction_worker(
    out_q: "queue.Queue[tuple[str, float] | tuple[str, str]]",
    stop_event: threading.Event,
    port: Optional[str],
    baudrate: int,
    min_confidence: float,
    speak: bool,
) -> None:
    speaker = GestureSpeaker()
    if speak:
        speaker.start()

    try:
        model, label_encoder = load_model_artifacts(MODEL_PATH, ENCODER_PATH)
        buffer = RollingSensorBuffer(maxlen=40)
        smoother = SensorSmoother(num_features=len(SENSOR_COLUMNS), median_window=5, spike_threshold=120.0)
        vote_history: Deque[str] = deque(maxlen=5)
        filter_ = NoiseFilter()

        with ArduinoSerialStream(port=port, baudrate=baudrate) as stream:
            while not stop_event.is_set():
                parsed = stream.read_parsed()
                if parsed is None:
                    continue

                sample = [parsed[col] for col in SENSOR_COLUMNS]
                smoothed_sample = smoother.smooth(sample)
                buffer.append(smoothed_sample)
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
                out_q.put((smoothed, confidence))
                if speak:
                    speaker.speak(smoothed)

    except Exception as exc:  # pragma: no cover
        out_q.put(("__ERROR__", str(exc)))
    finally:
        speaker.stop()


def _init_state() -> None:
    if "rt_running" not in st.session_state:
        st.session_state.rt_running = False
    if "rt_queue" not in st.session_state:
        st.session_state.rt_queue = queue.Queue()
    if "rt_stop_event" not in st.session_state:
        st.session_state.rt_stop_event = None
    if "rt_thread" not in st.session_state:
        st.session_state.rt_thread = None
    if "rt_latest_label" not in st.session_state:
        st.session_state.rt_latest_label = "-"
    if "rt_latest_conf" not in st.session_state:
        st.session_state.rt_latest_conf = 0.0
    if "rt_history" not in st.session_state:
        st.session_state.rt_history = []
    if "rt_error" not in st.session_state:
        st.session_state.rt_error = ""


def _start_runtime(port: Optional[str], baudrate: int, min_conf: float, speak: bool) -> None:
    if st.session_state.rt_running:
        return

    st.session_state.rt_error = ""
    st.session_state.rt_stop_event = threading.Event()
    st.session_state.rt_queue = queue.Queue()
    st.session_state.rt_thread = threading.Thread(
        target=_prediction_worker,
        args=(st.session_state.rt_queue, st.session_state.rt_stop_event, port, baudrate, min_conf, speak),
        daemon=True,
    )
    st.session_state.rt_thread.start()
    st.session_state.rt_running = True


def _stop_runtime() -> None:
    if not st.session_state.rt_running:
        return

    stop_event = st.session_state.rt_stop_event
    thread = st.session_state.rt_thread
    if stop_event is not None:
        stop_event.set()
    if thread is not None and thread.is_alive():
        thread.join(timeout=2.0)

    st.session_state.rt_running = False


def _drain_updates() -> None:
    q = st.session_state.rt_queue
    while not q.empty():
        item = q.get_nowait()
        if item[0] == "__ERROR__":
            st.session_state.rt_error = str(item[1])
            st.session_state.rt_running = False
            continue

        label, confidence = item
        st.session_state.rt_latest_label = str(label)
        st.session_state.rt_latest_conf = float(confidence)
        st.session_state.rt_history.append(f"{label} ({confidence:.3f})")
        if len(st.session_state.rt_history) > 30:
            st.session_state.rt_history = st.session_state.rt_history[-30:]


def main() -> None:
    st.set_page_config(page_title="Runtime Display", layout="wide")
    _init_state()

    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(135deg, #0f2027 0%, #203a43 45%, #2c5364 100%); color: #f9fcff; }
        .big-word { font-size: 130px; font-weight: 800; letter-spacing: 2px; line-height: 1.05; color: #ffffff; }
        .big-letter { font-size: 240px; font-weight: 900; line-height: 0.95; color: #8be9fd; }
        .meta { font-size: 28px; color: #d2eaf3; }
        .panel { padding: 1rem 1.2rem; border-radius: 14px; background: rgba(255,255,255,0.08); }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Live Runtime Display")
    st.caption("Continuous gesture-to-word display with speech")

    ports = list_available_ports()
    detected = auto_detect_port()

    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
    with c1:
        mode = st.radio("Port", options=["Auto", "Manual"], horizontal=True)
    with c2:
        port = detected if mode == "Auto" else (st.selectbox("COM", options=ports) if ports else None)
    with c3:
        baud = int(st.number_input("Baud", value=115200, step=1))
    with c4:
        min_conf = float(st.slider("Min confidence", min_value=0.10, max_value=0.95, value=0.45, step=0.05))

    speak = st.checkbox("Speech ON", value=True)

    a, b = st.columns(2)
    with a:
        if st.button("Start Runtime", type="primary", use_container_width=True):
            if not MODEL_PATH.exists() or not ENCODER_PATH.exists():
                st.error("Model not found. Train first from the main UI.")
            elif not port:
                st.error("No serial port available.")
            else:
                _start_runtime(port=port, baudrate=baud, min_conf=min_conf, speak=speak)
    with b:
        if st.button("Stop Runtime", use_container_width=True):
            _stop_runtime()

    _drain_updates()

    if st.session_state.rt_error:
        st.error(f"Runtime error: {st.session_state.rt_error}")

    label = str(st.session_state.rt_latest_label)
    main_word = label.replace("_", " ").upper()
    main_letter = main_word[:1] if main_word and main_word != "-" else "-"

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(f'<div class="big-word">{main_word}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="big-letter">{main_letter}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="meta">Confidence: {st.session_state.rt_latest_conf:.3f} | Running: {st.session_state.rt_running}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Recent Predictions")
    if st.session_state.rt_history:
        for line in st.session_state.rt_history[-12:][::-1]:
            st.write(line)
    else:
        st.write("No predictions yet.")

    if st.session_state.rt_running:
        time.sleep(0.25)
        st.rerun()


if __name__ == "__main__":
    main()
