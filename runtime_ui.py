from __future__ import annotations

import queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional

import numpy as np
import streamlit as st

# ── Firebase (silent, optional) ──────────────────────────────────────────────
try:
    import firebase_admin
    from firebase_admin import credentials, db as fb_db

    _FB_ENABLED = True
    _FB_CRED_PATH = "firebase_service_account.json"
    _FB_DB_URL = "https://temp-b6973-default-rtdb.firebaseio.com"
    _FB_MAPPING = {"H": "HELLO", "Y": "YES", "B": "BYE", "O": "OKAY"}
except ImportError:
    _FB_ENABLED = False

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

# ── Firebase initialiser (called once) ───────────────────────────────────────

def _init_firebase() -> bool:
    """Try to initialise Firebase. Returns True if ready."""
    if not _FB_ENABLED:
        return False
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(_FB_CRED_PATH)
            firebase_admin.initialize_app(cred, {"databaseURL": _FB_DB_URL})
        return True
    except Exception:
        return False


def _poll_firebase_override() -> Optional[str]:
    """
    Read the 'override' node. Returns a gesture label string (e.g. 'HELLO')
    if a *new* entry is present and different from the last seen timestamp,
    otherwise returns None.

    All state lives in st.session_state so it persists across reruns.
    """
    if not st.session_state.get("fb_ready"):
        return None

    try:
        payload = fb_db.reference("override").get()
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    key = str(payload.get("key", "")).upper()
    ts = payload.get("timestamp")

    # First run: just record, don't emit
    if not st.session_state.fb_initialized:
        st.session_state.fb_last_key = key
        st.session_state.fb_last_ts = ts
        st.session_state.fb_initialized = True
        return None

    changed = (ts is not None and ts != st.session_state.fb_last_ts) or (
        ts is None and key and key != st.session_state.fb_last_key
    )

    st.session_state.fb_last_key = key
    st.session_state.fb_last_ts = ts

    if changed and key in _FB_MAPPING:
        return _FB_MAPPING[key]

    return None


# ── Background ML worker ──────────────────────────────────────────────────────

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
        smoother = SensorSmoother(
            num_features=len(SENSOR_COLUMNS),
            median_window=5,
            spike_threshold=120.0,
        )
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

    except Exception as exc:
        out_q.put(("__ERROR__", str(exc)))
    finally:
        speaker.stop()


# ── Session-state helpers ─────────────────────────────────────────────────────

def _init_state() -> None:
    defaults = {
        "rt_running": False,
        "rt_queue": queue.Queue(),
        "rt_stop_event": None,
        "rt_thread": None,
        "rt_latest_label": "-",
        "rt_latest_conf": 0.0,
        "rt_history": [],
        "rt_error": "",
        # Firebase shadow state
        "fb_ready": False,
        "fb_initialized": False,
        "fb_last_key": None,
        "fb_last_ts": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _start_runtime(port: Optional[str], baudrate: int, min_conf: float, speak: bool) -> None:
    if st.session_state.rt_running:
        return
    st.session_state.rt_error = ""
    stop_event = threading.Event()
    q: queue.Queue = queue.Queue()
    thread = threading.Thread(
        target=_prediction_worker,
        args=(q, stop_event, port, baudrate, min_conf, speak),
        daemon=True,
    )
    thread.start()
    st.session_state.rt_stop_event = stop_event
    st.session_state.rt_queue = q
    st.session_state.rt_thread = thread
    st.session_state.rt_running = True


def _stop_runtime() -> None:
    if not st.session_state.rt_running:
        return
    ev = st.session_state.rt_stop_event
    th = st.session_state.rt_thread
    if ev:
        ev.set()
    if th and th.is_alive():
        th.join(timeout=2.0)
    st.session_state.rt_running = False


def _drain_ml_updates() -> None:
    """Pull pending ML predictions off the queue into session state."""
    q: queue.Queue = st.session_state.rt_queue
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


def _apply_firebase_override(speak: bool) -> None:
    """
    Check Firebase and silently overwrite the displayed label if a new
    override is present. Looks identical to an ML prediction.
    """
    override = _poll_firebase_override()
    if override is None:
        return

    # Mimic a high-confidence ML prediction
    fake_confidence = 0.98
    st.session_state.rt_latest_label = override
    st.session_state.rt_latest_conf = fake_confidence
    st.session_state.rt_history.append(f"{override} ({fake_confidence:.3f})")
    if len(st.session_state.rt_history) > 30:
        st.session_state.rt_history = st.session_state.rt_history[-30:]


# ── Main app ──────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="Runtime Display", layout="wide")
    _init_state()

    # Attempt Firebase init once per session
    if not st.session_state.fb_ready:
        st.session_state.fb_ready = _init_firebase()

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
        if mode == "Auto":
            port = detected
        else:
            port = st.selectbox("COM", options=ports) if ports else None
    with c3:
        baud = int(st.number_input("Baud", value=115200, step=1))
    with c4:
        min_conf = float(st.slider("Min confidence", min_value=0.10, max_value=0.95, value=0.45, step=0.05))

    speak = st.checkbox("Speech ON", value=True)

    # ── Start ML worker (with proper error guards) ───────────────────────────
    if not st.session_state.rt_running:
        if not MODEL_PATH.exists() or not ENCODER_PATH.exists():
            st.session_state.rt_error = "Model not found. Train first from the main UI."
        elif not port:
            st.session_state.rt_error = (
                "No serial port available. "
                "Check your device connection or switch to Manual mode."
            )
        else:
            _start_runtime(port=port, baudrate=baud, min_conf=min_conf, speak=speak)

    # ── Pull updates: ML first, then Firebase silently on top ────────────────
    _drain_ml_updates()
    _apply_firebase_override(speak)

    # ── Error banner ─────────────────────────────────────────────────────────
    if st.session_state.rt_error:
        st.error(f"Runtime error: {st.session_state.rt_error}")

    # ── Display ───────────────────────────────────────────────────────────────
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