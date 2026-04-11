from __future__ import annotations

import argparse
import logging
import time
from collections import deque
from pathlib import Path
from typing import Deque

import numpy as np

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
    setup_logging,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Real-time gesture prediction from Arduino stream")
    parser.add_argument("--port", default=None, help="Serial port (auto-detect if omitted)")
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baud rate")
    parser.add_argument("--model-path", default="models/gesture_model.pkl", help="Path to model")
    parser.add_argument(
        "--encoder-path",
        default="models/label_encoder.pkl",
        help="Path to label encoder",
    )
    parser.add_argument("--window-size", type=int, default=40, help="Buffer length for prediction")
    parser.add_argument("--smooth-size", type=int, default=5, help="Majority vote queue size")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.45,
        help="Ignore low-confidence predictions",
    )
    parser.add_argument(
        "--poll-sleep",
        type=float,
        default=0.005,
        help="Sleep between serial reads to reduce CPU usage",
    )
    parser.add_argument(
        "--disable-tts",
        action="store_true",
        help="Disable pyttsx3 speech output",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    setup_logging()

    model, label_encoder = load_model_artifacts(Path(args.model_path), Path(args.encoder_path))

    buffer = RollingSensorBuffer(maxlen=args.window_size)
    smoother = SensorSmoother(num_features=len(SENSOR_COLUMNS), median_window=5, spike_threshold=120.0)
    pred_history: Deque[str] = deque(maxlen=args.smooth_size)
    noise_filter = NoiseFilter()

    speaker = GestureSpeaker()
    if not args.disable_tts:
        speaker.start()

    last_announced = ""

    try:
        with ArduinoSerialStream(port=args.port, baudrate=args.baudrate) as stream:
            logging.info("Running real-time prediction loop. Press Ctrl+C to stop.")

            while True:
                parsed = stream.read_parsed()
                if parsed is None:
                    continue

                sample = [parsed[col] for col in SENSOR_COLUMNS]
                smoothed_sample = smoother.smooth(sample)
                buffer.append(smoothed_sample)

                if not buffer.is_full():
                    time.sleep(args.poll_sleep)
                    continue

                window = buffer.as_list()
                if noise_filter.is_idle(window):
                    time.sleep(args.poll_sleep)
                    continue

                feature = prepare_window_features(window).reshape(1, -1)
                pred_idx = int(model.predict(feature)[0])
                pred_label = str(label_encoder.inverse_transform([pred_idx])[0])

                confidence = 1.0
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(feature)[0]
                    confidence = float(np.max(proba))

                if confidence < args.min_confidence:
                    time.sleep(args.poll_sleep)
                    continue

                pred_history.append(pred_label)
                smoothed = majority_vote(pred_history)

                if smoothed != last_announced:
                    print(f"Predicted: {smoothed} | confidence={confidence:.3f}")
                    last_announced = smoothed
                else:
                    print(f"Predicted: {smoothed} | confidence={confidence:.3f}", end="\r")

                if not args.disable_tts:
                    speaker.speak(smoothed)

                time.sleep(args.poll_sleep)

    except KeyboardInterrupt:
        logging.info("Stopping prediction...")
    finally:
        speaker.stop()


if __name__ == "__main__":
    main()
