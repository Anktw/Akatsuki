from __future__ import annotations

import csv
import logging
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence

import joblib
import numpy as np
from serial import SerialException

from connection import SerialConnection, list_available_ports


SENSOR_COLUMNS: List[str] = [
    "f0",
    "f1",
    "f2",
    "f3",
    "f4",
]

SERIAL_COLUMNS: List[str] = ["timestamp", *SENSOR_COLUMNS]
DATA_COLUMNS: List[str] = [*SERIAL_COLUMNS, "label"]


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_serial_line(line: str) -> Optional[Dict[str, float]]:
    """Parse one CSV line from Arduino into a dict with numeric values."""
    line = line.strip()
    if not line:
        return None

    parts = [p.strip() for p in line.split(",")]
    try:
        values = [float(value) for value in parts]
    except ValueError:
        return None

    if len(values) == len(SERIAL_COLUMNS):
        return dict(zip(SERIAL_COLUMNS, values))

    if len(values) == len(SENSOR_COLUMNS):
        return {"timestamp": time.time(), **dict(zip(SENSOR_COLUMNS, values))}

    return None


def append_rows_to_csv(file_path: Path, rows: Iterable[Dict[str, float]]) -> int:
    """Append rows to CSV and write header automatically if file is new."""
    rows = list(rows)
    if not rows:
        return 0

    ensure_dir(file_path.parent)
    write_header = not file_path.exists()

    with file_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DATA_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def flatten_window(window: np.ndarray) -> np.ndarray:
    """Flatten a [window_size, num_features] array into a 1D feature vector."""
    if window.ndim != 2:
        raise ValueError("Window must be a 2D array")
    return window.astype(np.float32, copy=False).reshape(-1)


def prepare_window_features(window_samples: Sequence[Sequence[float]]) -> np.ndarray:
    """Prepare one feature vector from a sensor window for model inference."""
    window_arr = np.asarray(window_samples, dtype=np.float32)
    return flatten_window(window_arr)


def majority_vote(labels: Sequence[str]) -> str:
    if not labels:
        raise ValueError("Cannot vote on empty label sequence")
    counter = Counter(labels)
    return counter.most_common(1)[0][0]


@dataclass
class NoiseFilter:
    """Basic idle/noise filter to avoid predicting on static hand windows."""

    min_flex_range: float = 12.0
    min_motion_std: float = 0.8

    def is_idle(self, window_samples: Sequence[Sequence[float]]) -> bool:
        arr = np.asarray(window_samples, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != len(SENSOR_COLUMNS):
            return False

        flex = arr[:, :5]
        flex_dynamic = np.mean(np.max(flex, axis=0) - np.min(flex, axis=0))

        # In flex-only mode (5 sensors), rely on flex range only.
        if arr.shape[1] <= 5:
            return flex_dynamic < self.min_flex_range

        imu = arr[:, 5:]
        imu_dynamic = np.mean(np.std(imu, axis=0))
        return flex_dynamic < self.min_flex_range and imu_dynamic < self.min_motion_std


class RollingSensorBuffer:
    def __init__(self, maxlen: int) -> None:
        self._buffer: Deque[List[float]] = deque(maxlen=maxlen)

    def append(self, sample: Sequence[float]) -> None:
        self._buffer.append(list(sample))

    def is_full(self) -> bool:
        return len(self._buffer) == self._buffer.maxlen

    def as_list(self) -> List[List[float]]:
        return list(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()


class SensorSmoother:
    """Realtime smoothing using per-channel median + anti-spike clamp."""

    def __init__(
        self,
        num_features: int,
        median_window: int = 5,
        spike_threshold: float = 120.0,
        blend_alpha: float = 0.35,
    ) -> None:
        self.num_features = num_features
        self.median_window = max(1, int(median_window))
        self.spike_threshold = float(spike_threshold)
        self.blend_alpha = float(blend_alpha)

        self._history: Deque[List[float]] = deque(maxlen=self.median_window)
        self._last_output: Optional[np.ndarray] = None

    def smooth(self, sample: Sequence[float]) -> List[float]:
        current = np.asarray(sample, dtype=np.float32)
        if current.shape[0] != self.num_features:
            return list(current)

        self._history.append(list(current))
        hist = np.asarray(self._history, dtype=np.float32)
        median = np.median(hist, axis=0)

        if self._last_output is None:
            output = median
        else:
            delta = median - self._last_output
            clipped = np.clip(delta, -self.spike_threshold, self.spike_threshold)
            candidate = self._last_output + clipped
            output = (self.blend_alpha * candidate) + ((1.0 - self.blend_alpha) * self._last_output)

        self._last_output = output.astype(np.float32)
        return self._last_output.tolist()


class ArduinoSerialStream:
    """Robust serial reader with reconnect loop for Arduino streaming."""

    def __init__(
        self,
        port: Optional[str] = None,
        baudrate: int = 115200,
        timeout: float = 1.0,
        reconnect_delay: float = 2.0,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.reconnect_delay = reconnect_delay
        self._conn: Optional[SerialConnection] = None

    def connect(self) -> None:
        while self._conn is None:
            try:
                logging.info("Connecting to %s @ %s", self.port, self.baudrate)
                self._conn = SerialConnection(
                    port=self.port,
                    baudrate=self.baudrate,
                    timeout=self.timeout,
                )
                self._conn.connect()
                self.port = self._conn.port
                logging.info("Serial connected")
            except SerialException as exc:
                ports = list_available_ports()
                logging.warning("Serial connect failed: %s", exc)
                logging.warning("Available serial ports: %s", ports if ports else "none")
                time.sleep(self.reconnect_delay)

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except SerialException:
                pass
            self._conn = None

    def read_parsed(self) -> Optional[Dict[str, float]]:
        if self._conn is None:
            self.connect()

        assert self._conn is not None
        try:
            line = self._conn.readline()
            parsed = parse_serial_line(line)
            return parsed
        except SerialException as exc:
            logging.warning("Serial read failed, reconnecting: %s", exc)
            self.close()
            time.sleep(self.reconnect_delay)
            return None

    def __enter__(self) -> "ArduinoSerialStream":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.close()


def load_model_artifacts(model_path: Path, encoder_path: Path):
    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    return model, label_encoder
