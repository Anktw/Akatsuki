from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional

import pyttsx3


class GestureSpeaker:
    """Threaded pyttsx3 wrapper with anti-repeat debounce logic."""

    def __init__(self, debounce_seconds: float = 1.8) -> None:
        self.debounce_seconds = debounce_seconds
        self._last_spoken_label: Optional[str] = None
        self._last_spoken_time: float = 0.0
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._worker.start()

    def stop(self) -> None:
        if not self._started:
            return
        self._queue.put(None)
        self._worker.join(timeout=2.0)
        self._started = False

    def speak(self, label: str) -> bool:
        now = time.time()
        if (
            self._last_spoken_label == label
            and now - self._last_spoken_time < self.debounce_seconds
        ):
            return False

        self._last_spoken_label = label
        self._last_spoken_time = now
        self._queue.put(label)
        return True

    def _run(self) -> None:
        engine = pyttsx3.init()
        engine.setProperty("rate", 170)

        while True:
            label = self._queue.get()
            if label is None:
                break
            try:
                engine.say(label)
                engine.runAndWait()
            except Exception as exc:  # pragma: no cover
                logging.warning("TTS failed: %s", exc)

        engine.stop()
