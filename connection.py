from __future__ import annotations

import argparse
import time
from typing import List, Optional

import serial
from serial import SerialException
from serial.tools import list_ports


def list_available_ports() -> List[str]:
    return [p.device for p in list_ports.comports()]


def auto_detect_port() -> Optional[str]:
    ports = list_ports.comports()
    preferred = [
        p
        for p in ports
        if any(keyword in p.description.lower() for keyword in ["arduino", "ch340", "usb serial"])
    ]
    if preferred:
        return preferred[0].device
    if ports:
        return ports[0].device
    return None


class SerialConnection:
    def __init__(self, port: Optional[str], baudrate: int = 115200, timeout: float = 1.0) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._ser: serial.Serial | None = None

    def connect(self) -> None:
        if self._ser is not None:
            return
        if not self.port:
            self.port = auto_detect_port()
        if not self.port:
            ports = list_available_ports()
            raise SerialException(f"No serial ports found. Available ports: {ports if ports else 'none'}")
        self._ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
        # Arduino usually resets once serial is opened.
        time.sleep(2.0)

    def readline(self) -> str:
        if self._ser is None:
            self.connect()
        assert self._ser is not None
        return self._ser.readline().decode("utf-8", errors="ignore").strip()

    def close(self) -> None:
        if self._ser is not None:
            try:
                self._ser.close()
            except SerialException:
                pass
            self._ser = None

    def __enter__(self) -> "SerialConnection":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.close()


def run_debug_stream(port: Optional[str], baudrate: int) -> None:
    try:
        with SerialConnection(port=port, baudrate=baudrate) as conn:
            print(f"Connected to {port} @ {baudrate}")
            while True:
                print(conn.readline())
    except SerialException as exc:
        ports = list_available_ports()
        print(f"Serial error: {exc}")
        print(f"Available ports: {ports if ports else 'none'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug Arduino serial stream")
    parser.add_argument("--port", default=None, help="Serial port (auto-detect if omitted)")
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baudrate")
    args = parser.parse_args()
    run_debug_stream(port=args.port, baudrate=args.baudrate)


if __name__ == "__main__":
    main()