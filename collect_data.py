from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List

from utils import ArduinoSerialStream, append_rows_to_csv, setup_logging


def sanitize_label(label: str) -> str:
    return "_".join(label.strip().lower().split())


def collect_samples(
    stream: ArduinoSerialStream,
    label: str,
    target_samples: int,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    logging.info("Recording '%s' (%d samples)...", label, target_samples)
    start = time.time()

    while len(rows) < target_samples:
        parsed = stream.read_parsed()
        if parsed is None:
            continue

        parsed["label"] = label
        rows.append(parsed)

    duration = time.time() - start
    logging.info("Captured %d samples in %.2fs", len(rows), duration)
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect gesture data from Arduino serial")
    parser.add_argument("--port", default=None, help="Serial port (auto-detect if omitted)")
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baud rate")
    parser.add_argument("--samples", type=int, default=40, help="Samples per recording")
    parser.add_argument("--data-dir", default="data", help="Directory for CSV files")
    parser.add_argument(
        "--output-file",
        default="simulated_gestures.csv",
        help="Consolidated CSV filename to store all real gesture rows",
    )
    parser.add_argument("--label", default=None, help="Optional fixed label for non-interactive runs")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    setup_logging()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    output_file = data_dir / args.output_file

    with ArduinoSerialStream(port=args.port, baudrate=args.baudrate) as stream:
        while True:
            label = args.label
            if label is None:
                label = input("Enter gesture label (or 'q' to quit): ").strip()
            if not label:
                if args.label is not None:
                    break
                continue
            if label.lower() in {"q", "quit", "exit"}:
                break

            normalized = sanitize_label(label)

            input(f"Press Enter and perform '{normalized}' gesture...")
            rows = collect_samples(stream=stream, label=normalized, target_samples=args.samples)
            written = append_rows_to_csv(output_file, rows)
            logging.info("Saved %d rows for label '%s' to %s", written, normalized, output_file)

            if args.label is not None:
                break


if __name__ == "__main__":
    main()
