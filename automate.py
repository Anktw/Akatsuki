from __future__ import annotations

import argparse
import subprocess
import sys
from typing import List


def run_command(cmd: List[str]) -> int:
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    return result.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Automation launcher for sign-language pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect", help="Run data collection")
    collect.add_argument("--port", default=None)
    collect.add_argument("--baudrate", type=int, default=115200)
    collect.add_argument("--samples", type=int, default=40)
    collect.add_argument("--data-dir", default="data")
    collect.add_argument("--label", default=None)

    train = sub.add_parser("train", help="Run model training")
    train.add_argument("--data-dir", default="data")
    train.add_argument("--model-dir", default="models")
    train.add_argument("--window-size", type=int, default=40)
    train.add_argument("--step-size", type=int, default=20)
    train.add_argument("--test-size", type=float, default=0.2)

    predict = sub.add_parser("predict", help="Run real-time prediction")
    predict.add_argument("--port", default=None)
    predict.add_argument("--baudrate", type=int, default=115200)
    predict.add_argument("--model-path", default="models/gesture_model.pkl")
    predict.add_argument("--encoder-path", default="models/label_encoder.pkl")
    predict.add_argument("--disable-tts", action="store_true")
    predict.add_argument("--min-confidence", type=float, default=0.45)

    api = sub.add_parser("api", help="Run FastAPI server")
    api.add_argument("--host", default="0.0.0.0")
    api.add_argument("--port", type=int, default=8000)
    api.add_argument("--reload", action="store_true")

    sub.add_parser("ui", help="Run Streamlit UI")
    runtime_ui = sub.add_parser("runtime-ui", help="Run continuous runtime display UI")
    runtime_ui.add_argument("--port", type=int, default=8502)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    py = sys.executable

    if args.command == "collect":
        cmd = [
            py,
            "collect_data.py",
            "--baudrate",
            str(args.baudrate),
            "--samples",
            str(args.samples),
            "--data-dir",
            args.data_dir,
        ]
        if args.port:
            cmd.extend(["--port", args.port])
        if args.label:
            cmd.extend(["--label", args.label])
        raise SystemExit(run_command(cmd))

    if args.command == "train":
        cmd = [
            py,
            "train_model.py",
            "--data-dir",
            args.data_dir,
            "--model-dir",
            args.model_dir,
            "--window-size",
            str(args.window_size),
            "--step-size",
            str(args.step_size),
            "--test-size",
            str(args.test_size),
        ]
        raise SystemExit(run_command(cmd))

    if args.command == "predict":
        cmd = [
            py,
            "realtime_predict.py",
            "--baudrate",
            str(args.baudrate),
            "--model-path",
            args.model_path,
            "--encoder-path",
            args.encoder_path,
            "--min-confidence",
            str(args.min_confidence),
        ]
        if args.port:
            cmd.extend(["--port", args.port])
        if args.disable_tts:
            cmd.append("--disable-tts")
        raise SystemExit(run_command(cmd))

    if args.command == "api":
        cmd = [py, "-m", "uvicorn", "api:app", "--host", args.host, "--port", str(args.port)]
        if args.reload:
            cmd.append("--reload")
        raise SystemExit(run_command(cmd))

    if args.command == "ui":
        cmd = [py, "-m", "streamlit", "run", "gesture_ui.py"]
        raise SystemExit(run_command(cmd))

    if args.command == "runtime-ui":
        cmd = [
            py,
            "-m",
            "streamlit",
            "run",
            "runtime_ui.py",
            "--server.port",
            str(args.port),
        ]
        raise SystemExit(run_command(cmd))


if __name__ == "__main__":
    main()
