from __future__ import annotations

import argparse
from pathlib import Path
from collections import Counter
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from utils import SENSOR_COLUMNS, ensure_dir, flatten_window


def load_dataset(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    frames: List[pd.DataFrame] = []
    for file in files:
        df = pd.read_csv(file)
        required = set(["timestamp", *SENSOR_COLUMNS, "label"])
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{file} missing columns: {sorted(missing)}")
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=[*SENSOR_COLUMNS, "label"])
    return data


def load_dataset_file(data_file: Path) -> pd.DataFrame:
    if not data_file.exists():
        raise FileNotFoundError(f"CSV file not found: {data_file}")

    df = pd.read_csv(data_file)
    required = set(["timestamp", *SENSOR_COLUMNS, "label"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{data_file} missing columns: {sorted(missing)}")

    df = df.dropna(subset=[*SENSOR_COLUMNS, "label"])
    return df


def sliding_windows(
    data: pd.DataFrame,
    window_size: int,
    step_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    X: List[np.ndarray] = []
    y: List[str] = []

    for label, group in data.groupby("label"):
        group = group.reset_index(drop=True)
        values = group[SENSOR_COLUMNS].to_numpy(dtype=np.float32)

        for start in range(0, len(values) - window_size + 1, step_size):
            window = values[start : start + window_size]
            X.append(flatten_window(window))
            y.append(label)

    if not X:
        raise ValueError("No windows generated. Collect more data or lower window size.")

    return np.vstack(X), np.asarray(y)


def validate_trainable_labels(y_labels: np.ndarray) -> None:
    counts = Counter(y_labels.tolist())
    too_small = {label: count for label, count in counts.items() if count < 2}
    if too_small:
        details = ", ".join(f"{label}={count}" for label, count in sorted(too_small.items()))
        raise ValueError(
            "Not enough data to train. Each gesture class needs at least 2 window samples, "
            f"but these classes are too small: {details}. Collect more recordings for each gesture."
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train gesture classifier")
    parser.add_argument("--data-dir", default="data", help="Directory containing gesture CSV files")
    parser.add_argument("--data-file", default=None, help="Optional single CSV file to train from")
    parser.add_argument("--model-dir", default="models", help="Directory to save model artifacts")
    parser.add_argument("--window-size", type=int, default=40, help="Sliding window size")
    parser.add_argument("--step-size", type=int, default=20, help="Sliding window step size")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    ensure_dir(model_dir)

    if args.data_file:
        data = load_dataset_file(Path(args.data_file))
    else:
        data = load_dataset(data_dir)
    X, y_labels = sliding_windows(data, args.window_size, args.step_size)
    validate_trainable_labels(y_labels)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_labels)

    stratify = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=args.random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    model_path = model_dir / "gesture_model.pkl"
    encoder_path = model_dir / "label_encoder.pkl"
    joblib.dump(model, model_path)
    joblib.dump(label_encoder, encoder_path)

    print(f"Saved model to: {model_path}")
    print(f"Saved label encoder to: {encoder_path}")

if __name__ == "__main__":
    main()
