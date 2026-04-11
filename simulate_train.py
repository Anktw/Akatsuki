from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


SENSOR_COLUMNS: List[str] = [
    "f0",
    "f1",
    "f2",
    "f3",
    "f4",
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz",
]

ALL_COLUMNS: List[str] = ["timestamp", *SENSOR_COLUMNS, "label"]


@dataclass(frozen=True)
class GesturePattern:
    name: str
    flex_base: np.ndarray
    flex_amp: np.ndarray
    imu_base: np.ndarray
    imu_amp: np.ndarray
    frequency: float
    phase: float


def create_gesture_patterns() -> List[GesturePattern]:
    # Distinct templates per gesture so the model can learn meaningful differences.
    return [
        GesturePattern(
            name="HELLO",
            flex_base=np.array([330, 360, 380, 350, 340], dtype=np.float32),
            flex_amp=np.array([80, 90, 85, 75, 80], dtype=np.float32),
            imu_base=np.array([0.2, 0.1, 9.3, 2.0, 55.0, 8.0], dtype=np.float32),
            imu_amp=np.array([0.8, 1.0, 0.6, 35.0, 25.0, 20.0], dtype=np.float32),
            frequency=1.35,
            phase=0.0,
        ),
        GesturePattern(
            name="YES",
            flex_base=np.array([520, 540, 560, 540, 510], dtype=np.float32),
            flex_amp=np.array([45, 40, 50, 45, 40], dtype=np.float32),
            imu_base=np.array([0.0, -0.1, 9.6, 45.0, 4.0, 6.0], dtype=np.float32),
            imu_amp=np.array([0.6, 0.5, 0.5, 20.0, 10.0, 8.0], dtype=np.float32),
            frequency=2.1,
            phase=0.4,
        ),
        GesturePattern(
            name="NO",
            flex_base=np.array([430, 460, 470, 440, 420], dtype=np.float32),
            flex_amp=np.array([35, 32, 30, 28, 34], dtype=np.float32),
            imu_base=np.array([0.1, 0.0, 9.5, 3.0, 50.0, 4.0], dtype=np.float32),
            imu_amp=np.array([0.7, 0.7, 0.4, 12.0, 35.0, 10.0], dtype=np.float32),
            frequency=1.9,
            phase=1.1,
        ),
        GesturePattern(
            name="STOP",
            flex_base=np.array([250, 260, 270, 265, 255], dtype=np.float32),
            flex_amp=np.array([18, 15, 15, 14, 16], dtype=np.float32),
            imu_base=np.array([0.0, 0.0, 9.7, 1.5, 1.2, 1.0], dtype=np.float32),
            imu_amp=np.array([0.2, 0.2, 0.2, 2.0, 2.0, 2.0], dtype=np.float32),
            frequency=0.5,
            phase=2.0,
        ),
        GesturePattern(
            name="THANKS",
            flex_base=np.array([300, 420, 430, 380, 320], dtype=np.float32),
            flex_amp=np.array([50, 75, 70, 65, 45], dtype=np.float32),
            imu_base=np.array([0.3, -0.2, 9.2, 30.0, 14.0, 12.0], dtype=np.float32),
            imu_amp=np.array([1.0, 0.8, 0.7, 18.0, 14.0, 10.0], dtype=np.float32),
            frequency=1.2,
            phase=0.8,
        ),
        GesturePattern(
            name="WATER",
            flex_base=np.array([390, 410, 470, 500, 420], dtype=np.float32),
            flex_amp=np.array([30, 35, 65, 70, 40], dtype=np.float32),
            imu_base=np.array([-0.2, 0.2, 9.4, 20.0, 8.0, 30.0], dtype=np.float32),
            imu_amp=np.array([0.8, 0.9, 0.5, 15.0, 12.0, 22.0], dtype=np.float32),
            frequency=1.55,
            phase=1.4,
        ),
    ]


def flatten_window(window: np.ndarray) -> np.ndarray:
    if window.ndim != 2:
        raise ValueError("Window must be 2D: [timesteps, features]")
    return window.astype(np.float32, copy=False).reshape(-1)


def generate_single_timeseries(
    pattern: GesturePattern,
    timesteps: int,
    dt: float,
    noise_std_flex: float,
    noise_std_imu: float,
    rng: np.random.Generator,
) -> np.ndarray:
    t = np.arange(timesteps, dtype=np.float32) * dt
    wave_1 = np.sin(2.0 * np.pi * pattern.frequency * t + pattern.phase)
    wave_2 = np.sin(2.0 * np.pi * (pattern.frequency * 0.5) * t + (pattern.phase * 0.7))

    flex = pattern.flex_base + (pattern.flex_amp * wave_1[:, None])
    imu = pattern.imu_base + (pattern.imu_amp * wave_2[:, None])

    # Trial-level variation simulates person-to-person or repetition differences.
    flex += rng.normal(0.0, 10.0, size=(1, 5)).astype(np.float32)
    imu += rng.normal(0.0, 0.5, size=(1, 6)).astype(np.float32)

    flex += rng.normal(0.0, noise_std_flex, size=flex.shape).astype(np.float32)
    imu += rng.normal(0.0, noise_std_imu, size=imu.shape).astype(np.float32)

    flex = np.clip(flex, 0.0, 1023.0)
    data = np.concatenate([flex, imu], axis=1)
    return data


def generate_simulated_dataframe(
    patterns: Sequence[GesturePattern],
    samples_per_gesture: int,
    timesteps: int,
    dt: float,
    noise_std_flex: float,
    noise_std_imu: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, float | str]] = []
    global_time = 0.0

    for pattern in patterns:
        for _ in range(samples_per_gesture):
            ts_data = generate_single_timeseries(
                pattern=pattern,
                timesteps=timesteps,
                dt=dt,
                noise_std_flex=noise_std_flex,
                noise_std_imu=noise_std_imu,
                rng=rng,
            )

            for step_idx in range(timesteps):
                row: Dict[str, float | str] = {
                    "timestamp": float(global_time + (step_idx * dt)),
                    "label": pattern.name,
                }
                for col_idx, col_name in enumerate(SENSOR_COLUMNS):
                    row[col_name] = float(ts_data[step_idx, col_idx])
                rows.append(row)

            global_time += timesteps * dt

    df = pd.DataFrame(rows, columns=ALL_COLUMNS)
    return df


def make_windows(
    df: pd.DataFrame,
    timesteps: int,
    window_size: int,
    step_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    X: List[np.ndarray] = []
    y: List[str] = []

    for label, group in df.groupby("label"):
        values = group[SENSOR_COLUMNS].to_numpy(dtype=np.float32)

        # Reconstruct each full 40-step gesture sample before sliding.
        sample_count = len(values) // timesteps
        for sample_idx in range(sample_count):
            start = sample_idx * timesteps
            end = start + timesteps
            sample = values[start:end]
            if len(sample) < window_size:
                continue

            for w_start in range(0, len(sample) - window_size + 1, step_size):
                window = sample[w_start : w_start + window_size]
                X.append(flatten_window(window))
                y.append(label)

    if not X:
        raise ValueError("No sliding windows generated. Check window/timestep settings.")

    return np.vstack(X), np.asarray(y)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train gesture classifier with synthetic Arduino-like sensor data"
    )
    parser.add_argument("--samples-per-gesture", type=int, default=120)
    parser.add_argument("--timesteps", type=int, default=40)
    parser.add_argument("--window-size", type=int, default=40)
    parser.add_argument("--step-size", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.05, help="Seconds per timestep")
    parser.add_argument("--noise-std-flex", type=float, default=8.0)
    parser.add_argument("--noise-std-imu", type=float, default=0.8)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--save-sim-data", action="store_true")
    parser.add_argument("--sim-data-path", default="data/synthetic_gestures.csv")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    patterns = create_gesture_patterns()

    df = generate_simulated_dataframe(
        patterns=patterns,
        samples_per_gesture=args.samples_per_gesture,
        timesteps=args.timesteps,
        dt=args.dt,
        noise_std_flex=args.noise_std_flex,
        noise_std_imu=args.noise_std_imu,
        seed=args.random_state,
    )

    if args.save_sim_data:
        sim_path = Path(args.sim_data_path)
        sim_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(sim_path, index=False)
        print(f"Saved synthetic dataset to: {sim_path}")

    X, y_labels = make_windows(
        df=df,
        timesteps=args.timesteps,
        window_size=args.window_size,
        step_size=args.step_size,
    )

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
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

    print(f"Generated gestures: {', '.join(p.name for p in patterns)}")
    print(f"Total time-series rows: {len(df)}")
    print(f"Total windowed samples: {len(X)}")
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "gesture_model.pkl"
    encoder_path = model_dir / "label_encoder.pkl"
    joblib.dump(model, model_path)
    joblib.dump(encoder, encoder_path)

    print(f"Saved model to: {model_path}")
    print(f"Saved label encoder to: {encoder_path}")


if __name__ == "__main__":
    main()
