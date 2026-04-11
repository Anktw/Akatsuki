# Akatsuki(Team name)
# SignSense(Project name)
Vihan 9.0 Project

## Real-Time Sign Language to Speech (Arduino + Python)

This project converts live Arduino glove sensor data into:

Gesture -> Text -> Speech

## Sensor Input Format

Arduino sends one CSV line per sample at 115200 baud:

timestamp,f0,f1,f2,f3,f4

- f0-f4: flex sensors


## Project Structure

```text
hardware/
├── data/
├── models/
├── collect_data.py
├── train_model.py
├── realtime_predict.py
├── tts.py
├── utils.py
├── api.py
├── automate.py
├── gesture_ui.py
└── requirements.txt
```

## 1) Install Dependencies

```bash
pip install -r requirements.txt
```

## 2) Collect Gesture Data

Collect 40 samples (~2 sec) per gesture capture.

```bash
python collect_data.py --port COM3 --baudrate 115200 --samples 40
```

Port can be auto-detected now, so this also works:

```bash
python collect_data.py --baudrate 115200 --samples 40
```

You will be prompted for a gesture label repeatedly. Data is appended to:

- data/<label>.csv

## 3) Train Model

Uses sliding windows with:

- window size = 40
- step size = 20

```bash
python train_model.py --data-dir data --model-dir models --window-size 40 --step-size 20
```

Outputs:

- models/gesture_model.pkl
- models/label_encoder.pkl

## 4) Real-Time Prediction + Speech

```bash
python realtime_predict.py --port COM3 --baudrate 115200
```

Auto-detect port version:

```bash
python realtime_predict.py --baudrate 115200
```

Features included:

- rolling buffer of 40 samples
- smoothing with majority vote of last 5 predictions
- confidence display (if model supports `predict_proba`)
- idle/noise filtering
- offline TTS with debounce (prevents repeating same word continuously)

Optional flags:

```bash
python realtime_predict.py --port COM3 --disable-tts
python realtime_predict.py --port COM3 --min-confidence 0.55
```

## 5) Optional FastAPI Backend

Run API server:

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Predict endpoint:

- POST /predict
- Body: { "window": [[...11 values...], ...] }

Example request body shape should match your training window size (default 40x11).

## 6) Automation Script (Single Entry Point)

Use one launcher for all actions:

```bash
python automate.py collect --baudrate 115200 --samples 40
python automate.py train --data-dir data --model-dir models
python automate.py predict --baudrate 115200
python automate.py api --reload
python automate.py ui
python automate.py runtime-ui
```

## 7) Gesture UI (Letters, Numbers, Space, Custom)

Launch UI:

```bash
python automate.py ui
```

or:

```bash
streamlit run gesture_ui.py
```

UI features:

- Auto/manual COM port selection
- Add gestures using presets (A-Z, 0-9, SPACE) or custom labels
- Collect repeated samples with one click
- Train model from collected dataset
- Live prediction preview with optional speech

## 8) Continuous Runtime Display (New Page)

Start dedicated runtime page:

```bash
python automate.py runtime-ui
```

Open `http://localhost:8502`.

Features:

- Continuous prediction (no fixed runtime)
- Large word display
- Large first-letter display
- Optional speech output
- Start/Stop controls

## Notes for Best Accuracy

- Collect multiple sessions per gesture.
- Include an `idle` label to reduce false positives.
- Keep sampling rate consistent between collection and prediction.
- Use balanced data across gesture classes.
