# Gesture Control System

Real-time hand gesture recognition using MediaPipe landmarks + a KNN classifier.

## Supported Gestures
- `open`
- `close`
- `pointer`
- `ok`

## Project Structure
- `main.py` - real-time inference app (webcam + on-screen prediction)
- `camera.py` - camera and MediaPipe hand tracking helpers
- `KNN/collect_data.py` - collect labeled landmark samples
- `KNN/train_model.py` - train and evaluate KNN model
- `KNN/feature_utils.py` - shared landmark preprocessing

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run Inference
```bash
python main.py
```
Press `Esc` to quit.

## Collect Training Data
Collect one class at a time:
```bash
python KNN/collect_data.py open
python KNN/collect_data.py close
python KNN/collect_data.py pointer
python KNN/collect_data.py ok
```
Each run appends rows to `KNN/gesture_data.csv`.

## Train Model
```bash
python KNN/train_model.py
```
This writes:
- `KNN/gesture_model.pkl`
- `KNN/scaler.pkl`
- `KNN/labels.pkl`

## Notes
- Keep class counts reasonably balanced for best real-world accuracy.
- Use the same camera setup/lighting for collection and inference whenever possible.

## Dataset Credits
This project uses/derives training data from:
- https://github.com/kinivi/hand-gesture-recognition-mediapipe/tree/main#

Credit to the original repository and contributors for the MediaPipe hand gesture dataset and training resources.
