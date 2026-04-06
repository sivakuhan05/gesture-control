import cv2
from camera import cleanup, get_frame
import mediapipe as mp
import pickle
import numpy as np
from pathlib import Path
from KNN.feature_utils import extract_normalized_landmark_features

MODEL_DIR = Path(__file__).resolve().parent / "KNN"
CONFIDENCE_THRESHOLD = 0.6

def load_artifacts():
    with open(MODEL_DIR / "gesture_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODEL_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    labels = None
    labels_file = MODEL_DIR / "labels.pkl"
    if labels_file.exists():
        with open(labels_file, "rb") as f:
            labels = set(pickle.load(f))
    return model, scaler, labels


def main():
    model, scaler, labels = load_artifacts()
    mp_draw = mp.solutions.drawing_utils

    gesture_counter = 0
    gesture_threshold = 2
    current_gesture = None

    try:
        while True:
            frame, result = get_frame()
            if frame is None:
                break

            detected_gesture = None

            if result and result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
                    )

                    features = extract_normalized_landmark_features(hand_landmarks)
                    features = np.array(features, dtype=np.float32).reshape(1, -1)
                    features = scaler.transform(features)
                    probabilities = model.predict_proba(features)[0]
                    best_index = int(np.argmax(probabilities))
                    best_label = model.classes_[best_index]
                    best_confidence = float(probabilities[best_index])

                    if best_confidence >= CONFIDENCE_THRESHOLD:
                        if labels is None or best_label in labels:
                            detected_gesture = best_label

            if detected_gesture == current_gesture and detected_gesture is not None:
                gesture_counter += 1
            else:
                gesture_counter = 0
                current_gesture = detected_gesture

            if gesture_counter >= gesture_threshold and current_gesture:
                cv2.putText(
                    frame,
                    current_gesture,
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("Gesture Control", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cleanup()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
