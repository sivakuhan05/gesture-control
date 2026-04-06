import cv2
import mediapipe as mp
import csv
import argparse
from pathlib import Path
from KNN.feature_utils import extract_normalized_landmark_features

mp_hands = mp.solutions.hands
ALLOWED_LABELS = ["open", "close", "pointer", "ok"]

# file to store data
file_name = Path(__file__).resolve().parent / "gesture_data.csv"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect hand landmark data for one gesture label."
    )
    parser.add_argument("label", choices=ALLOWED_LABELS, help="Gesture label to record.")
    return parser.parse_args()


def main():
    args = parse_args()
    label = args.label

    hands = mp_hands.Hands(max_num_hands=1)
    cap = cv2.VideoCapture(0)

    try:
        with open(file_name, mode='a', newline='') as f:
            writer = csv.writer(f)

            while True:
                success, frame = cap.read()
                if not success:
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                result = hands.process(rgb)

                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        row = extract_normalized_landmark_features(hand_landmarks)
                        row.append(label)
                        writer.writerow(row)

                        cv2.putText(frame, f"Recording: {label}", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("Collect Data", frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
