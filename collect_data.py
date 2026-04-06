import cv2
import mediapipe as mp
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)

# CHANGE THIS EACH TIME YOU COLLECT DATA
LABEL = "fist"   # thumbs_up / point / fist

# file to store data
file_name = "gesture_data.csv"

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

                # extract landmarks
                row = []
                for lm in hand_landmarks.landmark:
                    row.append(lm.x)
                    row.append(lm.y)

                row.append(LABEL)

                writer.writerow(row)

                cv2.putText(frame, f"Recording: {LABEL}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Collect Data", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
