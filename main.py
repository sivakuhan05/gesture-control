import cv2
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)

# Gesture smoothing
gesture_counter = 0
gesture_threshold = 1  # almost instant

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    detected = False

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark

            # ---- Looser Thumbs Up Detection ----
            thumb_up = landmarks[4].y < landmarks[3].y

            index_down = landmarks[8].y > landmarks[6].y
            middle_down = landmarks[12].y > landmarks[10].y
            ring_down = landmarks[16].y > landmarks[14].y
            pinky_down = landmarks[20].y > landmarks[18].y

            if thumb_up and index_down and middle_down and ring_down and pinky_down:
                detected = True

    # ---- Stability check ----
    if detected:
        gesture_counter += 1
    else:
        gesture_counter = 0

    # ---- Show gesture ----
    if gesture_counter >= gesture_threshold:
        cv2.putText(frame, "THUMBS UP 👍", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
