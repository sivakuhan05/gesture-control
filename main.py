import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)

gesture_counter = 0
gesture_threshold = 1
current_gesture = None

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    detected_gesture = None

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            lm = hand_landmarks.landmark

            # Finger states
            thumb_up = lm[4].y < lm[3].y

            index_up = lm[8].y < lm[6].y
            middle_up = lm[12].y < lm[10].y
            ring_up = lm[16].y < lm[14].y
            pinky_up = lm[20].y < lm[18].y

            index_down = not index_up
            middle_down = not middle_up
            ring_down = not ring_up
            pinky_down = not pinky_up

            # ---- Gesture Detection ----

            # 👍 Thumbs Up
            if thumb_up and index_down and middle_down and ring_down and pinky_down:
                detected_gesture = "THUMBS UP 👍"

            # ✋ Open Palm
            elif thumb_up and index_up and middle_up and ring_up and pinky_up:
                detected_gesture = "OPEN PALM ✋"

            # 👉 Pointing
            elif index_up and middle_down and ring_down and pinky_down:
                detected_gesture = "POINT 👉"

    # Stability logic
    if detected_gesture == current_gesture and detected_gesture is not None:
        gesture_counter += 1
    else:
        gesture_counter = 0
        current_gesture = detected_gesture

    # Show gesture
    if gesture_counter >= gesture_threshold and current_gesture:
        cv2.putText(frame, current_gesture, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
