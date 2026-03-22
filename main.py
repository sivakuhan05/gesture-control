import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmark list
            landmarks = hand_landmarks.landmark

            # Get y values (IMPORTANT)
            thumb_up = landmarks[4].y < landmarks[3].y

            index_down = landmarks[8].y > landmarks[6].y
            middle_down = landmarks[12].y > landmarks[10].y
            ring_down = landmarks[16].y > landmarks[14].y
            pinky_down = landmarks[20].y > landmarks[18].y

            # Check thumbs up
            if thumb_up and index_down and middle_down and ring_down and pinky_down:
                cv2.putText(frame, "THUMBS UP 👍", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
