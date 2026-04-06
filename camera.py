import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands

hands = None
cap = None


def _ensure_camera():
    global hands, cap
    if hands is None:
        hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
    if cap is None:
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return False
    return True


def get_frame():
    if not _ensure_camera():
        return None, None

    success, frame = cap.read()
    if not success:
        return None, None

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)
    return frame, result


def cleanup():
    global hands, cap
    if cap is not None:
        cap.release()
        cap = None
    if hands is not None:
        hands.close()
        hands = None
