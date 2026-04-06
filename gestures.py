def detect_gesture(hand_landmarks):
    lm = hand_landmarks.landmark

    thumb_up = lm[4].y < lm[3].y

    index_up = lm[8].y < lm[6].y
    middle_up = lm[12].y < lm[10].y
    ring_up = lm[16].y < lm[14].y
    pinky_up = lm[20].y < lm[18].y

    index_down = not index_up
    middle_down = not middle_up
    ring_down = not ring_up
    pinky_down = not pinky_up

    if thumb_up and index_down and middle_down and ring_down and pinky_down:
        return "THUMBS UP 👍"

    elif index_up and middle_down and ring_down and pinky_down:
        return "POINT 👉"

    elif index_up and middle_up and ring_down and pinky_down:
        return "PEACE ✌️"

    return None