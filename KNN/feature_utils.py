def extract_normalized_landmark_features(hand_landmarks):
    points = [[lm.x, lm.y] for lm in hand_landmarks.landmark]

    base_x, base_y = points[0]
    flattened = []
    for x, y in points:
        flattened.append(x - base_x)
        flattened.append(y - base_y)

    max_value = max((abs(v) for v in flattened), default=1.0)
    if max_value == 0:
        max_value = 1.0

    return [v / max_value for v in flattened]
