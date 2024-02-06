import cv2
import mediapipe as mp
import numpy as np
import winsound  # This module is specific to Windows

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def find_hand_center(hand_landmarks, image_width, image_height):
    if hand_landmarks:
        x = [landmark.x for landmark in hand_landmarks.landmark]
        y = [landmark.y for landmark in hand_landmarks.landmark]
        center = np.array([np.mean(x) * image_width, np.mean(y) * image_height]).astype(int)
        return center
    return None

def find_red_object_center(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0,120,70])
    upper_red = np.array([10,255,255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170,120,70])
    upper_red = np.array([180,255,255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    mask = mask1 + mask2
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] != 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            return center
    return None

def emit_sound_based_on_distance(distance, max_distance=300):
    if distance is not None:
        # Frequency will be higher as the distance gets smaller
        frequency = int(2000 - (distance / max_distance) * 1500)  # Clamp frequency to a range
        frequency = max(200, min(2000, frequency))  # Keep frequency within 200Hz to 2000Hz
        winsound.Beep(frequency, 100)  # Beep for 100 milliseconds

with mp_hands.Hands(static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.5) as hands:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        image_height, image_width, _ = image.shape
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                hand_center = find_hand_center(hand_landmarks, image_width, image_height)
                red_object_center = find_red_object_center(image)

                if hand_center is not None and red_object_center is not None:
                    distance = np.linalg.norm(np.array(hand_center) - np.array(red_object_center)) - 30
                    cv2.putText(image, f"Distance: {int(distance)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    emit_sound_based_on_distance(distance)

        cv2.imshow('MediaPipe Hands', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
