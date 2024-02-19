# Importing the required libraries
import cv2
import mediapipe as mp
import numpy as np
import winsound  # This module is specific to Windows

# Setting up position estimation
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function that finds where the user's wrists are located
def find_wrist_centers(pose_landmarks, image_width, image_height):
    wrist_centers = []
    if pose_landmarks:
        for wrist_id in [mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST]:
            wrist = pose_landmarks.landmark[wrist_id]
            center = np.array([wrist.x * image_width, wrist.y * image_height]).astype(int)
            wrist_centers.append(center)
    return wrist_centers

# Apply color masking to isolate for a specific color, then find objects on screen that have that specific color
def find_red_objects_centers_and_boxes(img, min_area=500):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0,120,70])
    upper_red = np.array([10,255,255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170,120,70])
    upper_red = np.array([180,255,255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    mask = mask1 + mask2
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    centers_boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:  # Filter out small contours
            M = cv2.moments(contour)
            if M["m00"] != 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                rect = cv2.boundingRect(contour)
                if rect[2] * rect[3] >= min_area:  # Optional: Check if bounding box area meets the requirement
                    centers_boxes.append((center, rect))
    return centers_boxes


# Take into account distance from users' hands to the location of found red objects and correlate that to the frequency of a beep
def emit_sound_based_on_distance(distance, max_distance=500):  # Increased max_distance for a wider range
    if distance is not None:
        # Adjust the frequency range to be wider
        frequency = int(2500 - (distance / max_distance) * 2000)  # Frequency from 500Hz to 2500Hz
        frequency = max(500, min(2500, frequency))  # Clamp frequency within the new range

        # Adjust duration based on distance: farther away = longer duration
        duration = int(100 + (distance / max_distance) * 100)  # Duration from 100ms to 200ms
        duration = max(100, min(200, duration))

        try:
            winsound.Beep(frequency, duration)
        except RuntimeError as e:
            print(f"Error playing sound: {e}")
        
# This function finds the closest red object to the users wrist
def find_closest_object(hand_center, objects_centers):
    if not objects_centers or hand_center is None:
        return None, float('inf')
    distances = [np.linalg.norm(np.array(hand_center) - np.array(center)) for center, _ in objects_centers]
    min_distance_index = np.argmin(distances)
    return objects_centers[min_distance_index], distances[min_distance_index]

# This function checks if a wrist is touching any red object
def check_for_touch(wrist_center, objects_centers_and_boxes, touch_threshold=50):
    touched_objects_indexes = []
    for i, (center, _) in enumerate(objects_centers_and_boxes):
        distance = np.linalg.norm(np.array(wrist_center) - np.array(center))
        if distance <= touch_threshold:
            touched_objects_indexes.append(i)
    return touched_objects_indexes

# Ensure to define all required functions and setup from the original code snippet before this loop

# Ensure to define all required functions and setup from the original code snippet before this loop

with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    frame_skip = 5
    frame_count = 0

    success, initial_image = cap.read()
    if success:
        initial_image = cv2.flip(initial_image, 1)
        red_objects_centers_and_boxes = find_red_objects_centers_and_boxes(initial_image)

    while cap.isOpened():
        success, image = cap.read()
        if not success or frame_count % frame_skip != 0:
            frame_count += 1
            continue

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        image_height, image_width, _ = image.shape

        overall_closest_distance = float('inf')
        overall_closest_pair = None

        if results.pose_landmarks:
            wrist_centers = find_wrist_centers(results.pose_landmarks, image_width, image_height)
            for wrist_center in wrist_centers:
                closest_object, distance_to_closest = find_closest_object(wrist_center, red_objects_centers_and_boxes)
                if closest_object and distance_to_closest < overall_closest_distance:
                    overall_closest_distance = distance_to_closest
                    overall_closest_pair = (wrist_center, closest_object)

            if overall_closest_pair:
                wrist_center, closest_object = overall_closest_pair
                red_object_center, red_object_box = closest_object
                cv2.line(image, wrist_center, red_object_center, (0, 255, 0), 3)
                emit_sound_based_on_distance(overall_closest_distance)

                # Check for touch with the closest object and remove if touched
                if overall_closest_distance <= 50:  # Assuming 50 is the touch threshold
                    red_objects_centers_and_boxes.remove(closest_object)

        # Draw remaining red objects
        for _, red_object_box in red_objects_centers_and_boxes:
            x, y, w, h = red_object_box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('MediaPipe Pose', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break
        frame_count += 1

cap.release()
cv2.destroyAllWindows()
