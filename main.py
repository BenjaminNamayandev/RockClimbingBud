import cv2
import mediapipe as mp
import numpy as np
import winsound 
import pyttsx3

# Prepare drawing utilities and pose estimation from MediaPipe.
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize text-to-speech engine.
tts_engine = pyttsx3.init()

# A function to identify the positions of the user's wrists in the video frame.
def find_wrist_centers(pose_landmarks, image_width, image_height):
    wrist_centers = []
    if pose_landmarks:
        # Loop through both wrists using their specific IDs.
        for wrist_id in [mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST]:
            wrist = pose_landmarks.landmark[wrist_id]
            # Calculate the wrist's center point on the screen.
            center = np.array([wrist.x * image_width, wrist.y * image_height]).astype(int)
            wrist_centers.append(center)
    return wrist_centers

# This function isolates objects of a specific color (red) and identifies their locations.
def find_red_objects_centers_and_boxes(img, min_area=500):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Define ranges for the color red in HSV color space.
    lower_red = np.array([0,120,70])
    upper_red = np.array([10,255,255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170,120,70])
    upper_red = np.array([180,255,255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    mask = mask1 + mask2
    # Find contours within the mask.
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    centers_boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:  # Ignore tiny contours to focus on significant red objects.
            M = cv2.moments(contour)
            if M["m00"] != 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                rect = cv2.boundingRect(contour)
                centers_boxes.append((center, rect))
    return centers_boxes

# This function generates a sound with properties based on the distance to an object.
def emit_sound_based_on_distance(distance, max_distance=500):
    if distance is not None:
        # Convert the distance to a sound frequency and duration.
        frequency = int(2500 - (distance / max_distance) * 2000)
        frequency = max(500, min(2500, frequency))
        duration = int(100 + (distance / max_distance) * 100)
        duration = max(100, min(200, duration))
        try:
            winsound.Beep(frequency, duration)
        except RuntimeError as e:
            print(f"Couldn't play sound: {e}")
        
# Identify the closest red object to the user's wrist.
def find_closest_object(hand_center, objects_centers):
    if not objects_centers or hand_center is None:
        return None, float('inf')
    distances = [np.linalg.norm(np.array(hand_center) - np.array(center)) for center, _ in objects_centers]
    min_distance_index = np.argmin(distances)
    return objects_centers[min_distance_index], distances[min_distance_index]

# Check if the wrist is in proximity to any red objects, indicating a touch.
def check_for_touch(wrist_center, objects_centers_and_boxes, touch_threshold=50):
    touched_objects_indexes = []
    for i, (center, _) in enumerate(objects_centers_and_boxes):
        distance = np.linalg.norm(np.array(wrist_center) - np.array(center))
        if distance <= touch_threshold:
            touched_objects_indexes.append(i)
    return touched_objects_indexes

with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
    # Initialize webcam.
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    frame_skip = 5
    frame_count = 0

    success, initial_image = cap.read()
    if success:
        initial_image = cv2.flip(initial_image, 1)
        # Detect red objects in the initial frame.
        red_objects_centers_and_boxes = find_red_objects_centers_and_boxes(initial_image)

    while cap.isOpened():
        success, image = cap.read()
        if not success or frame_count % frame_skip != 0:
            frame_count += 1
            continue

        image = cv2.flip(image, 1)
        # Convert the image to RGB for pose detection.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        image_height, image_width, _ = image.shape

        overall_closest_distance = float('inf')
        overall_closest_pair = None
        closest_wrist_name = None  # Keep track of which wrist is closest to a red object.

        if results.pose_landmarks:
            wrist_centers = find_wrist_centers(results.pose_landmarks, image_width, image_height)
            wrist_names = ['Right Wrist', 'Left Wrist']

            for wrist_center, wrist_name in zip(wrist_centers, wrist_names):
                closest_object, distance_to_closest = find_closest_object(wrist_center, red_objects_centers_and_boxes)
                if closest_object and distance_to_closest < overall_closest_distance:
                    overall_closest_distance = distance_to_closest
                    overall_closest_pair = (wrist_center, closest_object)
                    closest_wrist_name = wrist_name  # Identify which wrist is closest.

            if overall_closest_pair:
                wrist_center, closest_object = overall_closest_pair
                red_object_center, red_object_box = closest_object
                # Draw a line between the wrist and the closest red object.
                cv2.line(image, wrist_center, red_object_center, (0, 255, 0), 3)
                emit_sound_based_on_distance(overall_closest_distance)

                # Display which wrist is closer.
                cv2.putText(image, f"{closest_wrist_name} is closer", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                # If the wrist is touching a red object, consider removing it from the list.
                if overall_closest_distance <= 50:  # Assuming touch threshold.
                    red_objects_centers_and_boxes.remove(closest_object)

        # Draw all detected red objects.
        for _, red_object_box in red_objects_centers_and_boxes:
            x, y, w, h = red_object_box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('MediaPipe Pose', image)

        if cv2.waitKey(5) & 0xFF == 27:  # Exit on ESC.
            break
        frame_count += 1

cap.release()
cv2.destroyAllWindows()
