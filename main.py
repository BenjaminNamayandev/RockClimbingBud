# Necessary Imports
import cv2
import mediapipe as mp
import numpy as np
import pyaudio
import pyttsx3
from threading import Thread

# Prepare drawing utilities and pose estimation from MediaPipe.
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize text-to-speech engine.
tts_engine = pyttsx3.init()

# Initialize sound engine.
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=44100, output=True)

locked_wrist_name = None  # Track the currently locked limb
last_locked_wrist_name = None # Last locked wrist name

def async_speak(text):
    def run():
        tts_engine.say(text)
        tts_engine.runAndWait()
    Thread(target=run).start()
    

# This function identifies the positions of the user's wrists and ankles in the video frame.
def find_extremities_centers(pose_landmarks, image_width, image_height):
    extremities_centers = []
    if pose_landmarks:
        # Loop through both wrists and ankles using their specific IDs.
        for extremity_id in [mp_pose.PoseLandmark.LEFT_INDEX, mp_pose.PoseLandmark.RIGHT_INDEX, mp_pose.PoseLandmark.LEFT_HEEL, mp_pose.PoseLandmark.RIGHT_HEEL]:
            extremity = pose_landmarks.landmark[extremity_id]
            # Calculate the extremity's center point on the screen.
            center = np.array([extremity.x * image_width, extremity.y * image_height]).astype(int)
            extremities_centers.append(center)
    return extremities_centers

# Global list to keep track of boxes
user_defined_boxes = []

# Callback function to update boxes on mouse click
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        box_size = 10  # You can adjust the size as needed
        box = (x - box_size // 2, y - box_size // 2, box_size, box_size)
        user_defined_boxes.append(((x, y), box))  # Append center and box

# Register callback
cv2.namedWindow('MediaPipe Pose')
cv2.setMouseCallback('MediaPipe Pose', mouse_callback)

# This function generates a sound with properties based on the distance to an object.
def emit_sound_based_on_distance(distance, max_distance=500):
    if distance is not None:
        frequency = int(2500 - (distance / max_distance) * 2000)
        frequency = max(300, min(3000, frequency))
        # Output a tone
        duration = 0.1  # in seconds
        samplerate = 44100  # in Hz
        samples = (np.sin(2 * np.pi * np.arange(samplerate * duration) * frequency / samplerate)).astype(np.float32) * 0.1
        stream.write(samples.tobytes())

# This function identifies the closest user-defined object to the user's wrist that is also above the wrist.
def find_closest_object(hand_center, objects_centers):
    if not objects_centers or hand_center is None:
        return None, float('inf')
    # Filter objects that are above the wrist's y-coordinate.
    filtered_objects_centers = [(center, box) for center, box in objects_centers if center[1] < hand_center[1]]
    if not filtered_objects_centers:
        return None, float('inf')
    distances = [np.linalg.norm(np.array(hand_center) - np.array(center)) for center, _ in filtered_objects_centers]
    min_distance_index = np.argmin(distances)
    return filtered_objects_centers[min_distance_index], distances[min_distance_index]

# This function checks if the wrist is in proximity to any user-defined objects, indicating a touch.
def check_for_touch(wrist_center, objects_centers_and_boxes, touch_threshold=50):
    for i, (center, _) in enumerate(objects_centers_and_boxes):
        distance = np.linalg.norm(np.array(wrist_center) - np.array(center))
        if distance <= touch_threshold:
            return True, i  # Return True and the index of the touched object
    return False, None  # No touch detected

with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.6, model_complexity=0) as pose:
    # Initialize webcam.
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) 
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  
    frame_skip = 1
    frame_count = 0

    previous_closest_wrist_name = None
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
        closest_wrist_name = None

        if results.pose_landmarks:
            wrist_centers = find_extremities_centers(results.pose_landmarks, image_width, image_height)
            wrist_names = ['Right Hand', 'Left Hand', 'Right Foot', 'Left Foot']

            for wrist_center, wrist_name in zip(wrist_centers, wrist_names):
                if locked_wrist_name is None or wrist_name == locked_wrist_name:
                    closest_object, distance_to_closest = find_closest_object(wrist_center, user_defined_boxes)
                    if closest_object and distance_to_closest < overall_closest_distance:
                        overall_closest_distance = distance_to_closest
                        overall_closest_pair = (wrist_center, closest_object)
                        closest_wrist_name = wrist_name

            if closest_wrist_name and (locked_wrist_name is None or closest_wrist_name == locked_wrist_name):
                wrist_center, closest_object = overall_closest_pair
                user_defined_object_center, user_defined_object_box = closest_object
                cv2.line(image, wrist_center, user_defined_object_center, (0, 255, 0), 3)
                emit_sound_based_on_distance(overall_closest_distance)

                touched, touched_index = check_for_touch(wrist_center, user_defined_boxes, touch_threshold=50)
                if touched:
                    user_defined_boxes.pop(touched_index)
                    # Update the condition here to ensure the locked limb changes after a touch
                    if locked_wrist_name is not None:
                        last_locked_wrist_name = locked_wrist_name
                    locked_wrist_name = None  # Unlock the limb after a touch
                else:
                    if locked_wrist_name is None:
                        # Choose a different limb if the current choice is the same as the last locked limb
                        eligible_wrist_names = [name for name in wrist_names if name != last_locked_wrist_name]
                        if closest_wrist_name in eligible_wrist_names:
                            locked_wrist_name = closest_wrist_name
                            async_speak(f"{closest_wrist_name}")
                        elif eligible_wrist_names:
                            # Optionally handle the case where the closest wrist is the same as the last locked one
                            locked_wrist_name = eligible_wrist_names[0]
                            async_speak(f"{eligible_wrist_names[0]}")

                cv2.putText(image, f"{closest_wrist_name} is closer", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            for center, box in user_defined_boxes:
                x, y, w, h = box
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('MediaPipe Pose', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break
        frame_count += 1

cap.release()
cv2.destroyAllWindows()
stream.stop_stream()
stream.close()
p.terminate()