# Necessary Imports
import cv2
import mediapipe as mp
import numpy as np
import pyaudio
import pyttsx3
from threading import Thread
from pygame import mixer  # Load the popular external library
import time

mixer.init()
mixer.music.load('bell.mp3')

# Prepare drawing utilities and pose estimation from MediaPipe.
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize text-to-speech engine.
tts_engine = pyttsx3.init()

# Initialize sound engine.
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=44100, output=True)

locked_wrist_name = None  # Track the currently locked limb
last_locked_wrist_name = None  # Last locked wrist name
locked_rock_index = None  # Track the currently locked rock

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
highest_box_y = float('inf')
highest_box_index = None

def update_highest_box():
    global highest_box_y, highest_box_index
    highest_box_y = float('inf')
    highest_box_index = None
    for i, ((x, y), box) in enumerate(user_defined_boxes):
        if y < highest_box_y:
            highest_box_y = y
            highest_box_index = i

# Callback function to update boxes on mouse click
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        box_size = 10  # Adjust the size as needed
        box = (x - box_size // 2, y - box_size // 2, box_size, box_size)
        user_defined_boxes.append(((x, y), box))  # Append center and box
        update_highest_box()

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cv2.namedWindow('MediaPipe Pose')
cv2.setMouseCallback('MediaPipe Pose', mouse_callback)

# Initial Setup Loop
while True:
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)

    for i, (center, box) in enumerate(user_defined_boxes):
        x, y, w, h = box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('MediaPipe Pose', image)
    key = cv2.waitKey(1)

    if key == 32:  # Spacebar pressed
        break
    elif key == 27:  # ESC key to exit early
        cap.release()
        cv2.destroyAllWindows()
        exit()


# This function generates a sound with properties based on the distance to an object.
def emit_sound_based_on_distance(distance, max_distance=500):
    if distance is not None:
        frequency = int(2500 - (distance / max_distance) * 2000)
        frequency = max(300, min(3000, frequency))
        # Output a tone
        duration = 0.1  # in seconds
        samplerate = 44100  # in Hz
        samples = (np.sin(2 * np.pi * np.arange(samplerate * duration) * frequency / samplerate)).astype(np.float32) * 0.01
        stream.write(samples.tobytes())

# This function identifies the closest user-defined object to the user's wrist that is also above the wrist.
def find_closest_object(hand_center, objects_centers):
    if not objects_centers or hand_center is None:
        return None, float('inf'), None
    # Filter objects that are above the wrist's y-coordinate.
    filtered_objects_centers = [(i, center, box) for i, (center, box) in enumerate(objects_centers) if center[1] < hand_center[1]]
    if not filtered_objects_centers:
        return None, float('inf'), None
    distances = [np.linalg.norm(np.array(hand_center) - np.array(center)) for _, center, _ in filtered_objects_centers]
    min_distance_index = np.argmin(distances)
    closest_object_index, closest_object_center, _ = filtered_objects_centers[min_distance_index]
    return closest_object_center, distances[min_distance_index], closest_object_index

# This function checks if the wrist is in proximity to any user-defined objects, indicating a touch.
def check_for_touch(wrist_center, objects_centers_and_boxes, touchThreshold=20):
    for i, (center, _) in enumerate(objects_centers_and_boxes):
        distance = np.linalg.norm(np.array(wrist_center) - np.array(center))
        if distance <= touchThreshold:
            return True, i  # Return True and the index of the touched object
    return False, None  # No touch detected

with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=0) as pose:
    # Initialize webcam.
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480) 
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  
    frame_skip = 5
    frame_count = 5

    while cap.isOpened():
        success, image = cap.read()
        if not success or frame_count % frame_skip != 0:
            frame_count += 1
            continue

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        image_height, image_width, _ = image.shape

        if results.pose_landmarks:
            wrist_centers = find_extremities_centers(results.pose_landmarks, image_width, image_height)
            wrist_names = ['Right Hand', 'Left Hand', 'Right Foot', 'Left Foot']

            # Modified part: We process each wrist/foot individually and make decisions based on lock status.
            for wrist_center, wrist_name in zip(wrist_centers, wrist_names):
                # No longer automatically finding the closest object here

                # Only proceed if this wrist is the locked wrist or if there's no locked limb yet
                if locked_wrist_name == wrist_name or locked_wrist_name is None:
                    if locked_rock_index is not None:
                        # If a rock is already locked with this limb, maintain that lock
                        closest_object_center, box = user_defined_boxes[locked_rock_index]
                        # Directly use the locked rock's center for distance and drawing
                        distance_to_closest = np.linalg.norm(np.array(wrist_center) - np.array(closest_object_center))
                        cv2.line(image, wrist_center, closest_object_center, (0, 255, 0), 3)
                        emit_sound_based_on_distance(distance_to_closest)

                        touched, _ = check_for_touch(wrist_center, [user_defined_boxes[locked_rock_index]], touchThreshold=20)
                        if touched:
                            mixer.music.play()
                            time.sleep(1)
                            user_defined_boxes.pop(locked_rock_index)
                            update_highest_box()
                            locked_wrist_name = None
                            locked_rock_index = None
                            if highest_box_index is None:
                                async_speak("Level completed!")
                                break
                    else:
                        # Lock the limb to the closest rock above it only if no limb is currently locked
                        _, _, closest_object_index = find_closest_object(wrist_center, user_defined_boxes)
                        if closest_object_index is not None and locked_wrist_name is None:
                            locked_wrist_name = wrist_name
                            locked_rock_index = closest_object_index
                            async_speak(f"{locked_wrist_name}")
            
            if locked_wrist_name:
                cv2.putText(image, f"{locked_wrist_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Drawing logic remains mostly unchanged, highlighting locked and highest rocks
            for i, (center, box) in enumerate(user_defined_boxes):
                x, y, w, h = box
                if i == highest_box_index:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                elif i == locked_rock_index:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                else:
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
