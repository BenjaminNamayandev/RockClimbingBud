import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Create a video capture object to access the webcam.
cap = cv2.VideoCapture(0)

# Initialize the Pose estimation model.
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Process the image and perform pose detection.
        results = pose.process(image)

        # Convert the image color back so it can be displayed.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw the pose annotations on the image.
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

        # Display the resulting frame
        cv2.imshow('MediaPipe Pose', image)

        # Press 'q' to exit the loop
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Release the capture once everything is done
cap.release()
cv2.destroyAllWindows()

