import cv2
import numpy as np

# Initialize the list to store centers of boxes
centers_boxes = []

# Callback function to capture mouse clicks
def click_event(event, x, y, flags, param):
    # If the left button is clicked, append the position to the centers_boxes list
    if event == cv2.EVENT_LBUTTONDOWN:
        centers_boxes.append((x, y))
        print(f"Point added to centers_boxes: {(x, y)}")
        # Optionally, you can show the click point on the window
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Webcam", frame)

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cv2.namedWindow("Webcam")
# Set the click_event function to handle mouse events in the "Webcam" window
cv2.setMouseCallback("Webcam", click_event)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot capture frame from webcam.")
        break
    
    # Display the resulting frame
    cv2.imshow("Webcam", frame)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()

print("Final centers_boxes list:", centers_boxes)
