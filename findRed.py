import cv2
import numpy as np

# Function to get the red color mask
def get_red_mask(hsv_frame):
    # Define the lower and upper bounds for the red color in HSV space
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for the red color (red has two ranges in HSV)
    mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)

    # Combine masks
    mask = mask1 + mask2
    return mask

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frames from the webcam
    _, frame = cap.read()

    # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Get the mask for red colors
    red_mask = get_red_mask(hsv_frame)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Optionally, filter out small contours
        if cv2.contourArea(contour) > 400:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('Frame', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
