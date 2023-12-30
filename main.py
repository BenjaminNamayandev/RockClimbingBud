import cv2
import pyautogui
import numpy as np

def get_cursor_color():
    # Get the current cursor position
    x, y = pyautogui.position()

    # Get the pixel color at the cursor position
    color = pyautogui.screenshot().getpixel((x, y))

    return color

def get_color_name(color):
    # Define predefined color values
    color_values = {
        'red': (255, 0, 0),
        'green': (0, 255, 0), 
        'blue': (0, 0, 255), 
        'yellow': (255,255,0), 
        'brown': (139,69,19), 
        'orange':(255,165,0), 
        'white': (255,255,255), 
        'black': (0,0,0), 
        'purple': (128, 0, 128), 
        'pink' : (255, 0, 255)
    }

    # Calculate the Euclidean distance to find the closest predefined color
    color_name = min(color_values, key=lambda key: np.linalg.norm(np.array(color) - np.array(color_values[key])))

    return color_name

def main():
    # Open the webcam
    cap = cv2.VideoCapture(0)

    # Initialize the previous color
    prev_color = None

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        # Get the cursor color
        cursor_color = get_cursor_color()

        # Get the color name based on similarity
        color_name = get_color_name(cursor_color)

        # Display the color name on the webcam feed
        cv2.putText(frame, f'Color: {color_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the webcam feed in a window
        cv2.imshow('Webcam Feed', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
