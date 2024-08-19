import cv2
import face_recognition
import pickle
import time
import requests
import json 

def open_camera():
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Could not open the camera. Please check if it's connected or if the access is allowed.")
        
        # Your existing gender and age detection code
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture image")
                break

            # Process the frame (e.g., gender and age detection)
            # ...

            # Display the resulting frame
            cv2.imshow('Camera', frame)

            # Exit loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    open_camera()
