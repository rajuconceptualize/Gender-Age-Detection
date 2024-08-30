import cv2

try:
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        raise Exception("Camera could not be opened. Check if the camera is connected and accessible.")
    
    ret, frame = cap.read()
    
    if not ret:
        raise Exception("Failed to capture image. Ensure the camera is not being used by another application.")
    
    
    print("Image captured successfully.")
    
except cv2.error as e:
    print(f"OpenCV error: {e}")
except Exception as e:
    print(f"Error: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()
