import cv2
from deepface import DeepFace
import os


backends = [
  'opencv', 
  'ssd', 
  'dlib', 
  'mtcnn', 
  'fastmtcnn',
  'retinaface', 
  'mediapipe',
  'yolov8',
  'yunet',
  'centerface',
]



alignment_modes = [True,False]



# Manually specify the path to the haarcascade_frontalface_default.xml file
haar_cascade_path = os.path.join(cv2.__path__[0], 'data', 'haarcascade_frontalface_default.xml')

# Load the face detection model
face_cascade = cv2.CascadeClassifier(haar_cascade_path)

# Start the webcam feed
cap = cv2.VideoCapture(0)

face_id = 0  # To assign a unique ID to each detected face

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Loop through each face detected
    for (x, y, w, h) in faces:
        face_id += 1  # Increment face ID for each face detected

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extract the face region of interest (ROI) to analyze
        face_roi = frame[y:y+h, x:x+w]

        # Analyze the face for age, gender, and emotion using DeepFace
        try:
            # Perform analysis
            predictions = DeepFace.analyze(face_roi, \
                                           detector_backend = backends[5],\
                                           align = alignment_modes[1],
                                           actions=['age', 'gender', 'race', 'emotion'], enforce_detection=False)
            
            # If predictions are returned as a list, extract the first item
            if isinstance(predictions, list):
                predictions = predictions[0]

            # Get the predictions
            gender = predictions['gender']
            age = predictions['age']
            dominant_emotion = predictions['dominant_emotion']
            race = predictions['race']

            # Put the text on the frame near the face
            # cv2.putText(frame, f"ID: {face_id}", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, f"Race: {race}", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, f"Gender: {gender}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Age: {int(age)}", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Mood: {dominant_emotion}", (x, y + h + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
 
        except Exception as e:
            print(f"Error in detection: {e}")
    
    # Display the frame with rectangles and predictions
    cv2.imshow('Live Feed', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
