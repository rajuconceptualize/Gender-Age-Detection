import cv2
import face_recognition
import pickle
import time
import requests
import json 



class API:
    BASE_URL = 'http://127.0.0.1:5544/'
    PLAYER_STATUS = BASE_URL + 'playback/status'
    PLAYER_STOP_CURRENT = BASE_URL + 'campaign/current/stop'
    PLAYER_SHOW = BASE_URL + 'playback/show'
    PLAYER_HIDE = BASE_URL + 'playback/hide'
    PLAYER_START = BASE_URL + 'playback/start'
    PLAYER_STOP = BASE_URL + 'playback/stop'

    PLAYER_GENERAL = BASE_URL + 'trigger/general'

    PLAYER_FEMALE_YOUNG_ADULT = BASE_URL + 'trigger/FemaleYoungAdult'

    PLAYER_MALE_YOUNG_ADULT = BASE_URL + 'trigger/MaleYoungAdult'
    PLAYER_MALE_SENIOR_ADULT = BASE_URL + 'trigger/MaleSeniorAdult'




def player(url):
    """
    Makes a POST request to the given URL without sending any data.

    Args:
        url (str): The API endpoint URL.

    Returns:
        response (dict): The JSON response from the API if the request is successful.
        None: If the request fails.
    """
    try:


        # Make the POST request without sending any data
        response = requests.post(url)

        # Check the status code of the response
        if response.status_code == 200:
            try:
                return response.json()
            except ValueError:
                # print('Response content is not valid JSON')
                # print('Response text:', response.text)
                return None
        else:
            print(f'Failed: {response.status_code} {response.text}')
            return None

    except Exception as e:
        print(f'An error occurred: {e}')
        return None



def save_known_faces(known_face_encodings, known_face_metadata):
    with open("known_faces.pkl", "wb") as f:
        pickle.dump((known_face_encodings, known_face_metadata), f)



def load_known_faces():
    try:
        with open("known_faces.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return [], []
    


def detect_and_highlight_faces(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    return frame, face_locations, face_encodings





# Function to categorize age based on the given age range
def categorize_age(age_range_str):
    # Remove parentheses and split the string into lower and upper bounds
    lower_bound, upper_bound = map(int, age_range_str.strip('()').split('-'))
    
    # Check the upper bound to determine the category
    if upper_bound < 40:
        return "Young Adult"
    else:
        return "Senior Adult"
    


    
def categorize_gender_age(gender, age_category):
    # Ensure that gender input is case insensitive
    gender = gender.lower()

    # Determine the code based on gender and age category
    if gender == 'female' and age_category == 'Young Adult':
        return 1
    elif gender == 'female' and age_category == 'Senior Adult':
        return 2
    elif gender == 'male' and age_category == 'Young Adult':
        return 3
    elif gender == 'male' and age_category == 'Senior Adult':
        return 4
    else:
        return None  # Return None for invalid input or unrecognized category




def player_trigger(number):
    if number == 1:
        response = player(API.PLAYER_FEMALE_YOUNG_ADULT)
    elif number == 2:
        response = player(API.PLAYER_GENERAL)
    elif number == 3:
        response = player(API.PLAYER_MALE_YOUNG_ADULT)
    elif number == 4:
        response = player(API.PLAYER_MALE_SENIOR_ADULT)
    else:
        response = player(API.PLAYER_GENERAL)
    return response



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
            
            nown_face_encodings, known_face_metadata = load_known_faces()


            aceProto = "opencv_face_detector.pbtxt"
            faceModel = "opencv_face_detector_uint8.pb"
            ageProto = "age_deploy.prototxt"
            ageModel = "age_net.caffemodel"
            genderProto = "gender_deploy.prototxt"
            genderModel = "gender_net.caffemodel"

            MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
            ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
            # ageList = ['(0-40)','(41-100)']
            genderList = ['Male', 'Female']



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
