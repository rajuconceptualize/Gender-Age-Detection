import cv2
import face_recognition
import pickle
import time
import requests



class API:
    BASE_URL = 'http://127.0.0.1:5544/'
    PLAYER_STATUS = BASE_URL + 'playback/status'
    PLAYER_STOP_CURRENT = BASE_URL + 'campaign/current/stop'
    PLAYER_SHOW = BASE_URL + 'playback/show'
    PLAYER_HIDE = BASE_URL + 'playback/hide'
    PLAYER_START = BASE_URL + 'playback/start'
    PLAYER_STOP = BASE_URL + 'playback/stop'
    PLAYER_TRIGGER_CAMPAIGN = BASE_URL + 'trigger/1'

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
                print('Response content is not valid JSON')
                print('Response text:', response.text)
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
    if number == '1':
        response = player(API.PLAYER_FEMALE_YOUNG_ADULT)
    elif number == '2':
        response = player(API.PLAYER_GENERAL)
    elif number == '3':
        response = player(API.PLAYER_MALE_YOUNG_ADULT)
    elif number == '4':
        response = player(API.PLAYER_MALE_SENIOR_ADULT)
    else:
        response = player(API.PLAYER_GENERAL)
    return response

def main():
    known_face_encodings, known_face_metadata = load_known_faces()

    video_capture = cv2.VideoCapture(0)

    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"
    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"
    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    # ageList = ['(0-40)','(41-100)']
    genderList = ['Male', 'Female']

    # print(type(ageList[0]))

    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture image")
            break

        result_img, face_boxes, face_encodings = detect_and_highlight_faces(frame)

        if not face_boxes:
            print("No face detected")
            continue

        for (top, right, bottom, left), face_encoding in zip(face_boxes, face_encodings):
            face = frame[top:bottom, left:right]
            if face.size == 0:
                print("Empty face array, skipping")
                continue
            rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            print(f"Face shape: {rgb_face.shape}")

            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            if True in matches:
                match_index = matches.index(True)
                identifier = f"Person {match_index + 1}"
                metadata = known_face_metadata[match_index]

                print(type(metadata['age']))
                print(type(metadata['gender']))

                age_category = categorize_age(metadata['age'])
                print(age_category)  # Output will be "Young Adult" for this example


                print(f"Known face detected: {identifier}, Gender: {metadata['gender']}, Category: {age_category}")
                

                cv2.putText(result_img, f"{identifier}, {metadata['gender']}, {age_category}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

                category = categorize_gender_age(metadata['gender'], age_category)
                print(f"Category: {category}")
                response_1 = player_trigger(category)
                print('Response 1:', response_1)
            else:
                faceNet.setInput(cv2.dnn.blobFromImage(rgb_face, 1.0, (300, 300), MODEL_MEAN_VALUES, swapRB=True))
                detections = faceNet.forward()
                if len(detections) > 0:
                    detection = detections[0, 0, 0, :]
                    if detection[2] > 0.7:  # Confidence threshold for face detection
                        genderNet.setInput(cv2.dnn.blobFromImage(rgb_face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=True))
                        gender_preds = genderNet.forward()
                        gender = genderList[gender_preds[0].argmax()]

                        ageNet.setInput(cv2.dnn.blobFromImage(rgb_face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=True))
                        age_preds = ageNet.forward()
                        age = ageList[age_preds[0].argmax()]

                        metadata = {"gender": gender, "age": age}
                        known_face_encodings.append(face_encoding)
                        known_face_metadata.append(metadata)
                        save_known_faces(known_face_encodings, known_face_metadata)

                        identifier = f"Person {len(known_face_metadata)}"
                        print(f"New face detected: {identifier}, Gender: {gender}, Age: {age}")
                        cv2.putText(result_img, f"{identifier}, {gender}, {age}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow("Video", result_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
