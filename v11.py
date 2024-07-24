import cv2
import face_recognition
import pickle
import time

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
    genderList = ['Male', 'Female']

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
                print(f"Known face detected: {identifier}, Gender: {metadata['gender']}, Age: {metadata['age']}")
                cv2.putText(result_img, f"{identifier}, {metadata['gender']}, {metadata['age']}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
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
