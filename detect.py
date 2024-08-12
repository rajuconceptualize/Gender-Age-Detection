# A Gender and Age Detection System
# used for onsign.tv player

import cv2
import math
import argparse
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



def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes


parser=argparse.ArgumentParser()
parser.add_argument('--image')

args=parser.parse_args()

faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

video=cv2.VideoCapture(args.image if args.image else 0)
padding=20
while cv2.waitKey(1)<0 :
    hasFrame,frame=video.read()
    if not hasFrame:
        cv2.waitKey()
        break
    
    resultImg,faceBoxes=highlightFace(faceNet,frame)
    if not faceBoxes:
        print("No face detected")

    face_number = 1
    for faceBox in faceBoxes:

        # response_1 = player(API.PLAYER_STATUS)

        # print('Response 1:', response_1)

        face=frame[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, frame.shape[1]-1)]

        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        gender=genderList[genderPreds[0].argmax()]
        print(f'Face #: {face_number}')
        print(f'Gender: {gender}')

        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]
        print(f'Age: {age[1:-1]} years')

        cv2.putText(resultImg, f'Face #{face_number} {gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        cv2.imshow("Detecting age and gender", resultImg)
        face_number+= 1
