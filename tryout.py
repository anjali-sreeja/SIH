import cv2
import os
import csv
import numpy as np
import mediapipe as mp
from matplotlib import pyplot as plt
import time
import pandas as pd
import pickle
import pyttsx3
from train import train
import statistics
from statistics import mode

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

engine = pyttsx3.init()

def most_common(List):
    return(mode(List))

def mediapipe_detection(frame, model):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
   # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def start_detection(skeleton):
    try:
        with open('FiveLearn.pkl', 'rb') as f:
            model = pickle.load(f)
    except:
        # print("No file or directory found..")
        return 0

    predictions = []
    threshold = 0.7
    cam = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.8, min_tracking_confidence=0.5) as holistic:
        while cam.isOpened():
            ret, frame = cam.read()
        
            image, results = mediapipe_detection(frame, holistic)

            if(skeleton==1):
                draw_landmarks(image, results)
            
            #print(results.right_hand_landmarks)

            try:

                if results.pose_landmarks:
                    pose_row = list(np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten())
                else:
                    pose_row = list(np.zeros(132).flatten())
                
                if results.left_hand_landmarks:
                    lh_row = list(np.array([[res.x, res.y, res.z, res.visibility] for res in results.left_hand_landmarks.landmark]).flatten())
                else:
                    lh_row = list(np.zeros(84).flatten())
                
                if results.right_hand_landmarks:
                    rh_row = list(np.array([[res.x, res.y, res.z, res.visibility] for res in results.right_hand_landmarks.landmark]).flatten())
                else:
                    rh_row = list(np.zeros(84).flatten())
                
                row = pose_row + rh_row + lh_row
                
                X = pd.DataFrame([row])
                body_language_class = model.predict(X)[0]
                body_language_prob = model.predict_proba(X)[0]

                predictions.append(body_language_class.split(' ')[0])
                if (len(predictions) > 30):
                    print("hello")
                    predictions = predictions[-8:]
                print(predictions)
                major_prediction = most_common(predictions)
                # print(body_language_class, body_language_prob)

                # Get status box
                if (round(body_language_prob[np.argmax(body_language_prob)],2) > threshold):
        
                    cv2.rectangle(image, (0,0), (640, 35), (0,0,0, 0.5), -1)

                    cv2.putText(image, 'Action:'
                                , (10,25), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(image, body_language_class.split(' ')[0]
                                , (110,25), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 1, cv2.LINE_AA)
                    
                    # Display Probability
                    cv2.putText(image, 'Accuracy:'
                                , (420,25), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                                , (560,25), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
            except:
                pass


            cv2.imshow("Sign Language Converter", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cam.release()
    cv2.destroyAllWindows()
    return 1


engine.runAndWait()