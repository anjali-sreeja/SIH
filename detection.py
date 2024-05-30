import cv2
import os
import csv
import numpy as np
import mediapipe as mp
from matplotlib import pyplot as plt
import time
import pandas as pd
import pickle

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

def test_print():
    print("lol")

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

def record_landmarks(results, action):
    try:
        pose = results.pose_landmarks.landmark
        pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

        right_hand = results.right_hand_landmarks.landmark
        right_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in right_hand]).flatten())

        left_hand = results.left_hand_landmarks.landmark
        left_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in left_hand]).flatten())
 
        row = pose_row + right_hand_row + left_hand_row
        row.insert(0, action)

        with open('coords.csv', mode='a', newline='') as f:
            csv_writer_obj = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer_obj.writerow(row)
            
    except:
        pass


def start_detection():

    with open('body_language.pkl', 'rb') as f:
        model = pickle.load(f)

    cam = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.8, min_tracking_confidence=0.5) as holistic:
        while cam.isOpened():
            ret, frame = cam.read()
        
            image, results = mediapipe_detection(frame, holistic)
            draw_landmarks(image, results)
            
            #print(results.right_hand_landmarks)

            try:
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
        
                right_hand = results.right_hand_landmarks.landmark
                right_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in right_hand]).flatten())
        
                left_hand = results.left_hand_landmarks.landmark
                left_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in left_hand]).flatten())
        
                row = pose_row + right_hand_row + left_hand_row
                
                X = pd.DataFrame([row])
                body_language_class = model.predict(X)[0]
                body_language_prob = model.predict_proba(X)[0]
                print(body_language_class, body_language_prob)

                '''   coords = tuple(np.multiply(
                                np.array(
                                    (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                    results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                            , [640,480]).astype(int))
                
                cv2.rectangle(image, 
                            (coords[0], coords[1]+5), 
                            (coords[0]+len(body_language_class)*20, coords[1]-30), 
                            (245, 117, 16), -1)
                cv2.putText(image, body_language_class, coords, 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)'''
                
                # Get status box
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
