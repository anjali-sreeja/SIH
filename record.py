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

csv_file_name = "coordsTESTadd.csv"

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
        row.insert(0, action)

        with open(csv_file_name, mode='a', newline='') as f:
            csv_writer_obj = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer_obj.writerow(row)
            
    except:
        pass


#recording per sequence
def start_record(action_name):
    
    actions = np.array([action_name])
    no_sequences = 10
    sequence_length = 30

    cam = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.8, min_tracking_confidence=0.5) as holistic:
        
        for action in actions:
            for sequence in range(no_sequences):
                for frame_no in range(sequence_length):
                    
                    ret, frame = cam.read()
                    image, results = mediapipe_detection(frame, holistic)

                    draw_landmarks(image, results)

                    if frame_no == 0:
                        cv2.putText(image, "Collecting Video for {} Video number {}".format(action, sequence), (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
                        cv2.waitKey(2000)
                    else:
                        cv2.putText(image, "Collecting Video for {} Video number {}".format(action, sequence), (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)

                    record_landmarks(results, action)
                    
                    cv2.imshow("Input Sign Language", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        cam.release()
                        cv2.destroyAllWindows()

        cam.release()
        cv2.destroyAllWindows()
