import cv2
import numpy as np
import matplotlib.pyplot as plt
import main

def predict(frame):
    # load the
    rows, cols, channels = frame.shape

    networkOutput = main.prediction(frame)

    for detection in networkOutput[0,0]:
            
        score = float(detection[2])
        if score > 0.2:
            
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows

            #draw a red rectangle around detected objects
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)

    return frame