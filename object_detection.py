import cv2
import numpy as np
import matplotlib.pyplot as plt
import main

frame = main.read_image('data/data_img.jpg')
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

cv2.imwrite("predict_data/prediction.jpg", frame)
plt.imshow(frame)
plt.show()