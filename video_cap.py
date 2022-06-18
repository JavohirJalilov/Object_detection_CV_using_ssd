import cv2
import matplotlib.pyplot as plt
import numpy as np
from object_detection import predict

cap = cv2.VideoCapture('data/road_mp.mp4')

while True:
    ret,frame = cap.read()
    # cv2.imwrite('frame.png', frame)
    
    predict_frame = predict(frame)
    
    w,h = frame.shape[:2]
    # predict_frame = cv2.resize(predict_frame, (h//2,w//2))
    cv2.imshow('frame',predict_frame)

    if cv2.waitKey(33) == ord('q'):
        break

cv2.release()
cv2.destroyAllWindows()