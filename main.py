import numpy as np
import matplotlib.pyplot as plt
import cv2

def read_image(path):
    """
    Read image from path and return a numpy array.
    Args:
        path(str): Path to image.
    Returns:
        numpy: Numpy array of image.
    """
    return cv2.imread(path)

def prediction(frame):
    # "bufferModel" buffer containing the content of the pb file
    bufferModel = 'ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'
    # "bufferConfig" buffer containing the content of the pbtxt file
    bufferConfig = 'ssd_mobilenet_v1_coco_2017_11_17.pbtxt'

    tensorflowNet = cv2.dnn.readNetFromTensorflow(bufferModel, bufferConfig)
    # Use the given image as input, which needs to be blob(s).
    tensorflowNet.setInput(cv2.dnn.blobFromImage(frame, size=(300, 300),swapRB=True, crop=False))

    return tensorflowNet.forward()
