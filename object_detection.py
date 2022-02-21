import cv2
import numpy as np


bufferModel = 'ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'
bufferConfig = 'ssd_mobilenet_v1_coco_2017_11_17.pbtxt'

tensorflowNet = cv2.dnn.readNetFromTensorflow(bufferModel, bufferConfig)