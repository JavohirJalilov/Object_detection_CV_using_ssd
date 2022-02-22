# How to load Tensorflow models with OpenCV
## Download and use the Model

A very useful functionality was added to OpenCV’s DNN module: a [Tensorflow](https://docs.opencv.org/3.4/d6/d0f/group__dnn.html#gad820b280978d06773234ba6841e77e8d) net importer.

Where the function has the following format:
```python
cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'graph.pbtxt')
```

As you might have seen, to use it, two files are needed:
- frozen_inference_graph.pb
- graph.pbtxt

## About Tensorflow’s .pb and .pbtxt files

Tensorflow models usually have a fairly high number of parameters. Freezing is the process to identify and save just the required ones (graph, weights, etc) into a single file that you can use later. So, in other words, it’s the TF way to “export” your model. The freezing process produces a Protobuf ( .pb) file.