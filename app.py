import tensorflow as tf
from utils import load_class_names, output_boxes, draw_outputs, resize_image
from yolov3 import YOLOv3Net
import cv2
import numpy as np
import argparse
import time
import os
from flask import Flask, request, Response, jsonify
import jsonpickle
#import binascii
import io as StringIO
import base64
from io import BytesIO
import io
import json
from PIL import Image
import time

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)


model_size = (608,608,3)
num_classes = 4
class_name = 'classname.txt'
max_output_size = 100
max_output_size_per_class= 20
iou_threshold = 0.5
confidence_threshold = 0.5

cfgfile = 'yolov3.cfg'
weightfile = 'yolov3_weights.tf'

def image_to_byte_array(image:Image):
  imgByteArr = io.BytesIO()
  image.save(imgByteArr, format='PNG')
  imgByteArr = imgByteArr.getvalue()
  return imgByteArr
def get_prediction(inputimage):
    model = YOLOv3Net(cfgfile,model_size,num_classes)
    model.load_weights(weightfile)
    class_names = load_class_names(class_name)
    win_name = 'Yolov3 detection'
    cv2.namedWindow(win_name)
    #specify the vidoe input.
    # 0 means input from cam 0.
    # For vidio, just change the 0 to video path
    frame = cv2.imread(inputimage,1)
    frame_size = frame.shape

    try:
        # Read frame
        resized_frame = tf.expand_dims(frame, 0)
        resized_frame = resize_image(resized_frame, (model_size[0],model_size[1]))
        pred = model.predict(resized_frame)
        boxes, scores, classes, nums = output_boxes( \
            pred, model_size,
            max_output_size=max_output_size,
            max_output_size_per_class=max_output_size_per_class,
            iou_threshold=iou_threshold,
            confidence_threshold=confidence_threshold)
        img = draw_outputs(frame, boxes, scores, classes, nums, class_names)
        cv2.imshow(win_name, img)
        cv2.imwrite('outputimgage.jpg',img)
            # print("Time taken : {0} seconds".format(seconds))
            # Calculate frames per second

    finally:
        cv2.waitKey()
        cv2.destroyAllWindows()
        print('Detections have been performed successfully.')
        return img
app = Flask(__name__)

# route http posts to this method
@app.route('/api/test', methods=['POST'])
def main():
  # load our input image and grab its spatial dimensions
    #image = cv2.imread("./test1.jpg")
    img = request.files["image"].read()
    img = Image.open(io.BytesIO(img))
    npimg=np.array(img)
    image=npimg.copy()
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    res=get_predection(image)
    # image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    # show the output image
    #cv2.imshow("Image", res)
    #cv2.waitKey()
    image=cv2.cvtColor(res,cv2.COLOR_BGR2RGB)
    np_img=Image.fromarray(image)
    img_encoded=image_to_byte_array(np_img)
    return Response(response=img_encoded, status=200,mimetype="image/jpeg")

    # start flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
