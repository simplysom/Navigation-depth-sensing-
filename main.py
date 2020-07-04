import cv2
from yolo import detectobjects

import face_recognition
import argparse
import imutils
import pickle
import time
import os

from yolo_utils import infer_image, show_image
from keras.models import model_from_json

import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\\Users\\Smit\\AppData\\Local\\Tesseract-OCR\\tesseract'
from nltk.corpus import words
# from keras import backend as K
# import keras
from utils import *

#init
BASE_URL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_URL = os.path.join(BASE_URL,'NAVIGATION_BLIND')



#objects
labels=os.path.join(BASE_URL,'coco-labels')
labels = open(labels).read().strip().split("\n")
net = cv2.dnn.readNetFromDarknet(os.path.join(BASE_URL,'yolov3.cfg'), os.path.join(BASE_URL,'yolov3.weights'))
layer_names = net.getLayerNames()
layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def detectobject(frame):
	print("entered")
	confidence = 0.5
	threshold = 0.3
	height, width = frame.shape[:2]
	objects=[]
	confidences, classids, idxs = infer_image(net, layer_names, height, width, frame, labels, confidence, threshold)
	if len(idxs) > 0:
		for i in idxs.flatten():
			if(labels[classids[i]] not in objects):
				objects.append(labels[classids[i]])
	print ("[INFO] Cleaning up...")

	return objects

#texts
json_file = open('text_recog_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('text_detect.h5')

def textrecog(frame):
    print("started")

    img_w = 512
    img_h = 512
    channels = 3
    input_size = (img_h,img_w,channels)
    img = cv2.resize(frame,(512,512))
    img=np.expand_dims(img,axis= 0)
    ans = model.predict((img - 127.5)/127.5)
    boxes = decode(ans[0] , img_w , img_h , 0.5)
    results=[]
    for (startX, startY, endX, endY) in boxes:
        if(int(startX) >0 and int(startY) >0 and int(endX)<512 and int(endY) < 512):
            roi = img[0][int(startY):int(endY), int(startX):int(endX)]
            config = ("-l eng --oem 1 --psm 7")
            text = pytesseract.image_to_string(roi, config=config)
            # if text in words.words():
            results.append(text)
    cv2.destroyAllWindows()
    return results

#faces
data = pickle.loads(open("encoding4.pickle", "rb").read())
def predict_face(frame):
	names=[]
	print("started")
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	rgb = imutils.resize(frame, width=500)
	r = frame.shape[1] / float(rgb.shape[1])
	boxes = face_recognition.face_locations(rgb,model="hog")
	encodings = face_recognition.face_encodings(rgb, boxes)
	for encoding in encodings:
		matches = face_recognition.compare_faces(data["encodings"],encoding)
		# print(len(matches))
		name = "Unknown"
		for i in range(len(matches)):
			if matches[i]:
				name = data["names"][i]
			if(name not in names):
					names.append(name)
	return names

#main
cap = cv2.VideoCapture(0)
frame_count=0
while cap.isOpened():
    ret ,frame = cap.read()
    frame_count+=1
    if ret and frame_count%300==0:
        objects=detectobjects(frame)
        print(objects)
        text=textrecog(frame)
        print(text)
        names = predict_face(frame)
        print(names)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    elif ret:
        continue
    else:
        break
cap.release()
cv2.destroyAllWindows()