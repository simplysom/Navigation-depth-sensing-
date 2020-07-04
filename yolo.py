import numpy as np
import argparse
import cv2 as cv
import subprocess
import time
import os
from yolo_utils import infer_image, show_image
BASE_URL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_URL = os.path.join(BASE_URL,'NAVIGATION_BLIND')


def detectobjects(frame, weights=os.path.join(BASE_URL,'yolov3.weights'), config=os.path.join(BASE_URL,'yolov3.cfg'), labels=os.path.join(BASE_URL,'coco-labels')):
	print("entered")
	confidence = 0.5
	threshold = 0.3

	labels = open(labels).read().strip().split("\n")
	net = cv.dnn.readNetFromDarknet(config, weights)

	layer_names = net.getLayerNames()
	layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


	frame_count = 0
	# while True:
		# grabbed, frame = vid.read()

		# print(frame_count)
	frame_count = frame_count + 1

	# if width is None or height is None:
	height, width = frame.shape[:2]
	objects=[]
	confidences, classids, idxs = infer_image(net, layer_names, height, width, frame, labels, confidence, threshold)

	if len(idxs) > 0:
		for i in idxs.flatten():
			if(labels[classids[i]] not in objects):
				objects.append(labels[classids[i]])


	print ("[INFO] Cleaning up...")

	return objects


