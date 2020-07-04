import numpy as np
import argparse
import cv2 as cv
import subprocess
import time
import os

def show_image(img):
    cv.imshow("Image", img)
    cv.waitKey(0)



def generate_boxes_confidences_classids(outs, height, width, tconf):
    confidences = []
    classids = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            classid = np.argmax(scores)
            confidence = scores[classid]

            if confidence > tconf:
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, bwidth, bheight = box.astype('int')


                x = int(centerX - (bwidth / 2))
                y = int(centerY - (bheight / 2))

                boxes.append([x, y, int(bwidth), int(bheight)])
                confidences.append(float(confidence))
                classids.append(classid)

    return boxes, confidences, classids

def infer_image(net, layer_names, height, width, img, labels, confidence, threshold,
            boxes=None, confidences=None, classids=None, idxs=None, infer=True):


    if infer:
        blob = cv.dnn.blobFromImage(img, 1 / 255.0, (416, 416),
                        swapRB=True, crop=False)

        net.setInput(blob)

        start = time.time()
        outs = net.forward(layer_names)
        end = time.time()


        boxes, confidences, classids = generate_boxes_confidences_classids(outs, height, width, confidence)

        idxs = cv.dnn.NMSBoxes(boxes, confidences, confidence, threshold)

    if confidences is None or idxs is None or classids is None:

        raise '[ERROR] Required variables are set to None before drawing boxes on images.'

    return confidences, classids, idxs
