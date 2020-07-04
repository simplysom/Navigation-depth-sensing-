
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2


def predict_faces(frame):

	print("[INFO] loading encodings...")
	data = pickle.loads(open("encoding4.pickle", "rb").read())
	# print(data)

	print("[INFO] starting video stream...")
	# vs = cv2.VideoCapture(videopath)
	names=[]
	# while vs.isOpened:
		# ret, frame = vs.read()
	print("started")
		# if(ret==True):

	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	rgb = imutils.resize(frame, width=500)
	r = frame.shape[1] / float(rgb.shape[1])

	boxes = face_recognition.face_locations(rgb,model="hog")
	encodings = face_recognition.face_encodings(rgb, boxes)
	# print(len(encodings))

	for encoding in encodings:
		matches = face_recognition.compare_faces(data["encodings"],encoding)
		# print(len(matches))
		name = "Unknown"
		for i in range(len(matches)):
			if matches[i]:
				name = data["names"][i]
			if(name not in names):
					names.append(name)
		# if True in matches:
		# 	matchedIdxs = [i for (i, b) in enumerate(matches) if b]
		# 	counts = {}
		# 	for i in matchedIdxs:
		# 		name = data["names"][i]
		# 		counts[name] = counts.get(name, 0) + 1
		# 	name = max(counts, key=counts.get)
		# 	if(name not in names):
		# 		names.append(name)
		# 	print(counts)
	return names




