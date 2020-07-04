import pytesseract
from keras import backend as K
import keras
import cv2
from utils import *
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.layers import *
from keras.applications.mobilenet_v2 import MobileNetV2
import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.models import model_from_json
import matplotlib.pyplot as plt
import os
# import enchant
from nltk.corpus import words
BASE_URL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_URL = os.path.join(BASE_URL,'NAVIGATION_BLIND')

def yolo_model(input_shape):
    classes = 1
    info = 5
    grid_w = 16
    grid_h = 16
    inp = Input(input_shape)

    model = MobileNetV2( input_tensor= inp , include_top=False, weights='imagenet')
    last_layer = model.output

    conv = Conv2D(512,(3,3) , activation='relu' , padding='same')(last_layer)
    conv = Dropout(0.4)(conv)
    bn = BatchNormalization()(conv)
    lr = LeakyReLU(alpha=0.1)(bn)


    conv = Conv2D(128,(3,3) , activation='relu' , padding='same')(lr)
    conv = Dropout(0.4)(conv)
    bn = BatchNormalization()(conv)
    lr = LeakyReLU(alpha=0.1)(bn)


    conv = Conv2D(5,(3,3) , activation='relu' , padding='same')(lr)

    final = Reshape((grid_h,grid_w,classes,info))(conv)

    model = Model(inp,final)

    return model



def predict_func(model , inp , iou ):
    img_w = 512
    img_h = 512
    ans = model.predict(inp)
    boxes = decode(ans[0] , img_w , img_h , iou)
    return boxes

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
            if text in words.words():
                results.append(text)
    cv2.destroyAllWindows()
    return results

# frame = cv2.imread("C:\\Users\\Smit\\OneDrive\\Navigation_blind\\download.jfif")
