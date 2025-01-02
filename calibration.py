import cv2
import pytesseract
import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt
import threading
import queue
import math
import time
import rawpy
import os

folder = 'try1/mode 7'
# folder = os.fsencode(folder)
files = [name for name in os.listdir(folder)]
# files = [f for f in files if f=='DSC00279.ARW']
print(files)
sample_rate = 1
total_frames = len(files)
x,y,wi,h = 2000,2300,3000,200
threshold = 250

for file in files:
    fname = folder+'/'+file
    with open(fname, 'rb') as f:
        with rawpy.imread(f) as raw:
            image = raw.postprocess(use_camera_wb=True)
            img = image.astype("uint8")
            img = img[y:y+h,x:x+wi]
            gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            # blur = cv2.GaussianBlur(gray,(5,5),0)
            # thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,117,100)
            ret, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            blur = cv2.GaussianBlur(thresh,(5,5),0)
            ret, thresh = cv2.threshold(blur, 254, 255, cv2.THRESH_BINARY)
            invert = 255 - thresh
            cv2.imshow('image', invert)
            cv2.waitKey()