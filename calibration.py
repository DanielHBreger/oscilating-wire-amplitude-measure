import cv2
import pytesseract
import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt
import threading
import queue
import math
import time

filename = 'WhatsApp Video 2024-12-26 at 14.22.04.mp4'
sample_rate = 1
video = cv2.VideoCapture(filename)
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
x,y,wi,h = 0, 100, 600, 100
threshold = 220

for fno in range(int(total_frames/2),total_frames,sample_rate):
    cap = cv2.VideoCapture(filename)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
    success, image = cap.read()
    img = image.astype("uint8")
    img = img[y:y+h,x:x+wi]
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    # thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,117,100)
    ret, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    # blur = cv2.GaussianBlur(thresh,(5,5),0)
    # ret, thresh = cv2.threshold(blur, 240, 255, cv2.THRESH_BINARY)
    invert = 255 - thresh
    cv2.imshow('image', invert)
    cv2.waitKey()