import cv2
import pytesseract
import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt

filename = 'IMG_3128.MOV'
sample_rate = 1
video = cv2.VideoCapture(filename)
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
x,y,w,h = 1600, 1000, 1500, 300

ids = []
widths = []
frequencies = []
for fno in range(0, total_frames, sample_rate):
    blackpoints = []
    video.set(cv2.CAP_PROP_POS_FRAMES, fno)
    _, image = video.read()
    img = image.astype("uint8")
    img = img[y:y+h,x:x+w]
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    # thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,117,100)
    ret, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(thresh,(5,5),0)
    ret, thresh = cv2.threshold(blur, 240, 255, cv2.THRESH_BINARY)
    invert = 255 - thresh
    for ix,line in enumerate(invert):
        for iy,pixel in enumerate(line):
            if pixel == 0:
                blackpoints.append((iy, ix))
    while True:
        maxy = max(blackpoints, key=itemgetter(1))
        miny = min(blackpoints, key=itemgetter(1))
        if maxy[0]-miny[0] > 200:
            blackpoints.remove(maxy)
            blackpoints.remove(miny)
        else:
            break
    ids.append(fno)
    widths.append(maxy[1]-miny[1])
    cv2.imshow('', invert)

plt.plot(ids, widths, marker='.', linestyle='none')
plt.xlabel('Frequency (s)')
plt.ylabel('Pixels')
plt.grid(True)
plt.show()