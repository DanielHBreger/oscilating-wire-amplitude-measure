import cv2
import pytesseract
import numpy as np

def getText(file):
    img = file
    img = img.astype("uint8")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),10)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,117,100)
    ret, thresh = cv2.threshold(blur, 205, 255, cv2.THRESH_BINARY)
    invert = 255 - thresh
    x,y,w,h = 535, 290, 175, 110
    ROI = invert[y:y+h,x:x+w]
    data = pytesseract.image_to_string(ROI, lang='eng', config='outputbase digits --psm 6')
    print(data)
    cv2.imshow('',ROI)
    cv2.waitKey()
    return data[:-2]

filename = 'IMG_3128.MOV'
framerate = 24
video = cv2.VideoCapture(filename)
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
frequencies = []
for fno in range(0, total_frames, framerate):
    video.set(cv2.CAP_PROP_POS_FRAMES, fno)
    _, image = video.read()
    try:
        frequencies.append(float(getText(image)))
    except ValueError as e:
        frequencies.append(0)

print(frequencies)