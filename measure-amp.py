import cv2
import pytesseract
import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt
import threading
import queue
import math

filename = 'IMG_3128.MOV'
sample_rate = 60
video = cv2.VideoCapture(filename)
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
x,y,wi,h = 1600, 1000, 1500, 300

class Worker(threading.Thread):

    def __init__(self, group = None, target = None, name = None, args = ..., kwargs = None, *, daemon = None):
        super().__init__(group, target, name, args, kwargs, daemon=daemon)
        self.queue = queue.Queue(maxsize=20)
    
    def measure(self, video_path, fnos, callback):
        self.queue.put((video_path, fnos, callback))

    def run(self):
        video_path, fnos, callback = self.queue.get()
        cap = cv2.VideoCapture(video_path)
        for fno in fnos:
            blackpoints = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
            success, image = cap.retrieve()
            img = image.astype("uint8")
            img = img[y:y+h,x:x+wi]
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
            callback(fno, maxy[1]-miny[1])

fnos = list(range(0,total_frames,sample_rate))
num_threads = 8
tasks = [[] for _ in range(0, num_threads)]
frames_per_thread = math.ceil(len(fnos)/num_threads)
tid=0
for idx, fno in enumerate(fnos):
    tasks[math.floor(idx/frames_per_thread)].append(fno)

threads = []
for _ in range(0, num_threads):
    w = Worker()
    threads.append(w)
    w.start()

results = queue.Queue(maxsize=2000)
on_done = lambda x,y: results.put((x,y))

ids = []
widths = []
frequencies = []

for idx, w in enumerate(threads):
    w.measure(filename, tasks[idx], on_done)

while True:
    result = results.get(timeout=5)
    print(result)
    id, amp = result
    ids.append(id)
    widths.append(amp)
    if len(ids) >= total_frames/sample_rate:
        break

plt.plot(ids, widths, marker='.', linestyle='none')
plt.xlabel('Frequency (s)')
plt.ylabel('Pixels')
plt.grid(True)
plt.show()