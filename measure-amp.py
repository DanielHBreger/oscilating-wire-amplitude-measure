import cv2
import pytesseract
import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt
import threading
import queue
import math
import time
import argparse
import os
import rawpy
import re

import rawpy._rawpy

parser = argparse.ArgumentParser(description='Measure amplitude of a video')
parser.add_argument('arw_path', type=str, help='The path of the raw files')
parser.add_argument('--sample_rate', type=int, default=1,
                    help='The sample rate to use (in frames)')
parser.add_argument('--x', type=int, default=0,
                    help='The x coordinate of the region of interest')
parser.add_argument('--y', type=int, default=100,
                    help='The y coordinate of the region of interest')
parser.add_argument('--wi', type=int, default=600,
                    help='The width of the region of interest')
parser.add_argument('--h', type=int, default=100,
                    help='The height of the region of interest')
parser.add_argument('--threshold', type=int, default=220,
                    help='The threshold to use for binarization')
parser.add_argument('--double_threshold', action='store_true',
                    help='Use double thresholding')
parser.add_argument('--num_threads', type=int, default=10,
                    help='The number of threads to use')
args = parser.parse_args()


class Worker(threading.Thread):
    """Thread worker class for measuring amplitude ina multithreaded way
    """

    def __init__(self, group=None, target=None, name=None, args=..., kwargs=None, *, daemon=None):
        super().__init__(group, target, name, args, kwargs, daemon=daemon)
        self.queue = queue.Queue(maxsize=20)

    def measure(self, files, callback):
        self.queue.put((files, callback))

    def run(self):
        files, callback = self.queue.get()
        for idx, filename in enumerate(files):
            with open(filename, 'rb') as f:
                try:
                    with rawpy.imread(f) as raw:
                        img = raw.postprocess(use_camera_wb=True)
                        blackpoints = []
                        img = img[args.y:args.y+args.h, args.x:args.x+args.wi]
                        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                        # blur = cv2.GaussianBlur(gray, (5, 5), 0)
                        # thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,117,100)
                        ret, thresh = cv2.threshold(
                            gray, args.threshold, 255, cv2.THRESH_BINARY)
                        if args.double_threshold:
                            blur = cv2.GaussianBlur(thresh, (5, 5), 0)
                            ret, thresh = cv2.threshold(blur, 250, 255, cv2.THRESH_BINARY)
                        invert = 255 - thresh
                        for ix, line in enumerate(invert):
                            for iy, pixel in enumerate(line):
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
                        filenumber = re.findall('([0-9]*)\.ARW$',filename)[0]
                        filenumber = int(filenumber)
                        callback(filenumber, maxy[1]-miny[1])
                except rawpy._rawpy.LibRawFileUnsupportedError as e:
                    continue


def main():
    files = [name for name in os.listdir(args.arw_path)]
    files = [file for file in files if file.endswith('.ARW')]
    total_frames = len(files)
    fnos = list(range(0, total_frames, args.sample_rate))
    tasks = [[] for _ in range(0, args.num_threads)]
    frames_per_thread = math.ceil(len(fnos)/args.num_threads)
    for idx, fno in enumerate(fnos):
        tasks[math.floor(idx/frames_per_thread)].append(args.arw_path+'/'+files[fno])

    threads = []
    for _ in range(0, args.num_threads):
        w = Worker()
        threads.append(w)
        w.start()

    results = queue.Queue(maxsize=2000)
    def on_done(x, y): return results.put((x, y))

    ids = []
    widths = []
    frequencies = []

    for idx, w in enumerate(threads):
        w.measure(tasks[idx], on_done)

    while True:
        result = results.get(timeout=10)
        id, amp = result
        ids.append(id)
        widths.append(amp)
        if len(ids) >= total_frames/args.sample_rate:
            break

    ids = [-1*id for id in ids]

    plt.plot(ids, widths, marker='.', linestyle='none')
    plt.xlabel('Frame')
    plt.ylabel('Width (pixels)')
    plt.grid(True)
    plt.savefig(f'{args.arw_path}-AMPS-{time.strftime("%Y%m%d-%H%M%S")}.png')
    plt.show()


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
