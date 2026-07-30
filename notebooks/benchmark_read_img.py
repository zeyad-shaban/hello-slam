import numpy as np
import cv2
import time

img = cv2.imread("../data/dummy_img.jpg")
assert img is not None, "img cant' be none"

avg_time = 0
for i in range(5):
    start = time.monotonic()
    img[:, :, 2] = 0
    end = time.monotonic()
    duration = (end - start)*1e3
    avg_time += duration
    print(f"tooK: {duration}ms")
    
print(f"avg duration: {avg_time / 5}")