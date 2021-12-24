import numpy as np
from PIL import Image
import requests
from zipfile import ZipFile
from io import BytesIO
import matplotlib.pyplot as plt

class DiscountedAveragerator:
    def __init__(self, alpha):
        """Creates an averagerator with a specified discounting factor alpha."""
        self.alpha = alpha
        self.w = 0.
        self.sum_x = 0.
        self.sum_x_sq = 0.

    def add(self, x):
        self.w = self.alpha * self.w + 1.
        self.sum_x = self.alpha * self.sum_x + x
        self.sum_x_sq = self.alpha * self.sum_x_sq + x * x

    @property
    def avg(self):
        return self.sum_x / self.w

    @property
    def std(self):
        mu = self.avg
        # The np.maximum is purely for safety.
        return np.sqrt(np.maximum(0., self.sum_x_sq / self.w - mu * mu))

class MotionDetection(object):
    def __init__(self, num_sigmas = 4., discount = 0.96):
        self.d = DiscountedAveragerator(discount)
        self.sigmas = num_sigmas

    def detect_motion(self, image):
        self.d.add(image)
        neg = image < (self.d.avg - (self.sigmas * self.d.std))
        pos = image > (self.d.avg + (self.sigmas * self.d.std))
        result = np.logical_or(neg, pos)
        return np.max(result, axis=2)

def detect_motion(image_list, num_sigmas = 4., discount = 0.96):
    detector = MotionDetection(num_sigmas = num_sigmas, discount = discount)
    detected_motion = []
    for i, img in enumerate(image_list):
        motion = detector.detect_motion(img)
        if np.sum(motion) > 500:
            detected_motion.append((i, motion))
    return detected_motion

ZIP_URL = "https://storage.googleapis.com/lucadealfaro-share/GardenSequence.zip"
r = requests.get(ZIP_URL)

image_array = []

with ZipFile(BytesIO(r.content)) as myzip:
    for names in myzip.namelist():
        with myzip.open(names) as image_file:
            image = Image.open(image_file)
            image_array.append(np.array(image).astype(np.float32))

motions = detect_motion(image_array[:60])

for i, m in motions:
    if np.sum(m) > 500:
        print("Motion at ", i)
        plt.imshow(image_array[i] / 255)
        plt.show()
        plt.imshow(m)
        plt.show()