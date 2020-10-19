import torch
import torchvision.transforms.functional as F
from torchvision import transforms as torch_transforms
import numpy as np
from PIL import ImageOps
from PIL import Image
import math
from numpy import random
import warnings


class Rotate(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img, expand=False, pad=False):
        if pad and self.angle % 90 != 0:
            w, h = img.size
            # # deterimne crop size (without cutting the image)
            # nw, nh = F.rotate(img, self.angle, expand=True).size

            rad_angle = np.deg2rad(self.angle)
            dw = np.abs(np.ceil(w * (np.cos(rad_angle) * np.sin(rad_angle)))).astype(int)
            dh = np.abs(np.ceil(h * (np.cos(rad_angle) * np.sin(rad_angle)))).astype(int)
            img = F.pad(img, padding=(dw, dh), padding_mode='reflect')

            # actual rotation
            img = F.rotate(img, self.angle)
            # img = F.center_crop(img, (nw, nh))
            img = F.center_crop(img, (w, h))
        else:
            img = F.rotate(img, self.angle, expand=expand)

        return img

