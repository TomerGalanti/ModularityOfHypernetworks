from __future__ import print_function
import argparse
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from rotation_prediction.transforms import Rotate
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision


class MNIST_ROTATE(datasets.MNIST):
    def __init__(self, **kwargs):
        super(MNIST_ROTATE, self).__init__(**kwargs)

    @staticmethod
    def load_img(img):
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        return img

    def get_image(self, index):
        img = self.data[index]
        img = self.load_img(img)
        if self.transform is not None:
            img = self.transform(img)

        return img


    def __getitem__(self, idx):

        angle = random.uniform(0,360)

        transform1 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        transform2 = transforms.Compose([
            Rotate(angle),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        img = self.get_image(idx)
        x = transform1(img)
        y = transform2(img)

        angles = [30*i for i in range(12)]

        closest = min(angles, key=lambda x:abs(x-angle))

        return x, y, closest/30


class CIFAR_ROTATE(datasets.CIFAR10):
    def __init__(self, **kwargs):
        super(CIFAR_ROTATE, self).__init__(**kwargs)

    @staticmethod
    def load_img(img):
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        return img

    def get_image(self, index):
        img = self.data[index]
        img = self.load_img(img)
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __getitem__(self, idx):

        angle = random.uniform(0,360)

        transform1 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        transform2 = transforms.Compose([
            Rotate(angle),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        img = self.get_image(idx)
        x = transform1(img)
        y = transform2(img)

        angles = [30*i for i in range(12)]

        closest = min(angles, key=lambda x:abs(x-angle))

        return x, y, closest/30

