from PIL import Image, ImageFilter
import numpy as np
import torch
import cv2
import torchvision.transforms.functional as F

class HistogramEqualization:
    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img = img.numpy()
            img = img.transpose((1, 2, 0))  

        if isinstance(img, Image.Image):
            img = np.array(img)

        img = img.astype(np.uint8)

        # Check if the image has 4 channels
        if img.ndim == 3 and img.shape[-1] == 4:
            equalized_img = np.zeros_like(img)
            for i in range(4):  
                equalized_img[..., i] = cv2.equalizeHist(img[..., i])
        elif img.ndim == 2:  
            equalized_img = cv2.equalizeHist(img)
        else:
            raise ValueError("Expected 4 channels or grayscale, but got {} dimensions".format(img.shape))

        return equalized_img
    

class ZScoreNormalization:
    def __call__(self, img):
        mean, std = img.mean(), img.std()
        return (img - mean) / std
    


class GaussianBlur:
    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size

    def __call__(self, img):
        return F.gaussian_blur(img, kernel_size=self.kernel_size)
    

class UnsharpMask:
    def __init__(self, radius=2, percent=100, threshold=5):
        self.radius = radius
        self.percent = percent
        self.threshold = threshold

    def __call__(self, img):
        return img.filter(ImageFilter.UnsharpMask(radius=self.radius, percent=self.percent, threshold=self.threshold))
    

class SuppressBackground:
    def __call__(self, img):
        mask = (img < 0.5)  
        img[mask] = img[mask] * 0.5  
        return img