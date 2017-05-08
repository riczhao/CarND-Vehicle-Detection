import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from functions import *

img = mpimg.imread('test_images/test1.jpg')
img = img/255

hog_full = get_hog_features(img[:,:,0], 8, 8, 2, feature_vec=False)
fea_full = get_hog_features(img[:,:,0], 8, 8, 2, feature_vec=True)
hog_part = get_hog_features(img[0:64,0:64,0], 8, 8, 2, feature_vec=False)
fea_part = get_hog_features(img[0:64,0:64,0], 8, 8, 2, feature_vec=True)

f_part = hog_part.ravel()

print('end')