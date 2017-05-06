import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from functions import *
from sklearn.cross_validation import train_test_split

cars = glob.glob('vehicles/*/*')
notcars = glob.glob('non-vehicles/*/*')

color_spaces = ['RGB','HSV','LUV','HLS','YUV','YCrCb']

def get_hist_score(color_space, hist_bins):
    car_features = extract_features(cars,
                        color_space=color_space, 
                        hist_bins=hist_bins, 
                        hist_feat=True)
    notcar_features = extract_features(notcars,
                        color_space=color_space, 
                        hist_bins=hist_bins, 
                        hist_feat=True)
    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
    # Fit a per-column scaler
    scaled_X = StandardScaler().fit_transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
                    scaled_X, y, test_size=0.2, random_state=rand_state)

    # Use a linear SVC 
    svc = LinearSVC()
    svc.fit(X_train, y_train)
    return svc.score(X_test, y_test)

def get_best_hist():
    max_score = 0
    hist_bins = range(1,64,2)
    best = {}
    for color_space in color_spaces:
        for num_bins in hist_bins:
            score = get_hist_score(color_space, num_bins)
            print(score,color_space,num_bins)
            if score > max_score:
                best['color_space'] = color_space
                best['hist_bins'] = num_bins
                max_score = score
    print(max_score)
    print(best)
def get_hog_score(color_space, orient,pix_per_cell,cell_per_block,hog_channel):
    car_features = extract_features(cars,
                        color_space=color_space, 
                        orient=orient,pix_per_cell=pix_per_cell,cell_per_block=cell_per_block,
                        hog_channel=hog_channel, hog_feat=True
                        )
    notcar_features = extract_features(notcars,
                        color_space=color_space, 
                        orient=orient,pix_per_cell=pix_per_cell,cell_per_block=cell_per_block,
                        hog_channel=hog_channel, hog_feat=True
                        )
    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
    # Fit a per-column scaler
    scaled_X = StandardScaler().fit_transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
                    scaled_X, y, test_size=0.2, random_state=rand_state)

    # Use a linear SVC 
    svc = LinearSVC()
    svc.fit(X_train, y_train)
    return svc.score(X_test, y_test)
def get_best_hog():
    orients = range(4,33,2)
    pix_per_cells = range(2,33,2)
    cell_per_blocks = [2,3,4,5]
    hog_channels = [0,1,2,'ALL']
    max_score = 0
    best = None
    for color_space in color_spaces:
        for orient in orients:
            for pix_per_cell in pix_per_cells:
                for cell_per_block in cell_per_blocks:
                    for hog_channel in hog_channels:
                        score = get_hog_score(color_space, orient, pix_per_cell, cell_per_block, hog_channel);
                        print(color_space, orient, pix_per_cell, cell_per_block, hog_channel)
                        if score > max_score:
                            max_score = score
                            best = (color_space, orient, pix_per_cell, cell_per_block, hog_channel)
    print('maxscore ', max_score)
    print(best)
    
def get_windows():
    wins = [
        (450,135, 1), # (bottom, w, overlap)
        (437,963-896,1),
        ]
    bboxes = []
    for win in wins:
        line = []
        step = int(win[1]*win[2])
        step = 5 if step < 5 else step
        for x in range(0,1280,step):
            line.append(((x,win[0]-win[1]//2),(x+win[1],win[0]+win[1]//2)))
        bboxes.append(line)
    return bboxes

import glob
test_imgs = glob.glob("test_images/*.jpg") 
wins = get_windows()
for fn in test_imgs:
    img = mpimg.imread(fn)#/255
    for line in wins:
        drawn = draw_boxes(img, line, thick=3)
        plt.imshow(drawn)
#get_best_hist()
#get_best_hog()