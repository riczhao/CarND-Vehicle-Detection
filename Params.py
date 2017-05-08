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
        (417,793-775,1),
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
def draw_wins(img):
    wins = get_windows()
    for line in wins:
        drawn = draw_boxes(img, line, thick=3)
        plt.imshow(drawn)

from moviepy.editor import VideoFileClip
def processImage(img):
    image = np.copy(img)
    draw_wins(img)
    '''
    image = image.astype(np.float32)/255
    frame = OneFrame(clf,image)
    hot_windows = frame.detect_cars()
    wins,heatmap = frame.merge_wins(hot_windows)
    img = draw_boxes(img, frame.wins_to_detect, color=(0, 255, 0), thick=1)
    img = draw_boxes(img, hot_windows, color=(0, 0, 255), thick=1)
    img = draw_boxes(img, wins, color=(255, 0, 0), thick=2)
    '''
 
    return img
def markVideo(fn):
    white_output = 'mark_'+fn
    clip1 = VideoFileClip(fn)
    white_clip = clip1.fl_image(processImage)
    white_clip.write_videofile(white_output, audio=False)

def tune_hog_images():
    cars = glob.glob('vehicles/*/*')
    notcars = glob.glob('non-vehicles/*/*')
    import random
    car_idx = random.randrange(len(cars))
    notcar_idx = random.randrange(len(notcars))
    car_img = mpimg.imread(cars[car_idx])/255
    notcar_img = mpimg.imread(notcars[notcar_idx])/255
    car_y = cv2.cvtColor(car_img, cv2.COLOR_RGB2YCrCb)
    notcar_y = cv2.cvtColor(notcar_img, cv2.COLOR_RGB2YCrCb)
    for ch in range(3):
        plt.subplot(3,4,ch*4+1)
        plt.imshow(car_y[:,:,ch],cmap='gray')
        _,hog = get_hog_features(car_y[:,:,ch], 8, 8, 2,vis=True, feature_vec=True)
        plt.subplot(3,4,ch*4+2)
        plt.imshow(hog,cmap='gray')
        plt.subplot(3,4,ch*4+3)
        plt.imshow(notcar_y[:,:,ch],cmap='gray')
        _,hog = get_hog_features(notcar_y[:,:,ch], 8, 8, 2,vis=True, feature_vec=True)
        plt.subplot(3,4,ch*4+4)
        plt.imshow(hog,cmap='gray')
    plt.show()

tune_hog_images()
#markVideo('test_video.mp4')
#markVideo('project_video.mp4')
#get_best_hist()
#get_best_hog()