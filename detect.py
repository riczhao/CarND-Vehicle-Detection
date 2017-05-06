import glob
from sklearn.cross_validation import train_test_split
import os.path
from scipy.ndimage.measurements import label





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
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
import pickle

class Classifier:
    def __init__(self):
        self.color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient = 8  # HOG orientations
        self.pix_per_cell = 8 # HOG pixels per cell
        self.cell_per_block = 2 # HOG cells per block
        self.hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
        self.spatial_size = (16, 16) # Spatial binning dimensions
        self.hist_bins = 30    # Number of histogram bins
        self.spatial_feat = True # Spatial features on or off
        self.hist_feat = True # Histogram features on or off
        self.hog_feat = True # HOG features on or off
        self.X_scaler = None
    def fit(self):
        cars = glob.glob('vehicles/*/*')
        notcars = glob.glob('non-vehicles/*/*')
        car_features = extract_features(cars, color_space=self.color_space, 
                        spatial_size=self.spatial_size, hist_bins=self.hist_bins, 
                        orient=self.orient, pix_per_cell=self.pix_per_cell, 
                        cell_per_block=self.cell_per_block, 
                        hog_channel=self.hog_channel, spatial_feat=self.spatial_feat, 
                        hist_feat=self.hist_feat, hog_feat=self.hog_feat)
        notcar_features = extract_features(notcars, color_space=self.color_space, 
                        spatial_size=self.spatial_size, hist_bins=self.hist_bins, 
                        orient=self.orient, pix_per_cell=self.pix_per_cell, 
                        cell_per_block=self.cell_per_block, 
                        hog_channel=self.hog_channel, spatial_feat=self.spatial_feat, 
                        hist_feat=self.hist_feat, hog_feat=self.hog_feat)
        X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
        # Fit a per-column scaler
        self.X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = self.X_scaler.transform(X)
        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=rand_state)
        print('Feature vector length:', len(X_train[0]))
        # Use a linear SVC 
        svc = LinearSVC()
        # Check the training time for the SVC
        t=time.time()
        svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        score = svc.score(X_test, y_test)
        self.svc = svc
        print('Test Accuracy of SVC = ', round(score, 4))
        return score
    
    def color_convert(self, img):
        if self.color_space != 'RGB':
            if self.color_space == 'HSV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif self.color_space == 'LUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            elif self.color_space == 'HLS':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            elif self.color_space == 'YUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            elif self.color_space == 'YCrCb':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(img)      
        return feature_image
 
        # Define a function to extract features from a single image window
        # This function is very similar to extract_features()
        # just for a single image rather than list of images
    def single_img_features(self, img):    
        #1) Define an empty list to receive features
        img_features = []
        #2) Apply color conversion if other than 'RGB'
        feature_image = img
        #3) Compute spatial features if flag is set
        if self.spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=self.spatial_size)
            #4) Append features to list
            img_features.append(spatial_features)
        #5) Compute histogram features if flag is set
        if self.hist_feat == True:
            hist_features = color_hist(feature_image, nbins=self.hist_bins)
            #6) Append features to list
            img_features.append(hist_features)
        #7) Compute HOG features if flag is set
        if self.hog_feat == True:
            if self.hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    self.orient, self.pix_per_cell, self.cell_per_block, 
                                    vis=False, feature_vec=True))      
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], self.orient, 
                        self.pix_per_cell, self.cell_per_block, vis=False, feature_vec=True)
            #8) Append features to list
            img_features.append(hog_features)

        #9) Return concatenated array of features
        return np.concatenate(img_features)

    def predict(self, img): # img after color convert
        features = self.single_img_features(img)
        test_features = self.X_scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        return self.svc.predict(test_features)

class OneFrame:
    def __init__(self,clf,img):
        self.clf = clf
        self.img = img
        p = [2.99187838, -1256.71178676]
        self.wins = [
            #(550,135, 0.1), # (y, w, overlap)
            #(450,135, 0.1), # (y, w, overlap)
            #(437,963-896,0.1),
            #(420,794-777,0.1),
            (y,int((y*p[0]+p[1])*0.6),0.2) for y in [538,497,466,450,436,424]
            ]
    def get_windows(self):
        bboxes = []
        for win in self.wins:
            step = int(win[1]*win[2])
            step = 5 if step < 5 else step
            for x in range(0,1280,step):
                bboxes.append(((x,win[0]-int(win[1])),(x+win[1],win[0])))
        return bboxes
    def detect_cars(self):
        wins = self.get_windows()
        self.wins_to_detect = wins
        img = self.clf.color_convert(self.img)
        #1) Create an empty list to receive positive detection windows
        on_windows = []
        #2) Iterate over all windows in the list
        for window in wins:
            #3) Extract the test window from original image
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
            #4) Extract features for that window using single_img_features()
            prediction = self.clf.predict(test_img)
            #7) If positive (prediction == 1) then save the window
            if prediction == 1:
                on_windows.append(window)
        #8) Return windows for positive detections
        return on_windows
    def merge_wins(self, wins):
        heatmap = np.zeros_like(self.img[:,:,0]).astype(np.float)
        for win in wins:
            heatmap[win[0][1]:win[1][1],win[0][0]:win[1][0]] += 1
        orig_heatmap = heatmap.copy()
        heatmap[heatmap < 3] = 0.
        labels = label(heatmap)
        bboxes = []
        for i in range(1,labels[1]+1):
            nonzero = (labels[0]==i).nonzero()
            ys = nonzero[0]
            xs = nonzero[1]
            bbox = ((xs.min(),ys.min()),(xs.max(),ys.max()))
            bboxes.append(bbox)
        return bboxes, orig_heatmap
if not os.path.isfile('clf.p'):
    clf = Classifier()
    clf.fit()
    with open('clf.p','wb') as f:
        pickle.dump(clf,f)
else:
    with open('clf.p','rb') as f:
        clf = pickle.load(f)

''' 
# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        clf.predict(test_img)
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows
    
    
# Read in cars and notcars
cars = glob.glob('vehicles/*/*')
notcars = glob.glob('non-vehicles/*/*')
# Reduce the sample size because
# The quiz evaluator times out after 13s of CPU time
sample_size = 500
#cars = cars[0:sample_size]
#notcars = notcars[0:sample_size]

### TODO: Tweak these parameters and see how the results change.
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 8  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 30    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [None, None] # Min and max in y to search in slide_window()

car_features = extract_features(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
'''
#image = mpimg.imread('test_images/test1.jpg')

# Uncomment the following line if you extracted training
# data from .png images (scaled 0 to 1 by mpimg) and the
# image you are searching is a .jpg (scaled 0 to 255)

#windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
#                    xy_window=(96, 96), xy_overlap=(0.5, 0.5))
'''
test_imgs = glob.glob("test_images/*.jpg") 
for fn in test_imgs:
    image = mpimg.imread(fn)
    t1 = time.time()
    image = image.astype(np.float32)/255
    frame = OneFrame(clf,image)
    print('frame time:',time.time()-t1)
    hot_windows = frame.detect_cars()
    wins,heatmap = frame.merge_wins(hot_windows)
    image = draw_boxes(image, hot_windows, color=(0., 0., 1.,0.5), thick=2)                    
    image = draw_boxes(image, wins, color=(1., 0., 0.,0.5), thick=2)                    
    plt.subplot(2,1,1)
    plt.imshow(image)
    plt.subplot(2,1,2)
    plt.imshow(heatmap)
print('end')
'''
from moviepy.editor import VideoFileClip
def processImage(img):
    image = np.copy(img)
    image = image.astype(np.float32)/255
    frame = OneFrame(clf,image)
    hot_windows = frame.detect_cars()
    wins,heatmap = frame.merge_wins(hot_windows)
    img = draw_boxes(img, frame.wins_to_detect, color=(0, 255, 0), thick=1)
    img = draw_boxes(img, hot_windows, color=(0, 0, 255), thick=1)
    img = draw_boxes(img, wins, color=(255, 0, 0), thick=2)
 
    return img
def markVideo(fn):
    white_output = 'mark_'+fn
    clip1 = VideoFileClip(fn)
    white_clip = clip1.fl_image(processImage)
    white_clip.write_videofile(white_output, audio=False)

if __name__ == '__main__':
    markVideo('test_video.mp4')
    #markVideo('project_video.mp4')
    pass
