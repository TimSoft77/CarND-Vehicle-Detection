import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from scipy.ndimage.measurements import label
from sklearn.externals import joblib
from project_functions import *


# Feature extraction parameters
spatial = 16
histbin = 32
colorspace = 'HSV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orientation = 9
pixels_per_cell = 8
cells_per_block = 2
windows_overlap = 0.75
window_sizes_positions = [[64, 400, 464], [96, 400, 592], [128, 400, 656], [192, 378, 666], [256, 410, 666]]  # Format: size, y_start, y_stop
heat_threshold = 1

# Read in car and non-car images
cars = glob.glob('./vehicles/*/*.png')
notcars = glob.glob('./non-vehicles/*/*.png')

# Extract features
t = time.time()
car_features = extract_features(cars, cspace=colorspace, spatial_size=(spatial, spatial), hist_bins=histbin, hist_range=(0, 256), orient=orientation, pix_per_cell=pixels_per_cell, cell_per_block=cells_per_block)
notcar_features = extract_features(notcars, cspace=colorspace, spatial_size=(spatial, spatial), hist_bins=histbin, hist_range=(0, 256), orient=orientation, pix_per_cell=pixels_per_cell, cell_per_block=cells_per_block)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract features...')

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)  # X_scaler type= <class 'sklearn.preprocessing.data.StandardScaler'>
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

print('Feature vector length:', len(X_train[0]))
# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t = time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these', n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict, 'labels with SVC')

# Save model - comment out if not desired
# joblib.dump(svc, 'svc.pkl')
# joblib.dump(X_scaler, 'scaler.pkl')

# Load a test image
test_image = mpimg.imread('./test_images/test1.jpg')

# Get the needed window sizes
window_defs = []
for window_config in window_sizes_positions:
    window_defs.extend(slide_window(test_image, y_start_stop=[window_config[1], window_config[2]], xy_window=(window_config[0], window_config[0]), xy_overlap=(windows_overlap, windows_overlap)))

# Run on the test image
t = time.time()
car_windows = search_windows(test_image, window_defs, svc, X_scaler, color_space=colorspace, spatial_size=(spatial, spatial), hist_bins=histbin, hist_range=(0, 256), orient=orientation, pix_per_cell=pixels_per_cell, cell_per_block=cells_per_block)
t2 = time.time()
print(round(t2-t, 5), 'Seconds to search for cars in', len(window_defs), 'windows, finding', len(car_windows), 'windows thought to contain cars')

# Heat mapping
heat = np.zeros_like(test_image[:, :, 0]).astype(np.float)
heat = add_heat(heat, car_windows)
heat = apply_threshold(heat, heat_threshold)
# Visualize the heatmap when displaying
heatmap = np.clip(heat, 0, 255)

# Find final boxes from heatmap using label function
labels = label(heatmap)
draw_img = draw_labeled_bboxes(np.copy(test_image), labels)

fig = plt.figure()
plt.subplot(121)
plt.imshow(draw_img)
plt.title('Car Positions')
plt.subplot(122)
plt.imshow(heatmap, cmap='hot')
plt.title('Heat Map')
fig.tight_layout()
plt.show()
