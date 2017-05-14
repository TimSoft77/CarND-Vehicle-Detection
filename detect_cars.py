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
from moviepy.editor import VideoFileClip


# Track the heat over multiple frames
class Heat:
    # Intialize a zeros array of the same shape as the images
    def __init__(self, img):
        self.heatmap = np.zeros_like(img[:, :, 0]).astype(np.float)

    # Reduce the heat across the heatmap
    def cool(self, cool_rate, max_heat_value):
        self.heatmap -= cool_rate
        self.heatmap = np.clip(self.heatmap, 0, max_heat_value)


# Config - needs to be the same as used to derive the model (i.e., as in build_classifier)
# TODO Save/load with model
spatial = 16
histbin = 32
colorspace = 'HSV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orientation = 9
pixels_per_cell = 8
cells_per_block = 2
windows_overlap = 0.75
window_sizes_positions = [[64, 400, 464], [96, 400, 592], [128, 400, 656], [192, 378, 666], [256, 410, 666]]  # Format: size, y_start, y_stop

# Config - new (or can be different) from build_classifier
cooling_rate = 1
max_heat = 20
heat_threshold = 5

# Load model
svc = joblib.load('svc.pkl')
scaler = joblib.load('scaler.pkl')

# Load a test image to initialize the heatmap
test_image = mpimg.imread('./test_images/test1.jpg')
heatmap = Heat(test_image)


# Implement pipeline
def process_image(img):
    # Define the windows
    windows = []
    for window_config in window_sizes_positions:
        windows.extend(slide_window(img, y_start_stop=[window_config[1], window_config[2]], xy_window=(window_config[0], window_config[0]), xy_overlap=(windows_overlap, windows_overlap)))

    # Search for cars
    car_windows = search_windows(img, windows, svc, scaler, color_space=colorspace, spatial_size=(spatial, spatial), hist_bins=histbin, hist_range=(0, 256), orient=orientation, pix_per_cell=pixels_per_cell, cell_per_block=cells_per_block)
    # Update heatmap and get labels
    # TODO Incorporate the methods below into the heat class
    heatmap.heatmap = add_heat(heatmap.heatmap, car_windows)
    heat = apply_threshold(heatmap.heatmap, heat_threshold)
    heat = np.clip(heat, 0, 255)
    labels = label(heat)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)

    # Cool the heatmap for the next iteration
    heatmap.cool(cooling_rate, max_heat)

    return draw_img

# Create video
output_name = 'project_output.mp4'
clip1 = VideoFileClip("project_video.mp4")
output_clip = clip1.fl_image(process_image)  # NOTE: this function expects color jpg images!!
output_clip.write_videofile(output_name, audio=False)
