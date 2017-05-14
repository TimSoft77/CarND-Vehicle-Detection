import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from project_functions import *

image = mpimg.imread('./test_images/test1.jpg')

windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[378, 666], xy_window=(192, 192), xy_overlap=(0.75, 0.75))

window_img = draw_boxes(image, windows, color=(0, 0, 255), thick=6)
plt.imshow(window_img)
plt.show()
