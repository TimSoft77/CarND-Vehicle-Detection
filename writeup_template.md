###**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image2]: ./report_images/car_HOG.png
[image3]: ./report_images/slide_windows_192pix.png
[image4]: ./report_images/test_image_output.png
[video1]: ./project_output.mp4

[Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

---

### Writeup

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

HOG features are extracted by the `get_hog_features` method, which is called by both the `extract_features` and `single_img_features` methods.  All the above methods are located in `project_functions.py`.  `extract_features` is used for training the classifier, while `single_img_features` is used for detecting cars in images.

Grayscaled images were used for HOG features because the goal is to look at gradients.  An example of a grayscaled test image of a car and its HOG features is below.


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters, but found nothing that produced a better trained classifier than the initial set.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The classifier was trained on line 58 of `build_classifier.py`.  A LinearSVC was used.

In addition to the HOG features, the images were transformed to the HSV color space, and spatial and histogram color features were also use.  The HSV space was used because I suspected that saturation would be a strong indicator of cars.  Although I had no reason to believe that the H or V channels would help, I left them in case the classifier found otherwise.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The method `slide_window` is used to obtain the windows to be searched for cars.  It is called on line 60 of `detect_cars.py`.

The method is iterated to obtain differently sized windows at different places in the image.  The window sizes and locations are specified in line 41 of `detect_cars.py`.

The windows used were chosen by experimenting with the test image below to develop a set of windows that ought to cover all possible cars of interest, using the test image to get an idea of how large cars ought to be at different places in the image.  The image below shows the set of 192x192 sized windows (including 75% overlap, as used in the project).

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I created a heatmap using the positive detections in the image, and thresholded the map to eliminate false positives.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  This worked well in the test image, below.

![alt text][image4]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Like in the test image, I created a heatmap using the positive detections.  However, in the video I maintained a persistent heatmap over the entire video.  Every frame, all the detections are added to the heatmap, and the heatmap is 'cooled' by a certain amount.  No pixel in the heatmap is allowed to get too 'hot', no matter how many detections.

The idea is to use data from several frames to better rule out false positives.  This approach allows a higher threshold to be used, making it harder for false positives to appear as genuine detections.  It also makes it less likely that a car will suddenly disappear.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The big problem with my implementation is that it does a poor job at identifying the white car.  If this were fixed, the threshold could be increased, eliminating the false positives that do appear.  It would also largely eliminate the lagging phenomenon on the white car.

One likely cause of the inability to detect the white car is its orientation relative to the camera making it more rectangular than square.  The classifier could be improved by introducing rectangular car images, and the detection algorithm could search for rectangular in addition to square images.

My implementation should be refactored to perform fewer HOG feature extractions.  At the moment, it takes ~1.45 s to process a frame.

