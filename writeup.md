## Vehicle Detection and Tracking Writeup
### Jack Zhang
#### 12/20/17

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/example_car.png
[image2]: ./output_images/example_notcar.png
[image3]: ./output_images/color_space_channels.png
[image4]: ./output_images/spatial_comparison.png
[image5]: ./output_images/color_comparison.png
[image6]: ./output_images/scale1.png
[image7]: ./output_images/scale1_5.png
[image8]: ./output_images/scale2_25.png
[image9]: ./output_images/image_test.png
[image10]: ./output_images/heatmap_test.png
[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in code cell 1 of the Jupyter notebook `Vehicle_detection_tracking.ipynb.py`, in the `extract_features`
function. This function reads in a list of files, opens each of them and converts to the appropriate color-space, and then extracts spatial, color, and HOG features. Each of these feature extraction steps are in separe functions of code cell 1. 

In code cells 4/5, I read in the `vehicle` and `non-vehicle` images. Below is an example of each of the `vehicle` and `non-vehicle` classes, from the combined GTI/KITTI database, along with their HOG parameters for the `YCrCb` color space using parameters of `orientations=12`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`. `L2-Hys` block normalization was used for HOG extraction, since this is what worked best in the work presented by [Dalal and Triggs](http://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf).

![Example of car image and HOG features][image1]
![Example of notcar image and HOG features][image2]

The color space was selected from empirical testing, as well as color visualization, as seen in the example image below. 

![Color space visualization on test image][image3]

I initially wanted to choose a color space where one channel could be used to provide HOG features, such as `YCrCb`, `LUV`, or `HLS`. After testing different color spaces on the training data, I ended up using all 3 channels. I ultimately chose the `YCrCb` color space because it provided good accuracy and had simple range conversion between 8-bit unsigned and 32-bit float.

Below, I show the extracted spatial and color features for the two example `vehicle` and `non-vehicle` images shown earlier. The blue line in the spatial features show the flattened feature array, while the orange line represents the downsampled features, using `spatial_size=(32, 32)`. Similarly, the color features also plot the flattened feature array, which covers all three channels. 

![Example spatial features][image4]
![Example color features][image5]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and color spaces, on both a small (~1200 images of each class) and large (~9000 images of each class) dataset. Ultimately I used the large dataset, and chose the following parameters:

```python
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 12 # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 128    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
```
This was the parameter set that achieved the highest accuracy on my training set (98.85%). It contains a lot of features (10,512) but efforts to reduce the number of features always reduced the accuracy and led to more false positives in the final video pipeline, so I settled for these parameters.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using HOG, color, and spatial features in code cell 6 of the Jupyter notebook. The whole dataset was randomly split into 80% training and 20% testing sets. An accuracy of 98.85% was achieved, and the model and parameters were saved to a pickle file to be recalled for the video pipeline. I used a C parameter of 0.01 (default is 1) in order to reduce the error penalty to make the model more generalizable.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

A sliding window search is implemented in the `find_cars` function in code cell 1 of the Jupyter notebook. A region of interest is selected in the image, and resized if necessary. A square window is moved over this region, and the HOG/spatial/color features are extracted from the window at each position, and classified as a car or not. If a car is detected, the box of the windowed region is returned.

The sliding window search parameters are set in code cell 8. I chose 3 different image scales to sample: `scales = [1, 1.5, 2.25]`. The starting and ending y positions for them are `ystarts = [400,440,540]` and `ystops = [500,590,720]`. These values were chosen empirically so that the resized region would roughly correspond to the size that cars would appear at those locations. An overlap of 75% between each window was used. Example images of the search region for each of the three scales are shown below.

![Scale 1 region][image6]
![Scale 1.5 region][image7]
![Scale 2 region][image8]

More scales could have been used (and were tested with!), but would add to the time of the vehicle detection, and the overlap between the different scales could be reduced, although this would have more significant consequence with the thresholding step to be discussed later. After a lot of trial and error, I came to the conclusion that increasing the number of scales and their locations have to be balanced very carefully, since this could lead to certain regions of the image getting oversampled/producing lots of false positives.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector.  Here are some example results on the test images:

![Pipeline on test images][image9]

These results were obtained after using a threshold of 1 (see below). I tried many different combinations of color-spaces, feature parameters, scales, and window regions. The final model that I used had an accuracy on the test data of 98.85%, with a C parameter of 0.01 to make the model more generalizable.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)

For the video implementation, I sum up all the bounding boxes for 15 consecutive frames (current frame and 14 previous frames), and use those for the heatmap. The camera looks to be recording at 25Hz (1261 frames over 50s) so 15 frames corresponds to 0.6s. This has the effect of greatly reducing the wobble in the detection boxes from frame to frame, compared to using just a single frame's features, while not significantly introducing blurring effects. The code is implemented in code cell 8 of the Jupyter notebook. Because of this, the threshold needs to be increased, by at least a factor of 15. In the final implementation in the video output, a threshold of 45 was used, which corresponds to 3 per frame. 

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video, which is a combination of all the scaling windows at each position.  For a given frame, its positive detections, plus those from the previous 14 frames are combined and added to a heatmap. This heatmap is then thresholded to identify the vehicle positions.  I used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Below is the heatmap of the six sample images, shown above with the bounding boxes from the image pipeline. As can be seen, there are false positives in image 3, 4 and 5 which need to be thresholded to remove. 

![Heatmap on test images][image10]

The same principle applies when frames are summed. 

Some additional thresholding approaches I tried included thresholding more towards the center of the image (since they generally have more window overlaps), and thresholding each pixel according to how many window searches overlapped that pixel. In the end, the simple threshold approach worked the best for me.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I started using the small dataset that contained about 1200 images of cars and non-cars. With this data I could get training data accuracy greater than 99.5%, with less than 1000 features, and 100% with more features. However I got lots of false detections on the video pipeline, which ultimately led me to use the larger dataset with about 9000 images of cars and non-cars. This dataset took much longer to train, and I ultimately needed to use parameters that resulted in over 10,000 features in order to reach 99.3% accuracy, which also significantly increased prediction times. I eventually turned down the C paramter to 0.01 to try to make the model more generalizable, resulting in a final accuracy of 98.85%.

There's always an issue with potential overfitting and making the model generalizable. Even with a model that has 99% training set accuracy, if we assume that it's fully generalizable, that still will generate 1 false classification every 1% of the time. In my final implementation, there are 357 window classifications every frame, so you would expect 3.6 false classifications per frame. Of course, I don't expect the model to be fully generalizable either, so the false classifications would increase even more. This is where the heatmap and averaging helps tremendously. However, obtaining a generalizable model with good accuracy is an essential starting point.

One possibility to improve the pipeline could be to combine multiple color-spaces, such as the `L` channel from `LUV` and `Y` channel from `YCrCb`, and similarly, using these channels for the color extraction. Another possibility is to augment the training data further, either with flipped images, or extracted images from the video.
