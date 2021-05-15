## Advanced Lane finding project

---

### Overview

The goal of this project is to write a SW pipeline that does the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/CameraCalibration/Figure_0.png "Undistorted"
[image2]: ./output_images/CameraCalibration/Figure_2.png "Road Undistorted"
[image3]: ./output_images/Thresholding/Final.png "Binary Example"
[image4]: ./output_images/LaneFinding_images/warped.png "Warp Example"
[image5]: ./output_images/LaneFinding_images/histogram.png "Histogram"
[image6]: ./output_images/LaneFinding_images/polyfit.png "Histogram"
[image7]: ./output_images/LaneFinding_images/Final.png "Output"
[video1]: ./Output_video.mp4 "Video"
---
### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the lines 20 through 52 of the file called `CameraCalibration.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

---
### Pipeline (single images)

#### 1. Example of a distortion-corrected image.

Using the distortion coefficients and camera matrix computed from calibrating camera, the test images were corrected using 'cv2.undistort()' function. Here is example of one of the image:

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used combination of color and gradient thresholding to generate binary image. The original image was visualized in RGB, HLS, LUV and LAB color spaces. I identified that combination of thresholded L spaces in LAB and LUV and B space in RGB channel gave the best results. This was again combined with combination of X,Y gradients, magnitude and direction thresholds to generate final binary image.  
This can be found in code through lines 41 to 106 in `ImageProcessing.py`
![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspectiveTransform()`, which appears in lines 204 through 213 in the file `AdvancedLaneLines_Library.py`.  The `perspectiveTransform()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the following source and destination points for transforming perspective to bird's eye view.

| Source        | Destination   |
|:-------------:|:-------------:|
| 255, 650      | 255, 650        |
| 575, 460      | 255, 0      |
| 705, 460     | 1025, 0      |
| 1025, 650      | 1025, 650        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Since the probability of finding lane lines is higher at the lower bottom of the image, an histogram was computed on the lower bottom of perspective transformed image to identify lane line regions. This can be seen through lines 215 to 224 in `AdvancedLaneLines_Library.py`. The two highest peaks represent the area of lane lines.

![alt text][image5]

Then, a sliding window was used to track the curvature and fit polynomials for the lane lines. This can be seen through lines 227 to 306 in `AdvancedLaneLines_Library.py`

![alt text][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature of left and right lanes were computed by following this [tutorial](https://www.intmath.com/applications-differentiation/8-radius-curvature.php). The position of vehicle was calculated by computing offset of center of the lane from the center of the image. I did this in the function `DrawLines` (lines 348 through 366 in my code in `AdvancedLaneLines_Library.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Here is an example of my result on a test image:

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_videos/Output_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Some of the limitations of this pipeline are:
1. This SW pipeline works with an assumption that the camera is mounted at the center of the car.
2. The thresholding of images done here is very manual and took a lot of time to tune. There could be adaptive threshoding implemented based on the colors and light conditions.
3. This pipeline works only when the lane lines are clearly visible. If there is snow or fog that obstructs the view of lane lines, the pipeline wont be able to detect lanes.
