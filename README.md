## Advanced Lane Finding
---
## Overview
The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

---
## Folder structure

* The file `CameraCalibration.py` includes the code to calibrate a camera with help of images of chessboard. The images can be found in folder `camera_cal`
* The file `AdvancedLaneLines_Library.py` caontains the functions that are used to undistort, create binary, apply perspective transformation, find lane lines and to calculate the curvatures.
* The file `ImageProcessing.py` contains the SW pipeline to identify lane lines on single images and draw the lines back on the image and display information about the curvature and vehicle position.
* The file `VideoProcessing.py` contains SW pipeline to process the frames of video to identify lane lines, calculate the radius of curvature and display them back on the written video.
* There are few single images present in folder `./test_images/test*.jpg` that are used to test the pipeline. The folder `./output_images` and `./output_videos`contains the results of an example image and video tested on the pipeline.
* The file `writeup.md` file is the final report explaining briefly the steps taken in the pipeline.
---
