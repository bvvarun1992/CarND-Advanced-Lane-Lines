#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 20:09:49 2021

@author: varun
"""
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import cv2
import glob
import os
from CameraCalibration import cal_undistort
import pickle
from AdvancedLaneLines_Library import *

abspath = os.path.abspath('') ## String which contains absolute path to the script file
os.chdir(abspath) ## Setting up working directory


########################################################################################################
############################################## Main ####################################################
######################################################################################################## 

# Reading image from folder
image = img.imread('test_images/test6.jpg')

# Image size
image_size = (image.shape[1], image.shape[0])

# Loading calibration parameters 
calib = pickle.load( open ("camcalib_data.pkl", "rb"))
mtx1 = calib["mtx"]
dist1 = calib["dist"]

# Undistroting the image
image = cal_undistort(image, mtx1, dist1)


# Performing gradient X and Y, magnitude and direction masking
bin_gradx = Thresholding_grad(image, orient = 'x', sobel_kernel = 5, gradThresh = (30, 100))
bin_grady = Thresholding_grad(image, orient = 'y', sobel_kernel = 5, gradThresh = (30, 100))
bin_mag = Thresholding_mag(image, sobel_kernel = 5, magThresh = (30, 100))
bin_dir = Thresholding_dir(image, sobel_kernel = 5, dirThresh = (0.7, 1.3))

# Combining above thresholds
bin_comb = Thresholding_combined(bin_gradx, bin_grady, bin_mag, bin_dir)

# # Visualize binaries of gradiants
# f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(24,9))
# f.tight_layout()
# ax1.imshow(bin_gradx)
# ax1.set_title('Gradient X')
# ax2.imshow(bin_grady)
# ax2.set_title('Gradient Y')
# ax3.imshow(bin_mag)
# ax3.set_title('Magnitude X and Y')
# ax4.imshow(bin_dir)
# ax4.set_title('Direction X and Y')

# Visualizing RGB, HLS and LAB thresholds
rgb_colr = Thresholding_RGBcolor(image, colorspace = 'R', colThresh = (150,255))
rgb_colg = Thresholding_RGBcolor(image, colorspace = 'G', colThresh = (150,255))
rgb_colb = Thresholding_RGBcolor(image, colorspace = 'B', colThresh = (220,255))
hls_colh = Thresholding_HLScolor(image, colorspace = 'H', colThresh = (150, 255))
hls_coll = Thresholding_HLScolor(image, colorspace = 'L', colThresh = (180, 255))
hls_cols = Thresholding_HLScolor(image, colorspace = 'S', colThresh = (150, 255))
lab_coll = Thresholding_LABcolor(image, colorspace = 'L', colThresh = (230, 255))
lab_cola = Thresholding_LABcolor(image, colorspace = 'A', colThresh = (150, 255))
lab_colb = Thresholding_LABcolor(image, colorspace = 'B', colThresh = (155, 255))
luv_coll = Thresholding_LUVcolor(image, colorspace = 'L', colThresh = (210, 255))
luv_colu = Thresholding_LUVcolor(image, colorspace = 'U', colThresh = (155, 255))
luv_colv = Thresholding_LUVcolor(image, colorspace = 'V', colThresh = (155, 255))


# f, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3,3, figsize=(24,9))
# f.tight_layout()
# ax1.imshow(rgb_colr)
# ax1.set_title('R space')
# ax2.imshow(rgb_colg)
# ax2.set_title('G space')
# ax3.imshow(rgb_colb)
# ax3.set_title('B space')
# ax4.imshow(hls_colh)
# ax4.set_title('H space')
# ax5.imshow(hls_coll)
# ax5.set_title('L space')
# ax6.imshow(hls_cols)
# ax6.set_title('S space')
# ax7.imshow(lab_coll)
# ax7.set_title('L space')
# ax8.imshow(lab_cola)
# ax8.set_title('A space')
# ax9.imshow(lab_colb)
# ax9.set_title('B space')


# Combining B, S and L space
bin_col = np.zeros_like(rgb_colb)
bin_col[ (luv_coll == 1) | (rgb_colb == 1) | (lab_coll == 1)] = 1


# Combining gradient and color thresholding
bin_combcol = np.zeros_like(bin_comb)
bin_combcol[(bin_comb == 1) | (bin_col == 1)] = 1



#Defining src and dest for perspective transformation
src = np.float32([[255,650],
                  [575,460],
                  [705,460],
                  [1025,650]])

dest = np.float32([[255,650],
                  [255,0],
                  [1025,0],
                  [1025,650]])


# Warping the image for bird eye view
warpedimage, Minv = Perspective_transform(bin_combcol, image_size, src, dest)

# Caclulating histogram
histogram = hist(warpedimage)

fig3, ax5 = plt.subplots()
ax5.plot(histogram)
ax5.set_title('Histogram of bottom half of image')

# Finding lana lines using moving window
out_image, leftx, lefty, rightx, righty, left_fitx, right_fitx, ploty = FindLaneLines(warpedimage)

# computing radius of curvature and offset from center of lane
left_curverad, right_curverad, offset = FindingCurvature(leftx, lefty, rightx, righty, image_size)

# Retransforming the warped image and drawing the fit lane lines
final_image = DrawLines(image, warpedimage, left_fitx, right_fitx, ploty, Minv)

# Visualing the stages of pipeline
fig1, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2,2)
fig1.tight_layout()

ax1.imshow(image)
ax1.set_title('Actual image')
ax2.imshow(bin_combcol)
ax2.set_title('binary thresholded image')
ax3.imshow(warpedimage)
ax3.set_title('Warped image')
ax4.imshow(out_image)
ax4.set_title('polynomial fit')
# Plots the left and right polynomials on the lane lines
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')

# Visualizing original image with lane lines and information about curvature
fig2, ax = plt.subplots()
cv2.putText(final_image, 'Left lane radius of curvature: '+ str(np.round(left_curverad,2))+'m', (50,100), fontFace = 16, fontScale = 1, color = (255,255,255), thickness = 2)
cv2.putText(final_image, 'Right lane radius of curvature: '+ str(np.round(right_curverad,2))+'m', (50,150), fontFace = 16, fontScale = 1, color = (255,255,255), thickness = 2)
cv2.putText(final_image, 'Car offset by: ' + str(np.round(offset,2)) + 'm', (50,200), fontFace = 16, fontScale = 1, color = (255,255,255), thickness = 2)
ax.imshow(final_image)