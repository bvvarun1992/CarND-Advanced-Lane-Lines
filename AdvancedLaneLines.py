#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  9 18:24:15 2021

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

abspath = os.path.abspath('') ## String which contains absolute path to the script file
os.chdir(abspath) ## Setting up working directory

##################### Function to plot image in RGB and HLS space to identify suitable channel ######################
def RGBAndHLSView(image):
    rgbimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hlsimage = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    rspace = rgbimage[:,:,0]
    gspace = rgbimage[:,:,1]
    bspace = rgbimage[:,:,2]
    hspace = hlsimage[:,:,0]
    lspace = hlsimage[:,:,1]
    sspace = hlsimage[:,:,2]
    
    f, ((ax1,ax2,ax3),(ax4, ax5, ax6)) = plt.subplots(2,3, figsize=(24,9))
    f.tight_layout()
    
    ax1.imshow(rspace, cmap='gray')
    ax1.set_title('R space')
    ax2.imshow(gspace, cmap='gray')
    ax2.set_title('G space')
    ax3.imshow(bspace, cmap='gray')
    ax3.set_title('B space')
    ax4.imshow(hspace, cmap='gray')
    ax4.set_title('H space')
    ax5.imshow(lspace, cmap='gray')
    ax5.set_title('L space')
    ax6.imshow(sspace, cmap='gray')
    ax6.set_title('S space')
    
    return 

########################## Function to mask X or Y gradient within provided thresholds ###########################
def Thresholding_grad(image, orient = 'x', sobel_kernel = 3, gradThresh = (0, 255)):
    
    # Converting to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Taking derivative in X or y direction
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    
    # Absolute of sobel
    abs_sobel = np.absolute(sobel)
    
    # Scaling sobel to 8-bit (0-255)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    # Creating a binary mask for gradients within threshold
    sbinary_grad = np.zeros_like(scaled_sobel)
    sbinary_grad[(scaled_sobel > gradThresh[0]) & (scaled_sobel < gradThresh[1])] = 1
    
    return sbinary_grad

####################### Function to mask magnitude of gradients within provided thresholds ######################    
def Thresholding_mag(image, sobel_kernel = 3, magThresh = (0,255)):
    
    # Converting to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Taking derivative in X and Y 
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    
    # Magnitude of gradients
    magxy = np.sqrt(sobelx**2 + sobely**2)
    
    # Scaling magnitude to 8-bit (0-255)
    scaled_magxy = np.uint8(255*magxy/np.max(magxy))
    
    # Creating a binary mask for magnitudes within threshold
    sbinary_mag = np.zeros_like(scaled_magxy)
    sbinary_mag[(scaled_magxy > magThresh[0]) & (scaled_magxy < magThresh[1])] = 1
    
    return sbinary_mag

####################### Function to mask direction of gradients within provided thresholds ######################
def Thresholding_dir(image, sobel_kernel, dirThresh = (0, np.pi/2)):
    
    # Converting to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Taking derivative in X and Y 
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    
    # Taking absolute value of the derivatives
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    
    # Calculating direction of gradients
    dir_sobel = np.arctan2(abs_sobelx, abs_sobely)
    
    # Creating a binary mask for direction of gradients within threshold
    sbinary_dir = np.zeros_like(dir_sobel)
    sbinary_dir[(dir_sobel > dirThresh[0]) & (dir_sobel < dirThresh[1])] = 1
    
    return sbinary_dir

############################# Function to combine all thresholding masks #########################################
def Thresholding_combined(sbinary_gradx, sbinary_grady, sbinary_mag, sbinary_dir):
    
    sbinary_combined = np.zeros_like(sbinary_dir)
    sbinary_combined[((sbinary_gradx == 1) & (sbinary_grady == 1)) | ((sbinary_mag == 1) & (sbinary_dir == 1))] = 1
    
    return sbinary_combined

######################### Function to mask HLS colorspace within provided thresholds ############################
def Thresholding_HLScolor(image, colorspace = 'H', colThresh = (0,255)):
    
    # Converting image to HLS space
    hlsimage = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hspace = hlsimage[:,:,0]
    lspace = hlsimage[:,:,1]
    sspace = hlsimage[:,:,2]
    
    if colorspace == 'H':
        binary_col = np.zeros_like(hspace)
        binary_col[(hspace > colThresh[0]) & (hspace < colThresh[1])] = 1
    elif colorspace == 'L':
        binary_col = np.zeros_like(hspace)
        binary_col[(lspace > colThresh[0]) & (lspace < colThresh[1])] = 1
    else:
        binary_col = np.zeros_like(hspace)
        binary_col[(sspace > colThresh[0]) & (sspace < colThresh[1])] = 1
    
    return binary_col

######################### Function to mask RGB colorspace within provided thresholds ############################
def Thresholding_RGBcolor(image, colorspace = 'R', colThresh = (0,255)):
    
    # Converting image to RGB space
    rgbimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rspace = rgbimage[:,:,0]
    gspace = rgbimage[:,:,1]
    bspace = rgbimage[:,:,2]
    
    if colorspace == 'R':
        binary_col = np.zeros_like(rspace)
        binary_col[(rspace > colThresh[0]) & (rspace < colThresh[1])] = 1
    elif colorspace == 'G':
        binary_col = np.zeros_like(gspace)
        binary_col[(gspace > colThresh[0]) & (gspace < colThresh[1])] = 1
    else:
        binary_col = np.zeros_like(bspace)
        binary_col[(bspace > colThresh[0]) & (bspace < colThresh[1])] = 1
    
    return binary_col

######################### Function to mask LAB colorspace within provided thresholds ############################
def Thresholding_LABcolor(image, colorspace = 'L', colThresh = (0,255)):
    
    # Converting image to RGB space
    labimage = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
    lspace = labimage[:,:,0]
    aspace = labimage[:,:,1]
    bspace = labimage[:,:,2]
    
    if colorspace == 'L':
        binary_col = np.zeros_like(lspace)
        binary_col[(lspace > colThresh[0]) & (lspace < colThresh[1])] = 1
    elif colorspace == 'A':
        binary_col = np.zeros_like(aspace)
        binary_col[(aspace > colThresh[0]) & (aspace < colThresh[1])] = 1
    else:
        binary_col = np.zeros_like(bspace)
        binary_col[(bspace > colThresh[0]) & (bspace < colThresh[1])] = 1
    
    return binary_col

######################### Function to mask LUV colorspace within provided thresholds ############################
def Thresholding_LUVcolor(image, colorspace = 'L', colThresh = (0,255)):
    
    # Converting image to RGB space
    luvimage = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    lspace = luvimage[:,:,0]
    uspace = luvimage[:,:,1]
    vspace = luvimage[:,:,2]
    
    if colorspace == 'L':
        binary_col = np.zeros_like(lspace)
        binary_col[(lspace > colThresh[0]) & (lspace < colThresh[1])] = 1
    elif colorspace == 'U':
        binary_col = np.zeros_like(uspace)
        binary_col[(uspace > colThresh[0]) & (uspace < colThresh[1])] = 1
    else:
        binary_col = np.zeros_like(vspace)
        binary_col[(vspace > colThresh[0]) & (vspace < colThresh[1])] = 1
    
    return binary_col

########################### Function to apply perspective transformation ########################################
def Perspective_transform(image, image_size, src, dest):
    
    # Calculating perspective transformation matrix for given src and dest points
    M = cv2.getPerspectiveTransform(src, dest)
    Minv = cv2.getPerspectiveTransform(dest, src)
    
    # Warping the image
    warpedimage = cv2.warpPerspective(image, M, image_size, flags=cv2.INTER_LINEAR)
    
    return warpedimage

########################### Function to calculate histogram to detect lane lines ################################
def hist(image):
    
    # Only grabbing bottom half of the image
    bottom_half = image[image.shape[0]//2:,:]
    
    # Summing image pixels vertically to observe peaks
    histogram = np.sum(bottom_half, axis=0)
    
    return histogram

################################### Function to find lane pixels ################################################
def FindLaneLines(warpedimage):
    
    # Only grabbing bottom half of the image
    bottom_half = warpedimage[warpedimage.shape[0]//2:,:]
    
    # Summing image pixels vertically to observe peaks
    histogram = np.sum(bottom_half, axis=0)
    
    # Creating image to visualize results
    out_image = np.dstack((warpedimage, warpedimage, warpedimage))*255
    
    #Finding peaks of lest and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    #Setting up parameters for sliding window
    nwindows = 9 # number of sliding windows
    margin = 100 # window margin
    minpix = 50 # minimum number of pixels found to recenter window
    window_height = np.int(warpedimage.shape[0]//nwindows) # Setting height of window based on number of windows
    
    # To identify X and Y of non zero pixels in image
    nonzero = warpedimage.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # stepping through the window one by one
    for window in range(nwindows):
        
        # Identifying window boundaries in X and Y
        win_y_low = warpedimage.shape[0] - (window+1)*window_height
        win_y_high = warpedimage.shape[0] - window*window_height
        
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Drawing the windows on the visualization image
        cv2.rectangle(out_image,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 4) 
        cv2.rectangle(out_image,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 4)  
    
        # Identifying non-zero pixels in X and Y within window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Appending these indiced to list
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If number of non-zero pixels > min, recenter pixels based on mean position of pixels
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenating arrays of indices 
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extracting left and right pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # Fitting second order polynomial
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, warpedimage.shape[0]-1, warpedimage.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    ## Visualization ##
    # Colors in the left and right lane regions
    out_image[lefty, leftx] = [100, 0, 0]
    out_image[righty, rightx] = [0, 0, 100]
    
    # Plots the left and right polynomials on the lane lines
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')
    
    return out_image, left_fit, right_fit
 
################################### Function to find lane pixels ################################################
def PolyFitUsingPrev(image, left_fit, right_fit):
    
    # Width of margin around previous polynomial to search
    margin = 100
    
    # Grabbing activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]                                                       
                                                          
    # Fit new polynomials
    left_fitx = np.polyfit(lefty, leftx, 2)
    right_fitx = np.polyfit(righty, rightx, 2)
    
    return left_fitx, right_fitx













########################################################################################################
############################################## Main ####################################################
######################################################################################################## 

# Reading image from folder
image = img.imread('test_images/test2.jpg')

# Image size
image_size = (image.shape[1], image.shape[0])

# Loading calibration parameters 
calib = pickle.load( open ("camcalib_data.pkl", "rb"))
mtx1 = calib["mtx"]
dist1 = calib["dist"]

# Undistroting the image
image = cal_undistort(image, mtx1, dist1)

# Performing gradient X and Y, magnitude and direction masking
bin_gradx = Thresholding_grad(image, orient = 'x', sobel_kernel = 3, gradThresh = (30, 100))
bin_grady = Thresholding_grad(image, orient = 'y', sobel_kernel = 3, gradThresh = (30, 100))
bin_mag = Thresholding_mag(image, sobel_kernel = 3, magThresh = (30, 100))
bin_dir = Thresholding_dir(image, sobel_kernel = 3, dirThresh = (0.7, 1.3))

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


f, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3,3, figsize=(24,9))
f.tight_layout()
ax1.imshow(rgb_colr)
ax1.set_title('R space')
ax2.imshow(rgb_colg)
ax2.set_title('G space')
ax3.imshow(rgb_colb)
ax3.set_title('B space')
ax4.imshow(hls_colh)
ax4.set_title('H space')
ax5.imshow(hls_coll)
ax5.set_title('L space')
ax6.imshow(hls_cols)
ax6.set_title('S space')
ax7.imshow(lab_coll)
ax7.set_title('L space')
ax8.imshow(lab_cola)
ax8.set_title('A space')
ax9.imshow(lab_colb)
ax9.set_title('B space')


# Combining B, S and L space
bin_col = np.zeros_like(rgb_colb)
bin_col[ (luv_coll == 1) | (rgb_colb == 1) | (lab_coll == 1)] = 1


# Combining gradient and color thresholding
bin_combcol = np.zeros_like(bin_comb)
bin_combcol[(bin_comb == 1) | (bin_col == 1)] = 1



#Defining src and dest for perspective transformation
src = np.float32([[560,460],
                  [710,460],
                  [260,680],
                  [1050,680]])

dest = np.float32([[100,0],
                  [1050,0],
                  [100,680],
                  [1050,680]])


# Warping the image for bird eye view
warpedimage = Perspective_transform(bin_combcol, image_size, src, dest)

# Finding lana lines using moving window
out_image = FindLaneLines(warpedimage)

# Visualing the stages of pipeline
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(24,9))
f.tight_layout()

ax1.imshow(image)
ax1.set_title('Actual image')
ax2.imshow(bin_combcol)
ax2.set_title('Thresholded image')
ax3.imshow(warpedimage)
ax3.set_title('Warped image')
ax4.imshow(out_image)
ax4.set_title('Lane lines')



    
    
    