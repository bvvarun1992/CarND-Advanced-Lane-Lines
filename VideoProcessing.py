#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 23:26:52 2021

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


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None  
        
        
    def LaneFinding(self, warpedimage):
        
        # Creating image to visualize results
        out_image = np.dstack((warpedimage, warpedimage, warpedimage))*255
        margin = 100 # window margin
        
        if self.detected == False:
            # Only grabbing bottom half of the image
            bottom_half = warpedimage[warpedimage.shape[0]//2:,:]
    
            # Summing image pixels vertically to observe peaks
            histogram = np.sum(bottom_half, axis=0)
            
            #Finding peaks of left and right lines
            midpoint = np.int(histogram.shape[0]//2)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint
            
            #Setting up parameters for sliding window
            nwindows = 9 # number of sliding windows
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
            
            self.detected = True
        
        else:
            
            # To identify X and Y of non zero pixels in image
            nonzero = warpedimage.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            
            left_fit, right_fit = self.best_fit
            
            # Identifying non-zero pixels in X and Y within window
            left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
            right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin))) 
            
        
        # Extracting left and right pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        # Fitting second order polynomial
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        
        # Updating the fit values
        self.current_fit = [left_fit, right_fit]
        self.recent_xfitted.append(self.current_fit)
        
        # Calculating best fit based on mean of previous iterations
        self.best_fit = np.mean(self.recent_xfitted, axis=0)   

        # Generate x and y values for plotting
        ploty = np.linspace(0, warpedimage.shape[0]-1, warpedimage.shape[0] )
        
        # Saving ploty
        self.ploty = ploty
        
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]         
        
        ## Visualization ##
        # Colors in the left and right lane regions
        out_image[lefty, leftx] = [255, 0, 0]
        out_image[righty, rightx] = [0, 0, 255]
        
        return out_image, leftx, lefty, rightx, righty, left_fitx, right_fitx, ploty

    def imageProcesing(self, image):
            
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
        
        rgb_colb = Thresholding_RGBcolor(image, colorspace = 'B', colThresh = (220,255))
        lab_coll = Thresholding_LABcolor(image, colorspace = 'L', colThresh = (230, 255))
        luv_coll = Thresholding_LUVcolor(image, colorspace = 'L', colThresh = (210, 255))
        
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
        
        # Finding lane lines and fitting polygon
        out_image, leftx, lefty, rightx, righty, left_fitx, right_fitx, ploty = self.LaneFinding(warpedimage)
        
        # computing radius of curvature and offset from center of lane
        left_curverad, right_curverad, offset = FindingCurvature(leftx, lefty, rightx, righty, image_size)
    
        # Retransforming the warped image and drawing the fit lane lines
        final_image = DrawLines(image, warpedimage, left_fitx, right_fitx, ploty, Minv)
        
        # Dsiplaying curvature and vehicle position from center
        cv2.putText(final_image, 'Left lane radius of curvature: '+ str(np.round(left_curverad,2))+'m', (50,100), fontFace = 16, fontScale = 1, color = (255,255,255), thickness = 2)
        cv2.putText(final_image, 'Right lane radius of curvature: '+ str(np.round(right_curverad,2))+'m', (50,150), fontFace = 16, fontScale = 1, color = (255,255,255), thickness = 2)
        cv2.putText(final_image, 'Car offset by: ' + str(np.round(offset,2)) + 'm', (50,200), fontFace = 16, fontScale = 1, color = (255,255,255), thickness = 2)
        
        return final_image.astype(np.uint8)

#######################################################################################
#################   VIDEO PIPELINE   ##################################################
#######################################################################################

from moviepy.editor import VideoFileClip
framelines = Line()
videoinput = VideoFileClip("project_video.mp4")
videoclip = videoinput.fl_image(framelines.imageProcesing)
videoclip.write_videofile('Output_video.mp4', audio=False)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    