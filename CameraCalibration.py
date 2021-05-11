#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  9 18:24:15 2021

@author: varun
"""
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import cv2
import glob
import os

abspath = os.path.abspath('') ## String which contains absolute path to the script file
os.chdir(abspath) ## Setting up working directory


def CameraCalibration(images, nx, ny):

    # Arrays to store object points and image points from all images
    objpoints = []
    imgpoints = []

    # Prepare object points
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
    
    for fname in images:
        
        # Reading the image
        image = img.imread(fname)
    
        # Converting to gray scale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Finding chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

        # If corners are found, add objects points and image points
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)
            
            # Drawing chessboard corners on the image
            #cv2.drawChessboardCorners(image, (nx,ny), corners, ret)
    
    # Performing the camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    return mtx, dist

def cal_undistort(image, mtx, dist):
    
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    return undist

# Defining images from the camera calibration folder        
images = glob.glob('camera_cal/calibration*.jpg')

# Calibrating camera to calculate ditorting matrix
mtx, dist = CameraCalibration(images, 9, 6)

# Defining dictionary to save calibration parameters
calib_params = {"mtx" : mtx, "dist" : dist}

# Saving mtx and dist in camcalib_data.pkl 
cam_calibfile = open("camcalib_data.pkl", "wb")
pickle.dump(calib_params, cam_calibfile)
cam_calibfile.close()

# Performing undistortion using calculated distortion co-efficients
calib = pickle.load( open("camcalib_data.pkl", "rb"))
mtx1 = calib["mtx"]
dist1 = calib["dist"]
image1 = img.imread('camera_cal/calibration1.jpg')
undistort_img = cal_undistort(image1, mtx1, dist1)
# plt.imshow(undistort_img)
# plt.show()
    
        