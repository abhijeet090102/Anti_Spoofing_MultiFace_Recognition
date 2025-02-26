# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 21:37:57 2024

@author: abhij
"""

# import the necessary packages
import numpy as np
import argparse
import cv2
import os
import face_recognition
from imutils import paths
import dlib

# open a pointer to the video file stream and initialize the total
# number of frames read and saved thus far
def face_dataset(inputPath,  minConfidence=0.5, minSamples=15):
	imagePaths = list(paths.list_images(inputPath))
	names = [p.split(os.path.sep)[-2] for p in imagePaths]
	(names, counts) = np.unique(names, return_counts=True)
	names = names.tolist()

	faces = []
	labels = []

	for imagePath in imagePaths:
		image = cv2.imread(imagePath)
		name = imagePath.split(os.path.sep)[-2]

		#create a grayscale image to pass into the dlib HOG detector
		image_to_detect_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		#load the pretrained HOG SVN model
		face_detection_classifier = dlib.get_frontal_face_detector()

		#detect all face locations using the HOG SVN classifier
		all_face_locations = face_detection_classifier(image,1)

		if counts[names.index(name)] < minSamples:
			continue

		# Detect faces in the image
		# face_locations = face_recognition.face_locations(image, model="dnn",number_of_times_to_upsample=1)
		for index, face_location in enumerate(all_face_locations):

			    #start and end co-ordinates
			# left_x, left_y, right_x, right_y = face_location.left(), face_location.top(),face_location.right(),face_location.bottom()
			
			left_x = max(0, face_location.left())
			left_y = max(0, face_location.top())
			right_x = min(image.shape[1], face_location.right())
			right_y = min(image.shape[0], face_location.bottom())

			
			#printing the location of current face
			# print('Found face {} at left_x:{},left_y:{},right_x:{},right_y:{}'.format(index+1,left_x,left_y,right_x,right_y))
			#slicing the current face from main image
			current_face_image = image[left_y:right_y,left_x:right_x]


			cv2.rectangle(image,(left_x,left_y),(right_x,right_y),(0,0,255),2)
			            # Handle empty face crops
			if current_face_image.shape[0] == 0 or current_face_image.shape[1] == 0:
				print(f"[WARNING] Empty face crop detected in {imagePath}, skipping...")
				continue
			current_face_image = cv2.cvtColor(current_face_image, cv2.COLOR_BGR2GRAY)
			current_face_image = cv2.resize(current_face_image, (47, 64))
			# face = cv2.resize(face, (64, 64))

			print(f'Found {name} face {index + 1} at top_y:{left_y}, left_x:{left_x}, right_x:{right_x}, right_y:{right_y}')
			faces.append(current_face_image)
			labels.append(name)

	faces = np.array(faces)
	labels = np.array(labels)
		# Return faces and their corresponding labels
	return (faces, labels)