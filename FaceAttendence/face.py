from imutils import paths
import numpy as np
import cv2
import os
import argparse

def detect_faces(net, image, minConfidence=0.5):

    # Grab the dimensions of the image and then construct a blob from it
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))
    # blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    # Pass the blob through the network to obtain the face detections
    net.setInput(blob)
    detections = net.forward()
	# ensure at least one face was found
    boxes = []
    
    # Loop over the detections
    for i in range(detections.shape[2]):
        # Extract the confidence (i.e., probability) associated with the detection
        
        confidence = detections[0, 0, i, 2]
        # Filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence > minConfidence:
            # Compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            boxes.append((startX, startY, endX, endY))
    # Return the face detection bounding boxes
    return boxes

def load_face_dataset(inputPath, net, minConfidence=0.5, minSamples=15):
    imagePaths = list(paths.list_images(inputPath))
    names = [p.split(os.path.sep)[-2] for p in imagePaths]
    (names, counts) = np.unique(names, return_counts=True)
    names = names.tolist()

    faces = []
    labels = []

    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        name = imagePath.split(os.path.sep)[-2]

        if counts[names.index(name)] < minSamples:
            continue

        boxes = detect_faces(net, image, minConfidence)

        if len(boxes) == 0:
            print(f"[INFO] No faces detected in {imagePath}")
            continue

        for (startX, startY, endX, endY) in boxes:
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(image.shape[1], endX)
            endY = min(image.shape[0], endY)

            # Validate bounding box coordinates
            if startX < 0 or startY < 0 or endX > image.shape[1] or endY > image.shape[0]:
                print(f"[WARNING] Invalid bounding box: {(startX, startY, endX, endY)}")
                continue

            faceROI = image[startY:endY, startX:endX]
            if faceROI.size == 0:  # Check if the ROI is still empty after clamping
                print(f"[WARNING] Empty ROI after clamping: {(startX, startY, endX, endY)}")
                continue


            try:
                faceROI = cv2.resize(faceROI, (62, 62))
                faceROI = cv2.cvtColor(faceROI, cv2.COLOR_BGR2GRAY)
                faces.append(faceROI)
                labels.append(name)
            except Exception as e:
                print(f"[ERROR] Failed to process ROI for {imagePath}: {e}")
                continue

    faces = np.array(faces)
    labels = np.array(labels)

    return (faces, labels)