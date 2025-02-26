# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:33:20 2024

@author: abhij
"""

# Import necessary packages
from imutils.video import VideoStream
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import time as times
import cv2
import os
import face_recognition
from datetime import datetime ,time
import mysql.connector

try:
    db = mysql.connector.connect(
        host="",        # Replace with your MySQL host
        user="amst",             # Replace with your MySQL username
        password="amst0916",     # Replace with your MySQL password
        database="attendancesystem"  # Replace with your database name
    )
    print("Connection Successful")
except mysql.connector.Error as err:
    print(f"Error: {err}")

cursor = db.cursor()


# Paths and Initialization
path = 'ImageAttendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

        
def markAttendance(name):
    # Get current date and time
    now = datetime.now()
    current_date = now.date()
    current_time = now.time()

    # Define time ranges
    in_time_start = time(9, 0, 0)     # 9:00 AM
    in_time_end = time(9, 30, 0)      # 9:30 AM
    out_time_start = time(17, 0, 0)   # 5:00 PM

    # Check if the user has an entry for today
    cursor.execute("SELECT id,name, in_time, out_time FROM attendance WHERE name=%s AND date=%s", (name, current_date))
    result = cursor.fetchone()
    print(result)
    
    if result:
        # Update the "Out Time" if it is missing and the time is after 5:00 PM
        record_id, name_db, in_time, out_time  = result
        print(name_db)
        if name_db == name:
            if out_time is None and current_time >= out_time_start:
                cursor.execute("UPDATE attendance SET out_time=%s WHERE name=%s", (current_time, name_db))
                db.commit()
                print(f"Out time recorded for {name} at {current_time}.")
    else:
        # Insert a new record
        if in_time_start <= current_time <= in_time_end:
            in_time = current_time
            out_time = None
        elif current_time >= out_time_start:
            in_time = None
            out_time = current_time
        else:
            print(f"Time {current_time} does not fall within attendance marking periods.")
            return

        cursor.execute("INSERT INTO attendance (name, date, in_time, out_time) VALUES (%s, %s, %s, %s)",
                    (name, current_date, in_time, out_time))
        db.commit()
        print(f"Attendance recorded for {name}: In Time - {in_time}, Out Time - {out_time}.")
        

# Load the face encodings
encodeListKnown = findEncodings(images)
print('Encoding Complete')


# Argument parser (use defaults for missing args)
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="Minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Load face detector
print("[INFO] Loading face detector...")
protoPath = "detection_model/deploy.prototxt"
modelPath = "detection_model/res10_300x300_ssd_iter_140000.caffemodel"

if not os.path.exists(protoPath) or not os.path.exists(modelPath):
    raise FileNotFoundError("Face detection model files not found!")
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# Load liveness detector and label encoder
print("[INFO] Loading liveness detector...")
# model = load_model("model/liveness_model.h5")
# le = pickle.loads(open("model/label_encoder.pickel", "rb").read())

model = load_model("3Layer_model/model_creation_new.h5")
le = pickle.loads(open("3Layer_model/lebel_model_new.pickel", "rb").read())

# Start video stream
print("[INFO] Starting video stream...")
vs = VideoStream(src=1).start()  # Use src=0 for primary camera
times.sleep(2.0)

try:
    while True:
        # Capture frame and resize
        frame = vs.read()
        frame = imutils.resize(frame, width=800)
        (h, w) = frame.shape[:2]

        # Create blob from image
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                    (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        # Process each detection
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Filter weak detections
            if confidence > args["confidence"]:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Ensure bounding box is within frame dimensions
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(w, endX), min(h, endY)

                # Extract and preprocess face ROI
                face = frame[startY:endY, startX:endX]
                if face.shape[0] == 0 or face.shape[1] == 0:  # Handle empty ROI
                    continue
                face = cv2.resize(face, (32, 32))
                face = face.astype("float") / 255.0
                face = img_to_array(face)
                face = np.expand_dims(face, axis=0)

                # Predict liveness
                preds = model.predict(face)[0]
                j = np.argmax(preds)
                label = le.classes_[j]
                label_text = f"{label}: {preds[j]:.4f}"
                    
                # Draw label and bounding box
                cv2.putText(frame, label_text, (startX, startY - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (222, 0, 255), 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                            (0, 0, 255), 2)

                # If face is real, run it through face recognition
                if label == 'real':
                    face_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face_resized_rgb = cv2.resize(face_rgb, (0, 0), None, 0.25, 0.25)
                    
                    facesCurFrame = face_recognition.face_locations(face_resized_rgb,number_of_times_to_upsample=2,model='dnn')
                    encodesCurFrame = face_recognition.face_encodings(face_resized_rgb, facesCurFrame)

                    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                        matchIndex = np.argmin(faceDis)

                        if matches[matchIndex]:
                            name = classNames[matchIndex].upper()
                            print(f"Recognized: {name}")
                            y1, x2, y2, x1 = faceLoc
                            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                            markAttendance(name)

        # Show frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # Break loop on 'q'
        if key == ord("q"):
            break
finally:
    # Cleanup
    print("[INFO] Cleaning up...")
    cv2.destroyAllWindows()
    vs.stop()
