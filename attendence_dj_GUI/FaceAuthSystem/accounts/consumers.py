import base64
import json
from channels.generic.websocket import AsyncWebsocketConsumer
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import face_recognition
from .models import Attendance, User
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import face_recognition
import os
import pickle
from datetime import datetime ,time


class FaceAuthConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        self.path = 'C:\\Users\\abhij\\attendence_system_using\\FaceAttendence\\ImageAttendance'
        self.images = []
        self.classNames = []
        myList = os.listdir(self.path)
        for cl in myList:
            curImg = cv2.imread(f'{self.path}/{cl}')
            self.images.append(curImg)
            self.classNames.append(os.path.splitext(cl)[0])
        def findEncodings(images):
            encodeList = []
            for img in images:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encode = face_recognition.face_encodings(img)[0]
                encodeList.append(encode)
            return encodeList

            # Load the face encodings
        self.encodeListKnown = findEncodings(self.images)
        print('Encoding Complete')

    async def disconnect(self, close_code):
        print("WebSocket disconnected:", close_code)

    async def receive(self, text_data):
        data = json.loads(text_data)
        command = data.get('command')

        if command == "start":
            # Initialize models and resources
            protoPath = "C:\\Users\\abhij\\attendence_system_using\\FaceAttendence\\detection_model\\deploy.prototxt"
            modelPath = "C:\\Users\\abhij\\attendence_system_using\\FaceAttendence\\detection_model\\res10_300x300_ssd_iter_140000.caffemodel"
            liveness_model_path = "C:\\Users\\abhij\\attendence_system_using\\FaceAttendence\\3Layer_model\\model_creation_new.h5"
            label_encoder_path = "C:\\Users\\abhij\\attendence_system_using\\FaceAttendence\\3Layer_model\\lebel_model_new.pickel"

            if not (os.path.exists(protoPath) and os.path.exists(modelPath)):
                await self.send(json.dumps({"status": "error", "message": "Face detection model files not found!"}))
                return

            # Load the models and encodings
            self.net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
            self.model = load_model(liveness_model_path)
            with open(label_encoder_path, "rb") as f:
                self.le = pickle.load(f)
            self.encodeListKnown = self.load_encodings()
            self.classNames = self.load_class_names()

            await self.send(json.dumps({"status": "ready", "message": "Models loaded successfully!"}))

        # Start video stream


        elif command == "frame":
            # Decode the frame data from Base64
            frame_data = data.get('frame')
            if not frame_data:
                await self.send(json.dumps({"status": "error", "message": "Frame data not received!"}))
                return

            image_data = base64.b64decode(frame_data.split(',')[1])
            frame = np.array(Image.open(BytesIO(image_data)))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            (h, w) = frame.shape[:2]

            # Create blob for face detection
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                        (300, 300), (104.0, 177.0, 123.0))
            self.net.setInput(blob)
            detections = self.net.forward()

            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    startX, startY = max(0, startX), max(0, startY)
                    endX, endY = min(w, endX), min(h, endY)

                    # Extract face ROI and preprocess
                    face = frame[startY:endY, startX:endX]
                    if face.shape[0] == 0 or face.shape[1] == 0:
                        continue
                    face = cv2.resize(face, (32, 32))
                    face = face.astype("float") / 255.0
                    face = img_to_array(face)
                    face = np.expand_dims(face, axis=0)

                    # Predict liveness
                    preds = self.model.predict(face)[0]
                    j = np.argmax(preds)
                    label = self.le.classes_[j]
                    if label == 'real':
                        face_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        face_resized_rgb = cv2.resize(face_rgb, (0, 0), None, 0.25, 0.25)

                        facesCurFrame = face_recognition.face_locations(face_resized_rgb, number_of_times_to_upsample=2, model='cnn')
                        encodesCurFrame = face_recognition.face_encodings(face_resized_rgb, facesCurFrame)

                        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                            matches = face_recognition.compare_faces(self.encodeListKnown, encodeFace)
                            faceDis = face_recognition.face_distance(self.encodeListKnown, encodeFace)
                            matchIndex = np.argmin(faceDis)

                            if matches[matchIndex]:
                                name = self.classNames[matchIndex].upper()
                                y1, x2, y2, x1 = faceLoc
                                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                                self.mark_attendance(name)
                                await self.send(json.dumps({"status": "success", "name": name}))
                            else:
                                await self.send(json.dumps({"status": "error", "message": "Face not recognized!"}))

        elif command == "stop":
            self.vs.stop()
            cv2.destroyAllWindows()
            await self.send(json.dumps({"status": "stopped", "message": "Video feed stopped."}))
    def markAttendance(name):
        # Get current date and time
        now = datetime.now()
        current_date = now.date()
        current_time = now.time()

        # Define time ranges
        in_time_start = time(9, 0, 0)     # 9:00 AM
        in_time_end = time(9, 30, 0)      # 9:30 AM
        out_time_start = time(17, 00, 0)   # 5:00 PM

            # Determine in_time and out_time
        in_time = current_time if in_time_start <= current_time <= in_time_end else None
        out_time = current_time if current_time >= out_time_start else None
        
        users = User.objects.filter(name__iexact=name)  # Case-insensitive search
        if not users.exists():
            print(f"No user found with the name '{name}'.")
            return
        # created = False
        # attendance_record = None

        # Get or create attendance record for the current date
        for user in users:
            # Check if an attendance record already exists for the current date
            attendance_record = Attendance.objects.filter(user_id=user.user_id, date=current_date).first()
            
            if attendance_record:
                # If an attendance record exists, check if in_time is filled
                if attendance_record.in_time and not attendance_record.out_time:
                    # Update the out_time if it's not already filled and within the time range
                    if current_time >= out_time_start:
                        attendance_record.out_time = current_time
                        attendance_record.save()
                        print(f"Out time recorded for {name} at {current_time}.")
                    else:
                        print(f"Cannot record out time for {name} before 5:00 PM.")
                else:
                    print(f"Attendance already marked for {name} on {current_date}.")
            else :
                if in_time is not None and out_time is None:
                    # If no attendance record exists, create a new one
                    attendance_record = Attendance.objects.create(
                        user_id=user.user_id,
                        name=name,
                        date=current_date,
                        in_time=in_time,
                        out_time=out_time
                    )
                    print(f"Attendance recorded for {name}: In Time - {attendance_record.in_time}, Out Time - {attendance_record.out_time}.")
                elif in_time is None and out_time is None:
        # Prevent recording attendance when both in_time and out_time are None
                    print(f"Cannot record attendance for {name} as both in_time and out_time are None.")
                else:
                    print(f"Cannot record attendance for {name} before 9:00 AM or after 5:00 PM.")

            break  # Exit loop after processing the first matched user

