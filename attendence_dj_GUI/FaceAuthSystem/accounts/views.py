from django.shortcuts import render, redirect
from .forms import UserRegistrationForm
from django.http import StreamingHttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators import gzip
from imutils.video import VideoStream
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import face_recognition
import numpy as np
import imutils
import pickle
from datetime import datetime ,time
import cv2
import os
import time as times 
import threading
from .models import Attendance , User


def register_user(request):
    form = UserRegistrationForm()
    if request.method == "POST":
        form = UserRegistrationForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('accounts:login')
        else:
            form = UserRegistrationForm()
    return render(request, 'accounts/register.html', {'form':form})

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

# Load images and encodings

path = 'C:\\Users\\abhij\\attendence_system_using\\attendence_dj_GUI\\FaceAuthSystem\\accounts\\static\\images'
images = []
classNames = []
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

    # Load the face encodings
encodeListKnown = findEncodings(images)
print('Encoding Complete')


class attendance_video():

    def face_authenticate(self):
        # Load face detector
        print("[INFO] Loading face detector...")

        # Load liveness detector and label encoder
        print("[INFO] Loading liveness detector...")
        
        # Load face detector
        protoPath = "C:\\Users\\abhij\\attendence_system_using\\FaceAttendence\\detection_model\\deploy.prototxt"
        modelPath = "C:\\Users\\abhij\\attendence_system_using\\FaceAttendence\\detection_model\\res10_300x300_ssd_iter_140000.caffemodel"
        net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

        # Load liveness detector and label encoder
        model = load_model("C:\\Users\\abhij\\attendence_system_using\\FaceAttendence\\3Layer_model\\model_creation_new.h5")
        le = pickle.loads(open("C:\\Users\\abhij\\attendence_system_using\\FaceAttendence\\3Layer_model\\lebel_model_new.pickel", "rb").read())

        if not os.path.exists(protoPath) or not os.path.exists(modelPath):
            raise FileNotFoundError("Face detection model files not found!")
        net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

        # Start video stream
        self.vs = VideoStream(src=1).start()  # Use src=0 for primary camera
        times.sleep(2.0)


        while True:
        # Capture frame and resize
            if self.vs is None:
                break
            frame = self.vs.read()
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
                if confidence > 0.5:
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

                    # # If face is real, run it through face recognition
                    # if label == 'real':
                    #     face_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    #     face_resized_rgb = cv2.resize(face_rgb, (0, 0), None, 0.25, 0.25)
                        
                    #     facesCurFrame = face_recognition.face_locations(face_resized_rgb,number_of_times_to_upsample=2,model='dnn')
                    #     encodesCurFrame = face_recognition.face_encodings(face_resized_rgb, facesCurFrame)
                    #     recognized = False
                    #     for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                    #         matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                    #         faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                    #         matchIndex = np.argmin(faceDis)
                    #         if faceDis.size > 0:
                    #             matchIndex = np.argmin(faceDis)
                    #             if matches[matchIndex]:
                    #                 recognized = True
                    #                 name = classNames[matchIndex].upper()
                    #                 print(f"Recognized: {name}")
                    #                 y1, x2, y2, x1 = faceLoc
                    #                 y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    #                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    #                 cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    #                 cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    #                 markAttendance(name)
                    #                             # If no match was found, label as "Unknown"
                    #     if not recognized:
                    #         for faceLoc in facesCurFrame: 
                    #             y1, x2, y2, x1 = faceLoc
                    #             y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    #             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box for unknown
                    #             cv2.putText(frame, "Unknown", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    # If face is real, run it through face recognition
                    if label == 'real':
                        face_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        face_resized_rgb = cv2.resize(face_rgb, (0, 0), None, 0.25, 0.25)
                        
                        facesCurFrame = face_recognition.face_locations(face_resized_rgb, number_of_times_to_upsample=2, model='dnn')
                        encodesCurFrame = face_recognition.face_encodings(face_resized_rgb, facesCurFrame)

                        recognized = False  # Flag to check if face is recognized

                        # Check if any faces were found
                        if len(encodesCurFrame) > 0:
                            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

                                # Check if faceDis is not empty before proceeding
                                if faceDis.size > 0:
                                    matchIndex = np.argmin(faceDis)

                                    if matches[matchIndex]:
                                        recognized = True  # Set flag to True if a match is found
                                        name = classNames[matchIndex].upper()
                                        print(f"Recognized: {name}")
                                        y1, x2, y2, x1 = faceLoc
                                        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                        cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                                        cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                                        markAttendance(name)

                        # If no match was found, label as "Unknown"
                        if not recognized:
                            for faceLoc in facesCurFrame: 
                                y1, x2, y2, x1 = faceLoc
                                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box for unknown
                                cv2.putText(frame, "Unknown", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            _, jpeg = cv2.imencode('.jpg', frame)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    def stop_video_feed(self):
        if self.vs is not None:
            
            self.vs.stop()
            cv2.destroyAllWindows()  # Ensure windows are destroyed
            self.running = False
            self.vs = None  # Clear the reference to the video stream
            print("Video feed stopped.")
        else:
            print("No active video feed to stop.")
def stop_video(request):  
    global current_video
    if current_video:
        current_video.stop_video_feed()
        current_video = None  # Clean up OpenCV windows
        print("Video feed stopped.")
        return JsonResponse({"status": "success", "message": "Video stopped"})
    else:
        return JsonResponse({"status": "error", "message": "No active video feed"})

def login(request):
    return render(request, 'accounts/login.html')

current_video = None
@gzip.gzip_page
def video_feed(request):
    global current_video
    if current_video is None:
        current_video = attendance_video()
    
    return StreamingHttpResponse(current_video.face_authenticate(), content_type='multipart/x-mixed-replace; boundary=frame') 

def admin_dashboard(request):
	attendance_records = Attendance.objects.all()
	return render(request, 'accounts/admin_dashboard.html', {'attendence' : attendance_records})

def User_register(request):
    users = User.objects.all()
    return render(request , 'accounts/register_data.html',{'users':users})