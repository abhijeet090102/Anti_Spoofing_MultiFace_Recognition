import cv2
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
import os

# Load the pretrained SSD classifier model
face_model_path = 'detection_model'  # Update this path as needed
prototxtPath = os.path.sep.join([face_model_path, "deploy.prototxt"])
weightsPath = os.path.sep.join([face_model_path, "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load the trained models
with open('pca_model.pkl', 'rb') as f:
    pca = pickle.load(f)

with open('svm_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Function to detect faces
def detect_faces(image, net, confidence_threshold=0.5):
    img_height, img_width = image.shape[:2]
    resized_image = cv2.resize(image, (300, 300))
    image_blob = cv2.dnn.blobFromImage(resized_image, 1.0, (300, 300), (104, 177, 123))
    net.setInput(image_blob)
    detections = net.forward()
    
    face_locations = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([img_height, img_width, img_width, img_height])
            (left_x, left_y, right_x, right_y) = box.astype("int")
            face_locations.append((left_x, left_y, right_x, right_y))
    return face_locations

# Start video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to capture video")
        break

    # Detect faces in the frame
    face_locations = detect_faces(frame, net)
    
    for (left_x, left_y, right_x, right_y) in face_locations:
        # Extract the face
        current_face_image = frame[left_y:right_y, left_x:right_x]
        
        # Check if the current face image is empty
        if current_face_image.size == 0:
            continue
        
            # Convert the face image to grayscale if your model was trained on grayscale images
        current_face_image = cv2.cvtColor(current_face_image, cv2.COLOR_BGR2GRAY)


        # Debugging: Print the shape of the current face image
        print(f"[DEBUG] Current face image shape: {current_face_image.shape}")

        # Resize the face image to match the training size (64, 47)
        current_face_image_resized = cv2.resize(current_face_image, (47, 64)).flatten()  # Resize to match training data
        
        # Debugging: Print the size after flattening
        print(f"[DEBUG] Resized and flattened face image size: {current_face_image_resized.size}")

        # Ensure the flattened image has the correct number of features
        if current_face_image_resized.size != 3008:
            print(f"[WARNING] Unexpected feature size: {current_face_image_resized.size}. Expected: 3008.")
            continue
        
        # Predict the identity
        face_pca = pca.transform([current_face_image_resized])
        prediction = model.predict(face_pca)
        pred_name = le.inverse_transform(prediction)[0]
        
        # Draw bounding box and label
        cv2.rectangle(frame, (left_x, left_y), (right_x, right_y), (0, 255, 0), 2)
        cv2.putText(frame, pred_name, (left_x, left_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Real-Time Face Recognition", frame)

    # Break the loop on 'q' key press
    if cv2 .waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()