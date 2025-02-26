from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from skimage.exposure import rescale_intensity
from face import load_face_dataset
from imutils import build_montages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import pickle
from recognize_faces import face_dataset

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True,
	help="path to input directory of images")
ap.add_argument("-f", "--face", type=str,
	default="detection_model",
	help="path to face detector model directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-n", "--num-components", type=int, default=150,
	help="# of principal components")
ap.add_argument("-v", "--visualize", type=int, default=-1,
	help="whether or not PCA components should be visualized")
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the CALTECH faces dataset
print("[INFO] loading dataset...")
(faces, labels) = face_dataset(args["input"],minConfidence=0.5, minSamples=40)
print("[INFO] {} images in dataset".format(len(faces)))


# flatten all 2D faces into a 1D list of pixel intensities
pcaFaces = np.array([f.flatten() for f in faces])
# encode the string labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# construct our training and testing split
split = train_test_split(faces, pcaFaces, labels, test_size=0.20,
	stratify=labels, random_state=42)
(origTrain, origTest, trainX, testX, trainY, testY) = split

# compute the PCA (eigenfaces) representation of the data, then
# project the training data onto the eigenfaces subspace
print("[INFO] creating eigenfaces...")
pca = PCA(
	svd_solver="randomized",
	n_components=args["num_components"],
	whiten=True)
start = time.time()
trainX = pca.fit_transform(trainX)
end = time.time()
print("[INFO] computing eigenfaces took {:.4f} seconds".format(
	end - start))
# check to see if the PCA components should be visualized
if args["visualize"] > 0:
	# initialize the list of images in the montage
	images = []
	# loop over the first 16 individual components
	for (i, component) in enumerate(pca.components_[:16]):
			# Reshape the component to match the input dimensionality
			# For grayscale images (64x64), reshape as follows:
			component = component.reshape((64, 47))


			# Convert the data type to an unsigned 8-bit integer
			component = rescale_intensity(component, out_range=(0, 255))
			component = np.dstack([component.astype("uint8")] * 3)

			images.append(component)

		# Construct the montage for the images
	montage = build_montages(images, (47, 64), (4, 4))[0]
	# show the mean and principal component visualizations
	# show the mean image
	mean = pca.mean_.reshape((64, 47))
	mean = rescale_intensity(mean, out_range=(0, 255)).astype("uint8")
	cv2.imshow("Mean", mean)
	cv2.imshow("Components", montage)
	cv2.waitKey(0)
	# train a classifier on the eigenfaces representation
print("[INFO] training classifier...")
model = SVC(kernel="rbf", C=10.0, gamma=0.001, random_state=42)
model.fit(trainX, trainY)
# evaluate the model
print("[INFO] evaluating model...")
predictions = model.predict(pca.transform(testX))
print(classification_report(testY, predictions,
	target_names=le.classes_))
# generate a sample of testing data
idxs = np.random.choice(range(0, len(testY)), size=10, replace=False)
# loop over a sample of the testing data
for i in idxs:
	# grab the predicted name and actual name
	predName = le.inverse_transform([predictions[i]])[0]
	actualName = le.classes_[testY[i]]
	# grab the face image and resize it such that we can easily see
	# it on our screen
	face = np.dstack([origTest[i]] * 3)
	face = imutils.resize(face, width=250)
	# draw the predicted name and actual name on the image
	cv2.putText(face, "pred: {}".format(predName), (5, 25),
		cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
	cv2.putText(face, "actual: {}".format(actualName), (5, 60),
		cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
	# display the predicted name  and actual name
	print("[INFO] prediction: {}, actual: {}".format(
		predName, actualName))
	# display the current face to our screen
	cv2.imshow("Face", face)
	cv2.waitKey(0)
# After training the model
# Save the PCA model
with open('pca_model.pkl', 'wb') as f:
    pickle.dump(pca, f)
print("[INFO] PCA model saved to pca_model.pkl")

# Save the SVM model
with open('svm_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("[INFO] SVM model saved to svm_model.pkl")

# Save the Label Encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
print("[INFO] Label Encoder saved to label_encoder.pkl")

# Later, to load the models
# Load the PCA model
with open('pca_model.pkl', 'rb') as f:
    pca = pickle.load(f)
print("[INFO] PCA model loaded from pca_model.pkl")

# Load the SVM model
with open('svm_model.pkl', 'rb') as f:
    model = pickle.load(f)
print("[INFO] SVM model loaded from svm_model.pkl")

# Load the Label Encoder
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)
print("[INFO] Label Encoder loaded from label_encoder.pkl")

# import cv2
# import numpy as np
# import os
# import pickle
# from sklearn.preprocessing import LabelEncoder
# from sklearn.decomposition import PCA
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# from imutils import paths

# # Load the pretrained SSD classifier model
# face_model_path = 'detection_model'  # Update this path as needed
# prototxtPath = os.path.sep.join([face_model_path, "deploy.prototxt"])
# weightsPath = os.path.sep.join([face_model_path, "res10_300x300_ssd_iter_140000.caffemodel"])
# net = cv2.dnn.readNet(prototxtPath, weightsPath)

# # Load images from a directory using imutils
# inputPath = 'facedetect_label'  # Update this path as needed
# imagePaths = list(paths.list_images(inputPath))
# names = [p.split(os.path.sep)[-2] for p in imagePaths]
# (names, counts) = np.unique(names, return_counts=True)
# names = names.tolist()

# print(f"[INFO] Found {len(imagePaths)} images belonging to {len(names)} classes.")

# faces = []
# labels = []


# # Prepare data for training
# face_images = []
# face_labels = []

# def detect_faces(image, net, confidence_threshold=0.5):
#     img_height, img_width = image.shape[:2]
#     resized_image = cv2.resize(image, (300, 300))
#     image_blob = cv2.dnn.blobFromImage(resized_image, 1.0, (300, 300), (104, 177, 123))
#     net.setInput(image_blob)
#     detections = net.forward()
    
#     face_locations = []
#     for i in range(detections.shape[2]):
#         confidence = detections[0, 0, i, 2]
#         if confidence > confidence_threshold:
#             box = detections[0, 0, i, 3:7] * np.array([img_height, img_width, img_width, img_height])
#             (left_x, left_y, right_x, right_y) = box.astype("int")
#             face_locations.append((left_x, left_y, right_x, right_y))
#     return face_locations

# # Process each image
# for imagePath in imagePaths:
#     image = cv2.imread(imagePath)
#     if image is None:
#         print(f"[WARNING] Could not read image: {imagePath}")
#         continue
    
#     name = imagePath.split(os.path.sep)[-2]
#     face_locations = detect_faces(image, net)
#     print(f"[INFO] Detected {len(face_locations)} faces in image: {name}")
    
#     for (left_x, left_y, right_x, right_y) in face_locations:
#         # Ensure the coordinates are within the image bounds
#         if left_x < 0 or left_y < 0 or right_x > image.shape[1] or right_y > image.shape[0]:
#             print(f"[WARNING] Invalid bounding box for image {name}: ({left_x}, {left_y}, {right_x}, {right_y})")
#             continue
        
#         current_face_image = image[left_y:right_y, left_x:right_x]
        
#         # Check if the current face image is empty
#         if current_face_image.size == 0:
#             print(f"[WARNING] Detected face image is empty for {name}.")
#             continue
        
#         # Resize and flatten the face image
#         current_face_image_resized = cv2.resize(current_face_image, (64, 47)).flatten()  # Resize to match training data
#         face_images.append(current_face_image_resized)
#         face_labels.append(name)

# # Convert to numpy arrays
# face_images = np.array(face_images)
# face_labels = np.array(face_labels)

# # Check if any faces were detected
# if len(face_images) == 0:
#     print("[ERROR] No faces detected. Please check your images and face detection model.")
#     exit()
# # Encode the string labels as integers
# le = LabelEncoder()
# face_labels_encoded = le.fit_transform(face_labels)

# # Split the dataset into training and testing sets
# trainX, testX, trainY, testY = train_test_split(face_images, face_labels_encoded, test_size=0.20, random_state=42)

# # Compute PCA (eigenfaces)
# print("[ INFO] creating eigenfaces...")
# pca = PCA(n_components=150, whiten=True)
# trainX = pca.fit_transform(trainX)

# # Train the SVM classifier
# print("[INFO] training classifier...")
# model = SVC(kernel="rbf", C=10.0, gamma=0.001, random_state=42)
# model.fit(trainX, trainY)

# # Evaluate the model
# print("[INFO] evaluating model...")
# predictions = model.predict(pca.transform(testX))

# # Check unique classes in predictions
# unique_predictions = np.unique(predictions)
# print(f"[INFO] Unique classes in predictions: {unique_predictions}")

# # Get the unique classes from the label encoder
# unique_classes = le.classes_
# print(f"[INFO] Unique classes from label encoder: {unique_classes}")

# # Generate the classification report
# try:
#     print(classification_report(testY, predictions, target_names=unique_classes, labels=np.unique(predictions)))
# except ValueError as e:
#     print(f"[ERROR] {e}")
    
# # Save the trained models
# with open('face_detector/pca_model.pkl', 'wb') as f:
#     pickle.dump(pca, f)
# print("[INFO] PCA model saved to pca_model.pkl")

# with open('face_detector/svm_model.pkl', 'wb') as f:
#     pickle.dump(model, f)
# print("[INFO] SVM model saved to svm_model.pkl")

# with open('face_detector/label_encoder.pkl', 'wb') as f:
#     pickle.dump(le, f)
# print("[INFO] Label Encoder saved to label_encoder.pkl")