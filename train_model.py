import cv2
import os
import numpy as np

# Load Haar Cascade classifier for face detection
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Create a face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Prepare training data
face_data = []
labels = []
label_dict = {}
label_count = 0

for file in os.listdir("dataset"):
    if file.endswith(".jpg"):
        img = cv2.imread(f"dataset/{file}", cv2.IMREAD_GRAYSCALE)
        face_data.append(np.array(img, dtype=np.uint8))
        
        # Extract label (name from filename)
        label = file.split("_")[0]
        
        if label not in label_dict:
            label_dict[label] = label_count
            label_count += 1
        
        labels.append(label_dict[label])

# Convert lists to numpy arrays
labels = np.array(labels)

# Train the face recognizer
if len(face_data) > 0:
    recognizer.train(face_data, labels)
    recognizer.save("trainer.yml")  # Save trained model
    print("✅ Model trained and saved as 'trainer.yml'")
else:
    print("⚠️ No faces found for training.")
