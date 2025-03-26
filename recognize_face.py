import cv2
import numpy as np
import os
import csv
from datetime import datetime

# Load Haar Cascade classifier
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()

# ✅ Ensure trainer.yml exists before reading
if not os.path.exists("trainer.yml"):
    print("❌ Error: 'trainer.yml' not found! Train the model first.")
    exit()

recognizer.read("trainer.yml")

# ✅ Load label dictionary properly
label_dict = {}
reverse_label_dict = {}

dataset_path = "dataset"
if not os.path.exists(dataset_path):
    print("⚠️ No dataset folder found!")
    exit()

image_files = [f for f in os.listdir(dataset_path) if f.endswith(".jpg")]

if not image_files:
    print("⚠️ No training images found in 'dataset' folder!")
    exit()

for file in image_files:
    label = file.split("_")[0]
    if label not in label_dict:
        label_dict[label] = len(label_dict)
        reverse_label_dict[label_dict[label]] = label  # Reverse mapping

print("🔹 Label Dictionary:", label_dict)
print("🔹 Reverse Label Dictionary:", reverse_label_dict)

# ✅ Create or open CSV file for attendance
csv_filename = "attendance.csv"
if not os.path.exists(csv_filename):
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Date", "Time"])  # CSV headers

# ✅ Open webcam
cam = cv2.VideoCapture(0)

recognized_students = set()  # Avoid duplicate entries in one session

while True:
    ret, frame = cam.read()
    if not ret:
        print("❌ Camera Error: Unable to capture frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_resized = cv2.resize(gray[y:y+h, x:x+w], (200, 200))  # Resize to match training

        try:
            label, confidence = recognizer.predict(face_resized)

            print(f"🔍 Predicted Label: {label}, Confidence: {confidence}")

            if label in reverse_label_dict:
                name = reverse_label_dict[label]

                if confidence < 70:
                    if name not in recognized_students:
                        recognized_students.add(name)
                        now = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")

                        # ✅ Append to CSV
                        with open(csv_filename, mode="a", newline="") as file:
                            writer = csv.writer(file)
                            writer.writerow([name, now.split(", ")[0], now.split(", ")[1]])

                        print(f"✅ Attendance Marked: {name} at {now}")

                    text = f"{name} ({round(confidence, 2)})"
                    cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                else:
                    print(f"⚠️ Low Confidence ({confidence}) - Face Not Recognized")
            else:
                print(f"❌ Error: Label {label} not found in dictionary!")

        except Exception as e:
            print(f"❌ Error recognizing face: {e}")

    cv2.imshow("Face Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ✅ Release camera and close windows
cam.release()
cv2.destroyAllWindows()
print("📌 Camera closed. Attendance saved to 'attendance.csv'.")
