import streamlit as st
import cv2
import os
import numpy as np
import csv
from datetime import datetime, timedelta
import pandas as pd

# Ensure required directories exist
os.makedirs("dataset", exist_ok=True)
os.makedirs("trained_model", exist_ok=True)

# Create a centralized attendance tracking file
ATTENDANCE_TRACKING_FILE = "all_attendance_tracking.csv"

st.title("📚 Subject-Specific Attendance System")

# Initialize session state
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False
if "attendance_start_time" not in st.session_state:
    st.session_state.attendance_start_time = None

# Sidebar Navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Register Face", "Train Model", "Recognize & Mark Attendance", "View Attendance"])

# ---- FUNCTION: Capture Face ----
def capture_face(name):
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        st.error("❌ Error: Cannot open camera")
        return

    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    count = 0

    # Create an empty placeholder for dynamic image updates
    image_placeholder = st.empty()

    while count < 10:
        ret, frame = cam.read()
        if not ret:
            st.error("❌ Error: Failed to capture frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face_resized = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            filename = f"dataset/{name}_{count}.jpg"
            cv2.imwrite(filename, face_resized)

      # Update only one frame dynamically
        image_placeholder.image(frame, channels="BGR", caption="Capturing Faces...")

    cam.release()
    st.success(f"✅ {count} images captured and saved!")


    # ---- FUNCTION: Train Model ----
def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_data, labels = [], []
    label_dict, label_count = {}, 0

    for file in os.listdir("dataset"):
        if file.endswith(".jpg"):
            img = cv2.imread(f"dataset/{file}", cv2.IMREAD_GRAYSCALE)
            face_data.append(np.array(img, dtype=np.uint8))
            
            label = file.split("_")[0]
            if label not in label_dict:
                label_dict[label] = label_count
                label_count += 1
            
            labels.append(label_dict[label])

    labels = np.array(labels)

    if len(face_data) > 0:
        recognizer.train(face_data, labels)
        recognizer.save("trained_model/trainer.yml")

        # Save label dictionary
        with open("trained_model/labels.csv", "w", newline="") as file:
            writer = csv.writer(file)
            for name, label in label_dict.items():
                writer.writerow([label, name])

        st.success("✅ Model trained and saved!")
    else:
        st.warning("⚠ No faces found for training.")


# Wednesday Class Schedule
WEDNESDAY_SCHEDULE = {
    "AOA": [(datetime.now().replace(hour=9, minute=0, second=0, microsecond=0), 
             datetime.now().replace(hour=11, minute=0, second=0, microsecond=0))],
    "Maths": [(datetime.now().replace(hour=10, minute=0, second=0, microsecond=0), 
               datetime.now().replace(hour=12, minute=0, second=0, microsecond=0))],
    "MP": [(datetime.now().replace(hour=11, minute=0, second=0, microsecond=0), 
            datetime.now().replace(hour=13, minute=0, second=0, microsecond=0))],
    "DBMS": [(datetime.now().replace(hour=13, minute=0, second=0, microsecond=0), 
              datetime.now().replace(hour=14, minute=0, second=0, microsecond=0))],
    "Python": [(datetime.now().replace(hour=14, minute=0, second=0, microsecond=0), 
                datetime.now().replace(hour=16, minute=0, second=0, microsecond=0))]
}

def initialize_attendance_tracking():
    """
    Initialize the centralized attendance tracking file if it doesn't exist
    """
    if not os.path.exists(ATTENDANCE_TRACKING_FILE):
        with open(ATTENDANCE_TRACKING_FILE, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Date", "Subject", "Time"])

def is_attendance_already_marked(name, subject, date):
    """
    Check if attendance is already marked for a student in a specific subject on a given date
    """
    try:
        df = pd.read_csv(ATTENDANCE_TRACKING_FILE)
        existing_record = df[
            (df['Name'] == name) & 
            (df['Subject'] == subject) & 
            (df['Date'] == date)
        ]
        return not existing_record.empty
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return False

def is_valid_attendance_time(current_time=None):
    """
    Check if current time is within Wednesday's class schedule
    Returns the current subject or None
    """
    if current_time is None:
        current_time = datetime.now()
    
    # Optional: Remove strict Wednesday check for testing
    # if current_time.weekday() != 2:  # 2 represents Wednesday
    #     st.warning("Attendance can only be marked on Wednesday!")
    #     return None
    
    for subject, time_ranges in WEDNESDAY_SCHEDULE.items():
        for start, end in time_ranges:
            if start <= current_time <= end:
                return subject
    
    return None

def is_live_face(frame):
    """
    Comprehensive and more tolerant live face detection
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Enhanced face detection with multiple cascades
    face_cascades = [
        cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml"),
        cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
    ]
    
    faces = []
    for cascade in face_cascades:
        detected_faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
        faces.extend(detected_faces)
    
    if len(faces) == 0:
        return False
    
    # Select the largest face
    face = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = face
    
    # Advanced blur detection
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 20:  # Very lenient blur threshold
        return False
    
    # Flexible aspect ratio and size checks
    aspect_ratio = w / h
    if aspect_ratio < 0.3 or aspect_ratio > 2.0:
        return False
    
    return True

def recognize_face():
    """
    More robust face recognition with one-time attendance marking
    """
    # Initialize attendance tracking file
    initialize_attendance_tracking()

    # Check if trained model exists
    if not os.path.exists("trained_model/trainer.yml") or not os.path.exists("trained_model/labels.csv"):
        st.error("❌ No trained model found. Train the model first!")
        return

    # Check if it's a valid time for attendance
    current_subject = is_valid_attendance_time()
    if not current_subject:
        st.warning("❌ Attendance not allowed at this time!")
        return

    st.session_state.camera_active = True
    st.success(f"✅ Marking attendance for {current_subject}")

    # Load recognizer and labels
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("trained_model/trainer.yml")

        label_dict = {}
        with open("trained_model/labels.csv", "r") as file:
            reader = csv.reader(file)
            for row in reader:
                label_dict[int(row[0])] = row[1]
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return

    # Camera setup with enhanced error handling
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        st.error("❌ Cannot open camera")
        return

    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    recognized_students = set()
    start_time = datetime.now()
    attendance_duration = timedelta(minutes=20)  # Extended duration

    image_placeholder = st.empty()
    time_left_placeholder = st.empty()
    
    while st.session_state.camera_active:
        current_time = datetime.now()
        today_date = current_time.strftime("%Y-%m-%d")
        time_elapsed = current_time - start_time
        
        # Attendance time check
        if time_elapsed >= attendance_duration:
            st.warning("🕒 Attendance marking time has expired!")
            break

        # Time remaining display
        time_remaining = attendance_duration - time_elapsed
        time_left_placeholder.warning(f"⏰ Time Remaining: {time_remaining.seconds} seconds")

        ret, frame = cam.read()
        if not ret:
            st.error("❌ Failed to capture frame")
            break

        # Relaxed live face check
        if not is_live_face(frame):
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            face_region = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_region, (200, 200))
            
            try:
                label, confidence = recognizer.predict(face_resized)

                if label not in label_dict:
                    st.error(f"❌ Unknown label: {label}")
                    continue

                name = label_dict[label]

                # Much more lenient confidence threshold
                if confidence < 100:  # Increased to allow more matches
                    # Check if attendance already marked for this subject today
                    if is_attendance_already_marked(name, current_subject, today_date):
                        st.warning(f"❌ {name} has already marked attendance for {current_subject} today!")
                        continue

                    if name not in recognized_students:
                        recognized_students.add(name)
                        now = datetime.now()

                        # Write to centralized attendance tracking file
                        with open(ATTENDANCE_TRACKING_FILE, mode="a", newline="") as file:
                            writer = csv.writer(file)
                            writer.writerow([
                                name, 
                                today_date, 
                                current_subject,
                                now.strftime("%H:%M:%S")
                            ])

                        st.success(f"✅ Attendance Marked: {name} for {current_subject}")

                    # Display name and confidence on frame
                    cv2.putText(frame, f"{name} ({confidence:.2f})", 
                                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.9, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            except Exception as e:
                st.error(f"❌ Face recognition error: {e}")

        image_placeholder.image(frame, channels="BGR", 
                                caption=f"Recognizing Faces for {current_subject}")

    cam.release()
    st.session_state.camera_active = False
    st.success(f"📌 Attendance marking completed for {current_subject}")

# The rest of the code remains the same
# (capture_face, train_model, and page routing functions)

# Keeping the existing page routing code from previous implementation
if page == "Register Face":
    st.subheader("🆕 Register New Student")
    name = st.text_input("Enter Student Name:")
    if st.button("📸 Capture Face"):
        if name:
            capture_face(name)
        else:
            st.warning("⚠ Please enter a name before capturing.")

elif page == "Train Model":
    st.subheader("📚 Train Face Recognition Model")
    if st.button("🚀 Train Model"):
        train_model()

elif page == "Recognize & Mark Attendance":
    st.subheader("✅ Recognize Student & Mark Attendance")
    st.warning("🕒 Attendance will be open for 20 minutes!")
    if st.button("🎥 Start Face Recognition"):
        recognize_face()

elif page == "View Attendance":
    st.subheader("📊 Subject Attendance Overview")
    
    # Read the centralized tracking file
    if os.path.exists(ATTENDANCE_TRACKING_FILE):
        df = pd.read_csv(ATTENDANCE_TRACKING_FILE)
        
        # Subjects selection
        subjects = df['Subject'].unique().tolist()
        selected_subject = st.selectbox("Select Subject", subjects)
        
        # Filter by selected subject
        subject_df = df[df['Subject'] == selected_subject]
        
        # Unique attendees per subject
        unique_attendees = subject_df.groupby(['Name', 'Date']).size().reset_index(name='Attendance Count')
        
        st.dataframe(unique_attendees)
        
        st.subheader(f"{selected_subject} Attendance")
        try:
            st.bar_chart(unique_attendees.groupby('Name')['Attendance Count'].sum())
        except Exception as e:
            st.error(f"Error creating bar chart: {e}")
    else:
        st.warning("No attendance data available.")

else:
    st.subheader("📚 Face Recognition Attendance System")
    st.write("Welcome! Use the sidebar to navigate:")
    st.write("1. Register Face: Capture student faces")
    st.write("2. Train Model: Train face recognition model")
    st.write("3. Recognize & Mark Attendance: Mark attendance")
    st.write("4. View Attendance: Check attendance records")