import cv2
import os

# Create dataset folder if not exists
os.makedirs("dataset", exist_ok=True)

def capture_face(name):
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("❌ Error: Cannot open camera")
        return
    
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    count = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            print("❌ Error: Failed to capture frame")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            count += 1
            face = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (200, 200))  # Resize the face
            cv2.imwrite(f"dataset/{name}_{count}.jpg", face_resized)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        cv2.imshow("Face Capture", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 10:  
            break
    
    cam.release()
    cv2.waitKey(1)  # Ensure all windows close properly
    cv2.destroyAllWindows()
    print(f"✅ {count} images captured and saved.")

# Run the function when this script is executed
if __name__ == "__main__":
    name = input("Enter your name: ")
    capture_face(name)
