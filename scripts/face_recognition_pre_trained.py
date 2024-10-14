import os
import cv2
import face_recognition
import numpy as np
from dotenv import load_dotenv

class FaceRecognitionSystem:
    def __init__(self, dataset_dir, threshold=0.9):
        self.dataset_dir = dataset_dir
        self.threshold = threshold
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()

    def load_known_faces(self):
        """Load known faces from the dataset and generate face encodings."""
        for person_name in os.listdir(self.dataset_dir):
            person_folder = os.path.join(self.dataset_dir, person_name)
            if os.path.isdir(person_folder):
                for img_file in os.listdir(person_folder):
                    img_path = os.path.join(person_folder, img_file)
                    image = face_recognition.load_image_file(img_path)
                    face_encodings = face_recognition.face_encodings(image)

                    # Add all face encodings for this person
                    for face_encoding in face_encodings:
                        self.known_face_encodings.append(face_encoding)
                        self.known_face_names.append(person_name)

        print(f"Loaded {len(self.known_face_encodings)} known faces.")

    def recognize_faces(self, frame):
        """Recognize faces in a frame using the pre-trained model."""
        rgb_frame = frame[:, :, ::-1]  # Convert BGR (OpenCV format) to RGB (for face_recognition)
        
        # Resize the frame to speed up face detection and improve accuracy
        small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)
        
        # Detect face locations (bounding boxes)
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        recognized_faces = []

        # Process each face found in the current frame
        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Calculate distances between the detected face encoding and known encodings
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

            # Find the best match index
            best_match_index = np.argmin(face_distances)
            best_distance = face_distances[best_match_index]

            if best_distance < self.threshold:
                name = self.known_face_names[best_match_index]
            else:
                name = "Unknown"

            recognized_faces.append((name, face_location))

        return recognized_faces

class FaceRecognitionApp:
    def __init__(self, face_recognition_system):
        self.face_recognition_system = face_recognition_system

    def start_video_capture(self):
        """Capture video from the webcam and recognize faces in real time."""
        video_capture = cv2.VideoCapture(0)

        try:
            while True:
                ret, frame = video_capture.read()

                # Recognize faces in the current frame
                recognized_faces = self.face_recognition_system.recognize_faces(frame)

                # Draw bounding boxes and labels on the frame
                for name, (top, right, bottom, left) in recognized_faces:
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

                # Display the resulting image
                cv2.imshow('Video', frame)

                # Exit if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            # Release video capture and close window
            video_capture.release()
            cv2.destroyAllWindows()

def main():
    # Load environment variables (if any)
    load_dotenv()

    # Create a face recognition system using the pre-trained model
    dataset_dir = os.getenv("DATASET_DIR", "C:/Users/Michael Ramirez/Documents/GitHub/Michael CV Recognition/data/")
    face_recognition_system = FaceRecognitionSystem(dataset_dir, threshold=0.45)  # Adjust threshold here

    # Start video capture for real-time face recognition
    face_recognition_app = FaceRecognitionApp(face_recognition_system)
    face_recognition_app.start_video_capture()

if __name__ == "__main__":
    main()
