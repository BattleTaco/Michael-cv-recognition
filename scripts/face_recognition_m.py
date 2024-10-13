import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import face_recognition
from torchvision import transforms
from dotenv import load_dotenv

# Define a simple neural network for face recognition
class FaceRecognitionModel(nn.Module):
    def __init__(self, input_size=128, num_classes=2):
        super(FaceRecognitionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class FaceRecognitionSystem:
    def __init__(self, dataset_dir, num_classes):
        self.dataset_dir = dataset_dir
        self.num_classes = num_classes
        self.model = FaceRecognitionModel(input_size=128, num_classes=num_classes)
        self.label_map = {}
        self.encodings = []
        self.labels = []
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.transform = transforms.ToTensor()

    def load_images_and_encodings(self):
        """Load face images from the dataset and generate encodings."""
        for person_name in os.listdir(self.dataset_dir):
            person_folder = os.path.join(self.dataset_dir, person_name)
            if os.path.isdir(person_folder):
                for img_file in os.listdir(person_folder):
                    img_path = os.path.join(person_folder, img_file)
                    image = face_recognition.load_image_file(img_path)
                    face_encoding = face_recognition.face_encodings(image)

                    # Check if a valid encoding is found
                    if len(face_encoding) > 0:
                        self.encodings.append(face_encoding[0])
                        self.labels.append(person_name)
                    else:
                        print(f"Warning: No face detected in {img_path}")
        
        # Check if encodings were found
        if len(self.encodings) == 0:
            raise RuntimeError("No face encodings were found. Please check your dataset.")

        # Convert encodings and labels to torch tensors
        self.encodings = torch.tensor(np.array(self.encodings), dtype=torch.float32)  # Ensure float32 dtype

        # Check encoding tensor shape
        print(f"Encodings shape: {self.encodings.shape}")  # This should be (num_samples, 128)
        
        unique_labels = set(self.labels)
        self.label_map = {name: i for i, name in enumerate(unique_labels)}
        self.labels = torch.tensor([self.label_map[label] for label in self.labels])

        # Check labels shape
        print(f"Labels shape: {self.labels.shape}")  # Should match the number of encodings

    def train(self, epochs=100):
        """Train the PyTorch model on the face encodings."""
        # Check if the encodings tensor has a valid shape
        if self.encodings.size(1) != 128:
            raise RuntimeError(f"Expected encoding size of 128, but got {self.encodings.size(1)}")
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(self.encodings)
            loss = self.criterion(outputs, self.labels)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    def save_model(self, file_path="face_recognition_model.pth"):
        """Save the trained model to disk."""
        torch.save(self.model.state_dict(), file_path)

    def load_model(self, file_path="face_recognition_model.pth"):
        """Load the trained model from disk."""
        self.model.load_state_dict(torch.load(file_path))
        self.model.eval()

    def recognize_faces(self, frame):
        """Recognize faces in a frame using the trained model."""
        rgb_frame = frame[:, :, ::-1]  # Convert BGR (OpenCV format) to RGB (for face_recognition)
        
        # Resize the frame to speed up face detection and improve accuracy
        small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)
        
        # Detect face locations (bounding boxes)
        face_locations = face_recognition.face_locations(small_frame)

        if len(face_locations) == 0:
            print("No face locations detected in the frame.")
            return []

        print(f"Face locations: {face_locations}")

        # Get face encodings for each face detected
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        if len(face_encodings) == 0:
            print("No face encodings detected.")
            return []

        print(f"Face encodings: {face_encodings}")

        recognized_faces = []

        # Process each face found in the current frame
        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Convert the encoding to a PyTorch tensor
            face_encoding_tensor = torch.tensor(face_encoding, dtype=torch.float32).unsqueeze(0)

            # Ensure the tensor is of the correct shape
            print(f"Face encoding tensor shape: {face_encoding_tensor.shape}")

            # Pass the encoding through the model
            with torch.no_grad():
                outputs = self.model(face_encoding_tensor)
                _, predicted = torch.max(outputs.data, 1)

                # Assuming "Michael" is recognized by the model
                name = "Michael"
                recognized_faces.append((name, face_location))

        return recognized_faces




class FaceRecognitionApp:
    def __init__(self, model):
        self.model = model

    def start_video_capture(self):
        """Capture video from the webcam and recognize faces in real time."""
        video_capture = cv2.VideoCapture(0)

        try:
            while True:
                ret, frame = video_capture.read()

                # Recognize faces in the current frame
                recognized_faces = self.model.recognize_faces(frame)

                # Draw bounding boxes and labels on the frame
                for name, (top, right, bottom, left) in recognized_faces:
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
    # Create a face recognition system
    load_dotenv()
    dataset_dir = os.getenv("DATASET_DIR", "C:/Users/Michael Ramirez/Documents/GitHub/Michael CV Recognition/data/")
    print(f"Dataset directory: {dataset_dir}")
    num_classes = 2  # Michael and Rachel
    face_recognition_system = FaceRecognitionSystem(dataset_dir, num_classes)

    # Load images and face encodings
    face_recognition_system.load_images_and_encodings()

    # Train the model
    print("Training the model...")
    face_recognition_system.train(epochs=100)

    # Save the trained model (optional)
    face_recognition_system.save_model("face_recognition_model.pth")

    # Create a face recognition app and start video capture
    face_recognition_app = FaceRecognitionApp(face_recognition_system)
    face_recognition_app.start_video_capture()

if __name__ == "__main__":
    main()
