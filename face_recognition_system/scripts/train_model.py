import os
import face_recognition
import pickle

# Paths
known_faces_dir = "dataset/known_faces"  # Directory with known faces
model_path = "models/face_encodings.pkl"  # Path to save/load encodings

# List to store encodings and names
known_encodings = []
known_names = []

# Load known face encodings from the "known_faces" directory
for file_name in os.listdir(known_faces_dir):
    file_path = os.path.join(known_faces_dir, file_name)
    
    # Only process image files
    if file_path.endswith(('.jpg', '.jpeg', '.png')):
        # Load the image
        image = face_recognition.load_image_file(file_path)
        
        # Get face encodings
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        # Check if at least one face was found
        if len(face_encodings) > 0:
            # Add the first face encoding (assuming only one face per image)
            known_encodings.append(face_encodings[0])
            known_names.append(os.path.splitext(file_name)[0])  # Use the filename (without extension) as the name

# Save encodings to a pickle file
with open(model_path, "wb") as f:
    pickle.dump({"encodings": known_encodings, "names": known_names}, f)

print(f"Saved {len(known_encodings)} known faces encodings to {model_path}")
