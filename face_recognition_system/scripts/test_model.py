import os
import face_recognition
import pickle
import cv2

# Paths
model_path = "models/face_encodings.pkl"  # Path to the trained encodings
test_images_dir = "dataset/test_images"   # Directory containing test images
output_dir = "dataset/test_results"       # Directory to save annotated results

# Load the trained encodings
with open(model_path, "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process all images in the test_images_dir
for file_name in os.listdir(test_images_dir):
    file_path = os.path.join(test_images_dir, file_name)
    
    # Only process image files
    if file_path.endswith(('.jpg', '.jpeg', '.png')):
        print(f"Processing image: {file_name}")
        
        # Load the image
        image = cv2.imread(file_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces in the image
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        # Annotate the image with face recognition results
        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
            name = "Unknown"
            
            if True in matches:
                match_index = matches.index(True)
                name = known_names[match_index]

            # Draw a rectangle around the face
            top, right, bottom, left = face_location
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(image, (left, top), (right, bottom), color, 2)
            
            # Add the name below the face
            cv2.putText(image, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Save the annotated image to the output directory
        output_path = os.path.join(output_dir, file_name)
        cv2.imwrite(output_path, image)

        print(f"Results saved to: {output_path}")

print("Testing completed.")
