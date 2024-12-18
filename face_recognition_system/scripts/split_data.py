import os

# Directories
known_faces_dir = "dataset/known_faces"
unknown_faces_dir = "dataset/unknown_faces"

# Create directories if they don't exist
os.makedirs(known_faces_dir, exist_ok=True)
os.makedirs(unknown_faces_dir, exist_ok=True)

print("Directory structure is ready.")
