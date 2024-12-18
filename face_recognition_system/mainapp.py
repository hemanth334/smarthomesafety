import cv2
import face_recognition
import pickle
import os
from datetime import datetime
from twilio.rest import Client

# Paths
model_path = "models/face_encodings.pkl"
unknown_faces_dir = "dataset/unknown_faces"

# Twilio Configuration
account_sid = "ACb8843328af6f98bf5ceddc66ef21fc9e"  # Replace with your Twilio account SID
auth_token = "fb5a5d5153dd9bc2777085e300473148"    # Replace with your Twilio auth token
from_number = "+18653445132"       # Replace with your Twilio phone number
to_number = "+919535170347"         # Replace with the recipient's phone number

client = Client(account_sid, auth_token)

# Load known face encodings
with open(model_path, "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

# Ensure unknown_faces_dir exists
os.makedirs(unknown_faces_dir, exist_ok=True)

# Cooldown configuration
cooldown_seconds = 600  # Adjust as needed
last_recognized = {}  # Store the last time a face was recognized

# Initialize webcam
cap = cv2.VideoCapture(0)

print("Starting real-time face detection. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]
        else:
            # Save the unknown face
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_name = f"unknown_{timestamp}.jpg"
            img_path = os.path.join(unknown_faces_dir, img_name)
            cv2.imwrite(img_path, frame)

            # Send alert
            message = client.messages.create(
                body=f"Unknown face detected at {timestamp}",
                from_=from_number,
                to=to_number
            )
            print(f"Alert sent: {message.sid}")

        # Draw rectangle around face
        top, right, bottom, left = face_location
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame
    cv2.imshow("Face Recognition", frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
