import face_recognition  # pip install face_recognition
import cv2  # pip install opencv-python
import numpy as np
import csv
import os
from datetime import datetime

# Initialize camera window
video_capture = cv2.VideoCapture(0)

# Image training from storage
jobs_image = face_recognition.load_image_file("training_data/jobs.jpg")
jobs_encoding = face_recognition.face_encodings(jobs_image)[0]

ratan_tata_image = face_recognition.load_image_file("training_data/tata.jpg")
ratan_tata_encoding = face_recognition.face_encodings(ratan_tata_image)[0]

sadmona_image = face_recognition.load_image_file("training_data/sadmona.jpg")
sadmona_encoding = face_recognition.face_encodings(sadmona_image)[0]

tesla_image = face_recognition.load_image_file("training_data/tesla.jpg")
tesla_encoding = face_recognition.face_encodings(tesla_image)[0]


# List of trained image encodings
known_face_encodings = [
    jobs_encoding,
    ratan_tata_encoding,
    sadmona_encoding,
    tesla_encoding
]

# List of trained image names
known_faces_names = [
    "Steve Jobs",
    "Ratan Tata",
    "Sadmona",
    "Tesla"
]

# Copy of known face names
students = known_faces_names.copy()

# Initailizing empty lists to store live data
face_locations = []
face_encodings = []
face_names = []
program_running = True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date+'.csv', 'w+', newline='')
lnwriter = csv.writer(f)

while True:
    # Read Image frames continuously
    _, frame = video_capture.read()
    # Resize image frames
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # rgb
    rgb_small_frame = small_frame[:, :, ::-1]
    if program_running:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding)
            name = ""
            face_distance = face_recognition.face_distance(
                known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]

            face_names.append(name)
            if name in known_faces_names:
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name, current_time])

    cv2.imshow("attendence system", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()
