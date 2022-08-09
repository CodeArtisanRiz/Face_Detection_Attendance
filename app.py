# Imports
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from PIL import ImageGrab

# import face_recognition  # pip install face_recognition
# import cv2  # pip install opencv-python
# import numpy as np
import csv


path = 'training_data'
known_face_encodings = []
known_faces_names = []
myList = os.listdir(path)
print(f'Reading Images from training folder: {myList} ')
print(f'Trained Image List : {myList}')

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    known_face_encodings.append(curImg)
    known_faces_names.append(os.path.splitext(cl)[0])
studentList = []

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
f = open(current_date+'.csv', 'w+', newline='')
lnwriter = csv.writer(f)


def findEncodings(images):
    encodeList = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# def updateCsv(name):
encodeListKnown = findEncodings(known_face_encodings)
print('Training Complete.')
print('Press Q to Quit.')

cap = cv2.VideoCapture(0)

# Program Running State
while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = known_faces_names[matchIndex].upper()

            # print(f'{name} : Present')
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            # Update attendance in csv
            if name not  in studentList:
                print(f'{name} : Present')
                current_time = now.strftime("%H-%M-%S")
                lnwriter.writerow([name, current_time])
                studentList.append(name)
            else :
                print(f'{name}: Already Present')


    cv2.imshow('Attendance', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
