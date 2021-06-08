import cv2
import numpy as np
import face_recognition
import os

def findencodings(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

path = 'face_recognition'
images = []
classNames = []
mylist = os.listdir(path)

for cls in mylist:
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)

encodeListKnown = findencodings(images)
print("Encoding complete")

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

while True:
    success,img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
    for encodeFace,faceloc in zip(encodesCurFrame,facesCurFrame):
        matches =face_recognition.compare_faces(encodeListKnown,encodeFace)
        facDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        print(facDis)
        matchIndex = np.argmin(facDis)
        if matches[matchIndex]:
            name = classNames[matchIndex]
            print(name)
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(img,name,(x1,y2),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
    cv2.imshow('webcam',img)
    cv2.waitKey(1)
