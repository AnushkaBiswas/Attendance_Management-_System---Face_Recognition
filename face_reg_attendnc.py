import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from deepface import DeepFace

path = 'images'   #directory name
images = []   #list of all images
prsn_name = []
myList = os.listdir(path)  #listing of current directories - list of images as path contains images
print(myList)

#splitting out the images
for cu_img in myList:
    current_Img = cv2.imread(f'{path}/{cu_img}')   #all iamges gets read and stored in currentimg
    images.append(current_Img)
    prsn_name.append(os.path.splitext(cu_img)[0])   #[0] only takes name and removes jpeg[1]
print(prsn_name)

#encodings - finds 128 unique features of a face which helps in diffirenciating faces => HOG ALGORITHM used to find encodings
def faceEncodings(images) :
    encodeList = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #bgr format to rgb formt convrsn. in cv2 it is bgr format so need convrsn
        encode = face_recognition.face_encodings(img)[0]  #all the 128 encoding values of face stored in encode
        encodeList.append(encode)
    return encodeList

encodeListKnown = faceEncodings(images)  #all the list array of 128 encoding values of face stored
print("All Encodings complete !!")


#marking the attendance
def attendance(name):
    with open('attendance.csv', 'r+') as f:  #mode = r+ = read and append mode
        myDataList = f.readlines()
        nameList = []

        for line in myDataList:  #for splitting name, time, date
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S')
            dStr = time_now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{tStr},{dStr}')


#camera reading
cam = cv2.VideoCapture(0)  # 0 -cam id for laptop by default

while True :
    ret, frame = cam.read()   #camera frame reading
    faces = cv2.resize(frame, (0,0), None, 0.25, 0.25)  #resizing camera  #destinatn=None , how much smaller = 0.25
    faces= cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)  #as cv2 used, cam gives in bgr format so need to convert from bgr to rgb

    facesCurrrentFrame = face_recognition.face_locations(faces)  # from camera we need to find face colatn and face encodings
    encodesCurrentFrame = face_recognition.face_encodings(faces, facesCurrrentFrame)


    # using this above 2, we have to find does the faces match , and find the facedistance 
    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrrentFrame):   #zip = can pass 2 packages in single functn

        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)  #for comparing faces
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)  #for finding face distance 
        #if face distance more= faces not matching. if face dist less teh faces are matchng

        # print(faceDis)
        matchIndex = np.argmin(faceDis)  #finding the minimum face distnc (index value) from face list

        if matches[matchIndex]:
            name = prsn_name[matchIndex].upper()   #if face matches gives the persons name
            #print(name)
            y1, x2, y2, x1 = faceLoc   #to make rectangle over the detected face
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  #as we had resized the cam by 0.25(1/4) so npw *1/4 -= oeginal shape
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  #rectngl surroundg detectd face
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2) #name under the rectng surrndng image
            attendance(name)

            #emotions
            try:
                analyze= DeepFace.analyze(frame,actions=['emotion'])
                cv2.putText(frame,str(analyze['dominant_emotion']),(200,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),3)
            except:
                pass

    cv2.imshow("Camera", frame)  #to check out camera frame
    if cv2.waitKey(1) == 13:  #checks per milisec that ascii value is 13(enter key code = for closng cam)
        break

    
cam.release()
cv2.destroyAllWindows()