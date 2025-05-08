import cv2
import mediapipe as mp
import putsuit

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_face_detection = mp.solutions.face_detection
faces = mp_face_detection.FaceDetection()

title='Hand & Face Recognition but Absolute Cinema'

isCinemaCount=0
notCinemaCount=0
cinema=False

import pygame

pygame.mixer.init()
pygame.mixer.music.load('bgm.mp3')

def checkCinema(hands,faces):
    squares=tuple(map(lambda x:x+["hand"],hands)) + tuple(map(lambda x:x+["face"],faces))
    squares=sorted(squares,key=lambda x: x[0])
    if len(squares)==3:
        if squares[0][-1]=="hand" and squares[1][-1]=="face" and squares[2][-1]=="hand":
            fw=squares[1][2]-squares[1][0]
            fh=squares[1][3]-squares[1][1]
            if abs(squares[0][1]-squares[1][1])<fh and abs(squares[1][1]-squares[2][1])<fh:
                if abs(squares[0][2]+fw-squares[1][0])<fw and abs(squares[1][2]+fw-squares[2][0])<fw:
                    return True
    return False

cap = cv2.VideoCapture(0)
cv2.namedWindow(title, cv2.WINDOW_GUI_NORMAL)
scrw,scrh=1200,900
cv2.resizeWindow(title, scrw, scrh)

prevTick=0
while cap.isOpened():
    secondsPast=((cv2.getTickCount()-prevTick)/cv2.getTickFrequency())
    prevTick=cv2.getTickCount()
    ret, frame = cap.read()
    if not ret:
        break
    
    frame=cv2.flip(frame,1)
    frame=cv2.resize(frame,(scrw,scrh))
    originalFrame=frame.copy()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    handRes = hands.process(rgb_frame)
    faceRes = faces.process(rgb_frame)

    handRecs=[]
    faceRecs=[]

    if handRes.multi_hand_landmarks:
        handNum=0
        for hand_landmarks in handRes.multi_hand_landmarks:
            h, w, _ = frame.shape
            x_min, y_min = w, h
            x_max, y_max = 0, 0
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x, x_min)
                y_min = min(y, y_min)
                x_max = max(x, x_max)
                y_max = max(y, y_max)
            box=[x_min, y_min,x_max, y_max]
            handRecs.append(box)
            display=[x_min, y_min,x_max-x_min, y_max-y_min]
            cv2.rectangle(frame, display, (0, 255, 0), 2)
            cv2.putText(frame,f"hand{handNum}",(x_min, y_min-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 0),2)
            handNum+=1

    if faceRes.detections:
        for detection in faceRes.detections:
            bbox_c = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            bbox = [int(bbox_c.xmin * w), int(bbox_c.ymin * h), \
                   int((bbox_c.xmin + bbox_c.width) * w), int((bbox_c.ymin + bbox_c.height) * h)]
            faceRecs.append(bbox)
            display=[int(bbox_c.xmin * w), int(bbox_c.ymin * h), \
                   int((bbox_c.width) * w), int((bbox_c.height) * h)]
            cv2.rectangle(frame, display, (255, 0, 0),2)
            cv2.putText(frame,"head",(bbox[0], bbox[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 0, 0),2)

    isCinema=checkCinema(handRecs,faceRecs)

    if cinema:
        if isCinema:
            notCinemaCount=0
        else:
            notCinemaCount+=secondsPast
            if notCinemaCount>0.5:
                cinema=False
                notCinemaCount=0
                pygame.mixer.music.fadeout(800)
        putsuit.putSuit(originalFrame,rgb_frame)
        cv2.putText(originalFrame,"ABSOLUTE",(440,700),cv2.FONT_HERSHEY_SIMPLEX,2,(255, 255, 255),8)
        cv2.putText(originalFrame,"CINEMA",(250,830+40+5),cv2.FONT_HERSHEY_SIMPLEX,6,(255, 255, 255),24)
        cv2.imshow(title, cv2.cvtColor(originalFrame, cv2.COLOR_RGB2GRAY))
    else:
        if isCinema:
            isCinemaCount+=secondsPast
            if isCinemaCount>0.5:
                cinema=True
                isCinemaCount=0
                pygame.mixer.music.play()
        else:
            isCinemaCount=0
        cv2.imshow(title, frame)


    key=cv2.waitKey(1)
    if  key& 0xFF == ord('q') or key==27:
        break

cap.release()
cv2.destroyAllWindows()