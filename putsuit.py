import mediapipe as mp
import cv2

mp_pose = mp.solutions.pose
pose=mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

suitImg=cv2.imread("suit.png",cv2.IMREAD_UNCHANGED)
scaleFacor=1.28
extra=0.28

def putSuit(frame,rgb_frame):
    results = pose.process(rgb_frame).pose_landmarks.landmark
    #mp_drawing.draw_landmarks(frame,results.pose_landmarks,mp_pose.POSE_CONNECTIONS)
    leftShoulder=results[12]
    rightShoulder=results[11]
    length=((leftShoulder.x-rightShoulder.x)**2+(leftShoulder.y-rightShoulder.y)**2)**0.5*1200
    width=length*scaleFacor
    height=width/suitImg.shape[0]*suitImg.shape[1]
    scaledImg=cv2.resize(suitImg,(int(width),int(height)))

    x=int(leftShoulder.x*1200-length*extra/2)
    y=int(leftShoulder.y*900-length*extra/1.3)

    if y >= frame.shape[0] or x >= frame.shape[1]:return

    h=scaledImg.shape[0]
    if y + h > frame.shape[0]:
        h = frame.shape[0] - y
    w=scaledImg.shape[1]
    if x + w > frame.shape[1]:
        w = frame.shape[1] - x

    alpha = scaledImg[:h, :w, 3] / 255.0
    for c in range(0, 3): 
        frame[y:y+h, x:x+w, c] = (alpha * scaledImg[:h, :w, c] + (1 - alpha) * frame[y:y+h, x:x+w, c])



