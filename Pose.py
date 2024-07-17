import cv2
import mediapipe as mp
import math
import matplotlib.pyplot as plt
import os

os.system("cls")


def calcAngle3d(a,b,c):
    bone1 = [a.x-b.x, a.y-b.y, a.z-b.z]
    bone2 = [c.x-b.x, c.y-b.y, c.z-b.z]

    dot_product = bone1[0]*bone2[0]+bone1[1]*bone2[1]+bone1[2]*bone2[2]
    m1 = (bone1[0]**2 + bone1[1]**2 + bone1[2]**2) ** 0.5
    m2 = (bone2[0]**2 + bone2[1]**2 + bone2[2]**2) ** 0.5

    angle = math.acos(dot_product/m1*m2)
    angle = math.degrees(angle)
    return angle



def calcAngle2d(a,b,c):       
    bone1 = [a.x - b.x, a.y - b.y]
    bone2 = [c.x - b.x, c.y - b.y]
        
    dot_product = bone1[0] * bone2[0] + bone1[1] * bone2[1]
    m1 = (bone1[0]**2 + bone1[1]**2) ** 0.5
    m2 = (bone2[0]**2 + bone2[1]**2) ** 0.5

    angle = math.acos(dot_product/m1*m2)
    angle = math.degrees(angle)
    return angle
    



# Init
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
cap = cv2.VideoCapture('pose.mp4')


# Frame rate stuff
frame_rate = cap.get(cv2.CAP_PROP_FPS)
print(f"Frame rate: {frame_rate} FPS")
knee_angles_2d = []  
knee_angles_3d = []  
timestamps = [] 

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

        angle = calcAngle2d(hip,knee,ankle)
        knee_angles_2d.append(angle)

        angle = calcAngle3d(hip,knee,ankle)
        knee_angles_3d.append(angle)

        timestamps.append(frame_count / frame_rate)

    frame_count += 1

cap.release()


# Create a figure and a single axis
fig, ax = plt.subplots()

ax.plot(timestamps, knee_angles_2d, 'b-', label='2D Knee Angles')
ax.plot(timestamps, knee_angles_3d, 'r-', label='3D Knee Angles')

ax.legend()
plt.show()






