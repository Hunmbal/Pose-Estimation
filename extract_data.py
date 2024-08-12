import csv
import cv2
import mediapipe as mp
import math
import matplotlib.pyplot as plt
import os

os.system("cls")

#_______________________________________________________________________________________ Mathematics
def calcAngle3d(a, b, c):
    bone1 = [a.x - b.x, a.y - b.y, a.z - b.z]
    bone2 = [c.x - b.x, c.y - b.y, c.z - b.z]

    dot_product = (bone1[0] * bone2[0]) + (bone1[1] * bone2[1]) + (bone1[2] * bone2[2])
    m1 = math.sqrt((bone1[0] ** 2) + (bone1[1] ** 2) + (bone1[2] ** 2))
    m2 = math.sqrt((bone2[0] ** 2) + (bone2[1] ** 2) + (bone2[2] ** 2))

    angle = math.acos(dot_product / (m1 * m2))
    angle = math.degrees(angle)
    return angle

def calcAngle2d(a, b, c):
    bone1 = [a.x - b.x, a.y - b.y]
    bone2 = [c.x - b.x, c.y - b.y]

    dot_product = (bone1[0] * bone2[0]) + (bone1[1] * bone2[1])
    m1 = math.sqrt((bone1[0] ** 2) + (bone1[1] ** 2))
    m2 = math.sqrt((bone2[0] ** 2) + (bone2[1] ** 2))

    angle = math.acos(dot_product / (m1 * m2))
    angle = math.degrees(angle)
    return angle

def tcalcAngle3d(a, b, c):
    bone1 = [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    bone2 = [c[0] - b[0], c[1] - b[1], c[2] - b[2]]

    dot_product = (bone1[0] * bone2[0]) + (bone1[1] * bone2[1]) + (bone1[2] * bone2[2])
    m1 = math.sqrt((bone1[0] ** 2) + (bone1[1] ** 2) + (bone1[2] ** 2))
    m2 = math.sqrt((bone2[0] ** 2) + (bone2[1] ** 2) + (bone2[2] ** 2))

    angle = math.acos(dot_product / (m1 * m2))
    angle = math.degrees(angle)
    return angle

def tcalcAngle2d(a, b, c):
    bone1 = [a[0] - b[0], a[1] - b[1]]
    bone2 = [c[0] - b[0], c[1] - b[1]]

    dot_product = (bone1[0] * bone2[0]) + (bone1[1] * bone2[1])
    m1 = math.sqrt((bone1[0] ** 2) + (bone1[1] ** 2))
    m2 = math.sqrt((bone2[0] ** 2) + (bone2[1] ** 2))

    angle = math.acos(dot_product / (m1 * m2))
    angle = math.degrees(angle)
    return angle


#_______________________________________________________________________________________ DAY#1
mc_time = []
mc_knee_angle_2d = []
mc_knee_angle_3d = []

with open('mc.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    headers = next(reader)  # Skip the first header row
    headers = next(reader)  # Skip the second header row
    
    for row in reader:
        time = float(row[0])
        # Extract coordinates
        L_toes = (float(row[1]), float(row[2]), float(row[3]))
        L_foot = (float(row[4]), float(row[5]), float(row[6]))
        L_knee = (float(row[7]), float(row[8]), float(row[9]))
        L_hip = (float(row[10]), float(row[11]), float(row[12]))

        # Calculate angles
        angle_2d = tcalcAngle2d(L_hip[:2], L_knee[:2], L_foot[:2])
        angle_3d = tcalcAngle3d(L_hip, L_knee, L_foot)

        # Append data to arrays
        mc_time.append(time)
        mc_knee_angle_2d.append(angle_2d)
        mc_knee_angle_3d.append(angle_3d)


#_______________________________________________________________________________________ DAY#2
oc_time = []
oc_knee_angle = []

with open('oc.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row if present
    for row in reader:
        oc_time.append(float(row[0]))
        oc_knee_angle.append(float(row[1]))



#_______________________________________________________________________________________ DAY#3
mp_time = []
mp_knee_angles_2d = []
mp_knee_angles_3d = []

# Init
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
cap = cv2.VideoCapture('mp.mp4')

# Frame rate stuff
frame_count = 0
frame_rate = cap.get(cv2.CAP_PROP_FPS)
print(f"Frame rate: {frame_rate} FPS")

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

        angle_2d = calcAngle2d(hip, knee, ankle)
        mp_knee_angles_2d.append(angle_2d)

        angle_3d = calcAngle3d(hip, knee, ankle)
        mp_knee_angles_3d.append(angle_3d)

        mp_time.append(frame_count / frame_rate)

    frame_count += 1

cap.release()