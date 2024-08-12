import cv2
import mediapipe as mp
import math
import matplotlib.pyplot as plt
import os
import extract_data as data

os.system("cls")

# Create a figure and a single axis
fig, ax = plt.subplots()

ax.plot(data.oc_time, data.oc_knee_angle, 'r-', label='OpenCap   ')

#ax.plot(data.mc_time, data.mc_knee_angle_2d, 'g-', label='MoCap (2D)')
ax.plot(data.mc_time, data.mc_knee_angle_3d, 'g-', label='MoCap (3D)')


ax.plot(data.mp_time, data.mp_knee_angles_2d, 'b-', label='MediaPipe (2D)')
#ax.plot(data.mp_time, data.mp_knee_angles_3d, 'b-', label='MediaPipe (3D)')

ax.legend()
plt.show()
