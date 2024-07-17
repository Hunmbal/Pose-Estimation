import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize MediaPipe Drawing Utilities
mp_drawing = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)  # 0 for default camera

# Set resolution to 720p
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe Pose
    results = pose.process(rgb_frame)
    

    
    # Check if pose landmarks are detected
    if results.pose_landmarks:
        # Draw landmarks on the frame (optional, can be removed if not needed)
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Get left wrist joint coordinates
        landmarks = results.pose_landmarks.landmark
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        
        # Calculate coordinates in pixel space
        h, w, _ = frame.shape
        left_wrist_x = int(left_wrist.x * w)
        left_wrist_y = int(left_wrist.y * h)
        
        # Draw a dot on the left wrist
        cv2.circle(frame, (left_wrist_x, left_wrist_y), 5, (0, 255, 0), -1)  # Green dot with radius 5
    
        left_wrist_coords = (left_wrist.x, left_wrist.y, left_wrist.z)

        # Mirror the video frame horizontally
        frame = cv2.flip(frame, 1)
        
        # Show text for left wrist
        cv2.putText(frame, f"Left Wrist: ({left_wrist_coords[0]:.2f}, {left_wrist_coords[1]:.2f}, {left_wrist_coords[2]:.2f})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        


    
    # Display the mirrored frame
    cv2.imshow('Pose Detection', frame)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
