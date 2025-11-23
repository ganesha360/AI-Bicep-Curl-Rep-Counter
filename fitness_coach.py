import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

# 1. Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

# 2. Setup Video Capture
cap = cv2.VideoCapture(0)

# --- LEFT ARM VARIABLES ---
counter_l = 0 
stage_l = None
angle_history_l = deque(maxlen=15) 
start_time_l = 0
rep_duration_l = 0
l_rep_speeds = [] # List to store all speeds

# --- RIGHT ARM VARIABLES ---
counter_r = 0 
stage_r = None
angle_history_r = deque(maxlen=15) 
start_time_r = 0
rep_duration_r = 0
r_rep_speeds = [] # List to store all speeds

# Visibility Feedback
status_msg = "Ready"

print("Starting AI Fitness Coach... Press 'q' to quit and view results.")

with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose: 
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            continue

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # =============================
            #       LEFT ARM LOGIC
            # =============================
            l_shoulder_vis = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility
            l_elbow_vis = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility
            l_wrist_vis = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].visibility

            if l_shoulder_vis > 0.7 and l_elbow_vis > 0.7 and l_wrist_vis > 0.7:
                
                l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # Calculate & Smooth Angle
                angle_l = calculate_angle(l_shoulder, l_elbow, l_wrist)
                angle_history_l.append(angle_l)
                avg_angle_l = np.mean(angle_history_l)
                
                # Visual Indicator
                cv2.putText(image, str(int(avg_angle_l)), 
                            tuple(np.multiply(l_elbow, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
                # State Machine
                if avg_angle_l > 160:
                    stage_l = "down"
                    start_time_l = time.time()
                    
                # CHANGE: Increased threshold from 50 to 80 degrees to catch reps easier
                if avg_angle_l < 80 and stage_l == 'down': 
                    stage_l = "up"
                    counter_l += 1
                    rep_duration_l = time.time() - start_time_l
                    l_rep_speeds.append(rep_duration_l) # Store speed
                    print(f"LEFT ARM  | Rep #{counter_l} | Tempo: {rep_duration_l:.2f}s")
            
            # =============================
            #       RIGHT ARM LOGIC
            # =============================
            r_shoulder_vis = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility
            r_elbow_vis = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility
            r_wrist_vis = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].visibility

            if r_shoulder_vis > 0.7 and r_elbow_vis > 0.7 and r_wrist_vis > 0.7:
                
                r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                # Calculate & Smooth Angle
                angle_r = calculate_angle(r_shoulder, r_elbow, r_wrist)
                angle_history_r.append(angle_r)
                avg_angle_r = np.mean(angle_history_r)
                
                # Visual Indicator
                cv2.putText(image, str(int(avg_angle_r)), 
                            tuple(np.multiply(r_elbow, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
                # State Machine
                if avg_angle_r > 160:
                    stage_r = "down"
                    start_time_r = time.time()
                
                # CHANGE: Increased threshold from 50 to 80 degrees to catch reps easier
                if avg_angle_r < 80 and stage_r == 'down':
                    stage_r = "up"
                    counter_r += 1
                    rep_duration_r = time.time() - start_time_r
                    r_rep_speeds.append(rep_duration_r) # Store speed
                    print(f"RIGHT ARM | Rep #{counter_r} | Tempo: {rep_duration_r:.2f}s")

            # Global Visibility Status
            if (l_elbow_vis < 0.5 and r_elbow_vis < 0.5):
                status_msg = "Please step back"
            else:
                status_msg = "Tracking..."
                        
        except:
            pass
        
        # --- DRAW UI ---
        cv2.rectangle(image, (0,0), (640, 85), (245,117,16), -1)
        
        # Left Side UI
        cv2.putText(image, 'LEFT', (10,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, f"Reps: {counter_l}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(image, f"Sec: {rep_duration_l:.1f}s", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        
        # Right Side UI
        cv2.putText(image, 'RIGHT', (450,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, f"Reps: {counter_r}", (450,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(image, f"Sec: {rep_duration_r:.1f}s", (450,70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        
        # Center Status
        cv2.line(image, (320,0), (320,85), (255,255,255), 2)
        cv2.putText(image, status_msg, (240, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        
        # Render Skeleton
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('AI Fitness Coach', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # --- FINAL TERMINAL REPORT ---
    # Calculate averages avoiding division by zero
    avg_speed_l = sum(l_rep_speeds)/len(l_rep_speeds) if len(l_rep_speeds) > 0 else 0.0
    avg_speed_r = sum(r_rep_speeds)/len(r_rep_speeds) if len(r_rep_speeds) > 0 else 0.0
    
    print("\n" + "="*45)
    print("     WORKOUT SESSION SUMMARY     ")
    print("="*45)
    print(f"LEFT ARM:")
    print(f"  Total Reps: {counter_l}")
    print(f"  Avg Speed : {avg_speed_l:.2f} seconds/rep")
    print("-" * 25)
    print(f"RIGHT ARM:")
    print(f"  Total Reps: {counter_r}")
    print(f"  Avg Speed : {avg_speed_r:.2f} seconds/rep")
    print("-" * 25)
    print(f"Total Aggregate Reps: {counter_l + counter_r}")
    print("="*45 + "\n")