import cv2
import mediapipe as mp
import numpy as np
import json
import sys
import time
import datetime
import traceback

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def detect_exercise(landmarks, exercise_type):
    """Detect exercises based on pose landmarks and exercise type."""
    # Extract coordinates
    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    
    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    
    # Calculate angles
    arm_angle = calculate_angle(shoulder, elbow, wrist)
    leg_angle = calculate_angle(hip, knee, ankle)
    
    # Detect exercises based on type
    if exercise_type == "bicep_curl":
        # Bicep curl detection
        if arm_angle > 160:
            return "down"
        elif arm_angle < 30:
            return "up"
    elif exercise_type == "squat":
        # Squat detection
        if leg_angle > 160:
            return "up"
        elif leg_angle < 90:
            return "down"
    elif exercise_type == "pushup":
        # Pushup detection
        if arm_angle > 160:
            return "up"
        elif arm_angle < 90:
            return "down"
    
    return None

def main():
    # Parse command line arguments
    if len(sys.argv) < 3:
        print(json.dumps({
            "error": "Usage: python workout_monitering.py <userId> <workoutType>"
        }))
        sys.exit(1)
    
    user_id = sys.argv[1]
    workout_type = sys.argv[2]
    
    # Initialize counters and state
    counter = 0
    stage = None
    start_time = time.time()
    
    # Initialize video capture
    print(json.dumps({"status": "initializing", "message": "Opening camera..."}))
    sys.stdout.flush()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(json.dumps({
            "error": "Failed to open camera. Please check if your camera is connected and not in use by another application."
        }))
        sys.stdout.flush()
        sys.exit(1)
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print(json.dumps({"status": "ready", "message": "Camera initialized successfully"}))
    sys.stdout.flush()
    
    # Initialize MediaPipe Pose
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_count = 0
        last_log_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print(json.dumps({"error": "Failed to read frame from camera"}))
                sys.stdout.flush()
                break
                
            frame_count += 1
            
            # Convert the BGR image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make detection
            results = pose.process(image)
            
            # Convert back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            if results.pose_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    image, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )
                
                # Detect exercise
                landmarks = results.pose_landmarks.landmark
                current_stage = detect_exercise(landmarks, workout_type)
                
                # Count reps
                if current_stage == "up" and stage != "up":
                    counter += 1
                    # Print to stdout for the Node.js server to capture
                    print(json.dumps({
                        "timestamp": datetime.datetime.now().isoformat(),
                        "userId": user_id,
                        "workoutType": workout_type,
                        "repCount": counter,
                        "stage": current_stage,
                        "duration": int(time.time() - start_time),
                        "calories": int((time.time() - start_time) * 0.1)  # Rough estimate: 0.1 calories per second
                    }))
                    sys.stdout.flush()  # Ensure the output is sent immediately
                
                stage = current_stage
                
                # Display rep count
                cv2.putText(image, f'REPS: {counter}', (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f'STAGE: {stage}', (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Log status every 5 seconds
            current_time = time.time()
            if current_time - last_log_time >= 5:
                print(json.dumps({
                    "status": "running",
                    "frameCount": frame_count,
                    "fps": frame_count / (current_time - start_time),
                    "repCount": counter
                }))
                sys.stdout.flush()
                last_log_time = current_time
            
            # Display the image
            cv2.imshow('Workout Tracking', image)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(json.dumps({"status": "stopping", "message": "User requested stop"}))
                sys.stdout.flush()
                break
    
    # Release resources
    print(json.dumps({"status": "cleaning_up", "message": "Releasing camera resources"}))
    sys.stdout.flush()
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final workout summary
    end_time = time.time()
    duration = end_time - start_time
    
    print(json.dumps({
        "timestamp": datetime.datetime.now().isoformat(),
        "userId": user_id,
        "workoutType": workout_type,
        "totalReps": counter,
        "duration": int(duration),
        "calories": int(duration * 0.1),
        "status": "completed"
    }))
    sys.stdout.flush()  # Ensure the final output is sent

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(json.dumps({
            "error": f"Unexpected error: {str(e)}",
            "traceback": traceback.format_exc()
        }))
        sys.stdout.flush()
        sys.exit(1)


    
    