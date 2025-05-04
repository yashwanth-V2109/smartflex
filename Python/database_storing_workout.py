import cv2
import mediapipe as mp
import numpy as np
import mysql.connector

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Database connection
db_connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Yashwanth1234@",
    database="workout"
)

db_cursor = db_connection.cursor()

# Create a table if it doesn't exist
db_cursor.execute("""
    CREATE TABLE IF NOT EXISTS reps_data (
        id INT AUTO_INCREMENT PRIMARY KEY,
        exercise VARCHAR(255),
        count INT
    )
""")

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# video Feed
cap = cv2.VideoCapture(0)
counter_curls = 0
counter_squats = 0
stage_curls = None
stage_squats = None

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Detect stuff and render
        # RECOLOR THE IMAGE
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # MAKE DETECTION
        results = pose.process(image)

        # RECOLOR BACK TO BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract Landmark for Bicep Curls
        try:
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # GET COORDINATES for Bicep Curls
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # Calculate angle for Bicep Curls
                angle_curls = calculate_angle(shoulder, elbow, wrist)

                # Curl Counter Logic
                if angle_curls > 160 and stage_curls != 'down':
                    stage_curls = 'down'
                if angle_curls < 30 and stage_curls == 'down':
                    stage_curls = 'up'
                    counter_curls += 1
                    print("Bicep Curls:", counter_curls)

        except Exception as e:
            print(f"Error (Bicep Curls): {e}")

        # Extract Landmark for Squats
        try:
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # GET COORDINATES for Squats
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                # Calculate angle for Squats
                angle_squats = calculate_angle(hip, knee, ankle)

                # Squat Counter Logic
                if angle_squats > 160 and stage_squats != 'down':
                    stage_squats = 'down'
                if angle_squats < 90 and stage_squats == 'down':
                    stage_squats = 'up'
                    counter_squats += 1
                    print("Squats:", counter_squats)

        except Exception as e:
            print(f"Error (Squats): {e}")

        # Insert data into the database
        try:
            db_cursor.execute("INSERT INTO reps_data (exercise, count) VALUES (%s, %s)", ("Bicep Curls", counter_curls))
            db_cursor.execute("INSERT INTO reps_data (exercise, count) VALUES (%s, %s)", ("Squats", counter_squats))
            db_connection.commit()
        except Exception as e:
            print(f"Error (Database): {e}")

        # render detection
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture object
cap.release()

# Close the database connection
db_cursor.close()
db_connection.close()

# Destroy any OpenCV windows
cv2.destroyAllWindows()
