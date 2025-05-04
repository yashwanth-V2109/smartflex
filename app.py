# IMPORTANT: Place this code at the very top of your script
# before importing any other libraries

# 1. Set environment variables before any imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 3 = ERROR only (most aggressive)
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
os.environ['AUTOGRAPH_VERBOSITY'] = '0'  # Reduce TensorFlow Autograph logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimization messages
os.environ['PYTHONWARNINGS'] = 'ignore'  # Additional warning suppression
os.environ['TF_CPP_MAX_VLOG_LEVEL'] = '0'  # Suppress TensorFlow verbose logging including XNNPACK messages

# 2. Configure Python warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# 3. Configure standard logging
import logging
logging.basicConfig(level=logging.ERROR)
for logger_name in ['tensorflow', 'mediapipe', 'absl', 'matplotlib', 'google']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# 4. Now you can import the rest of your libraries
from flask import Flask, request, jsonify, render_template
import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from flask_cors import CORS

# Initialize Flask application
app = Flask(__name__, static_folder='public', template_folder='public')
CORS(app)

# Load bone density dataset
try:
    print("Loading bone density dataset...")
    df = pd.read_csv("Bone Mineral Density.txt", sep='\t')
    print(f"Dataset loaded successfully with {len(df)} records")
    df_male = df[df.gender == 'male']
    df_female = df[df.gender == 'female']
    min_bmd = df.spnbmd.min()
    max_bmd = df.spnbmd.max()
    print(f"Male records: {len(df_male)}, Female records: {len(df_female)}")
    print(f"BMD range: {min_bmd} to {max_bmd}")
except Exception as e:
    print(f"Error loading bone density dataset: {str(e)}")
    df = None
    df_male = None
    df_female = None
    min_bmd = 0
    max_bmd = 1

class ParametricBMDModel:
    def __init__(self, peak_age=30):
        self.peak_age = peak_age

    def fit(self, X, y):
        X = X.flatten()
        self.X_pre = X[X <= self.peak_age].reshape(-1, 1)
        self.y_pre = y[X <= self.peak_age]
        self.X_post = X[X > self.peak_age].reshape(-1, 1)
        self.y_post = y[X > self.peak_age]

        if len(self.X_pre) > 0:
            self.poly_pre = PolynomialFeatures(degree=2)
            X_pre_poly = self.poly_pre.fit_transform(self.X_pre)
            self.model_pre = LinearRegression().fit(X_pre_poly, self.y_pre)
        else:
            self.model_pre = None

        if len(self.X_post) > 0:
            self.poly_post = PolynomialFeatures(degree=1)
            X_post_poly = self.poly_post.fit_transform(self.X_post)
            self.model_post = LinearRegression().fit(X_post_poly, self.y_post)
        else:
            self.model_post = None

    def predict(self, X):
        X = np.array(X).reshape(-1, 1)
        preds = []
        for x in X:
            if x <= self.peak_age and self.model_pre:
                x_poly = self.poly_pre.transform([[x[0]]])
                preds.append(self.model_pre.predict(x_poly)[0])
            elif x > self.peak_age and self.model_post:
                x_poly = self.poly_post.transform([[x[0]]])
                preds.append(self.model_post.predict(x_poly)[0])
            elif self.model_pre:
                x_poly = self.poly_pre.transform([[x[0]]])
                preds.append(self.model_pre.predict(x_poly)[0])
            elif self.model_post:
                x_poly = self.poly_post.transform([[x[0]]])
                preds.append(self.model_post.predict(x_poly)[0])
            else:
                preds.append(np.nan)
        print(preds)
        return np.array(preds)

# Initialize bone density models if dataset is available
if df is not None:
    model_male = ParametricBMDModel()
    model_male.fit(df_male[['age']].values, df_male['spnbmd'].values)

    model_female = ParametricBMDModel()
    model_female.fit(df_female[['age']].values, df_female['spnbmd'].values)
else:
    model_male = None
    model_female = None


# Initialize MediaPipe Pose with specific configuration to avoid warnings
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Global variables to store workout data
workout_data = {
    "repCount": 0,
    "duration": 0,
    "calories": 0,
    "stage": None
}

# Flag to control the workout monitoring thread
is_workout_running = False
workout_thread = None
start_time = 0

def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

def detect_exercise(landmarks, workout_type):
    """Detect exercise based on pose landmarks and workout type."""
    if workout_type == "bicep_curl":
        # Get coordinates for bicep curl
        try:
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            # Calculate angle
            angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            
            # Count rep when angle is less than 30 degrees (arm is curled)
            return angle < 30
        except:
            return False
    
    elif workout_type == "squat":
        # Get coordinates for squat
        try:
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            
            # Calculate angle
            angle = calculate_angle(left_hip, left_knee, left_ankle)
            
            # Count rep when angle is less than 100 degrees (squat position)
            return angle < 100
        except:
            return False
    
    elif workout_type == "pushup":
        # Get coordinates for pushup
        try:
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            # Calculate angle
            angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            
            # Count rep when angle is less than 90 degrees (pushup position)
            return angle < 90
        except:
            return False
    
    return False

def workout_monitoring(workout_type):
    """Monitor workout using OpenCV and MediaPipe."""
    global workout_data, is_workout_running, start_time
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    # Ensure camera is opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        is_workout_running = False
        return
    
    # Set specific camera properties to help with MediaPipe
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Get frame dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {width}x{height}")
    
    # Initialize pose with static images mode off for video
    # Using specific settings to address the NORM_RECT warning
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
        static_image_mode=False,
        enable_segmentation=False,
        smooth_segmentation=False
    ) as pose:
        
        # Initialize variables
        rep_count = 0
        stage = None
        start_time = time.time()
        
        while is_workout_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # Get image dimensions for MediaPipe
            h, w, _ = frame.shape
            
            # Flip the frame horizontally for a selfie-view display
            frame = cv2.flip(frame, 1)
            
            # To improve performance, mark the image as not writeable to pass by reference
            frame.flags.writeable = False
            
            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame with MediaPipe Pose
            # Using a dictionary to provide image dimensions, addressing the NORM_RECT warning
            results = pose.process(frame_rgb)
            
            # Set the frame to writeable again to modify it
            frame.flags.writeable = True
            
            # Convert back to BGR for display
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            if results.pose_landmarks:
                # Draw pose landmarks on the frame
                mp_drawing.draw_landmarks(
                    frame, 
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )
                
                # Get landmarks
                landmarks = results.pose_landmarks.landmark
                
                # Detect exercise
                is_rep = detect_exercise(landmarks, workout_type)
                
                # Count reps
                if is_rep and stage == None:
                    stage = "down"
                elif not is_rep and stage == "down":
                    stage = "up"
                    rep_count += 1
                    
                # Calculate duration and calories
                duration = int(time.time() - start_time)
                calories = int(duration * 0.1)  # Rough estimate: 0.1 calories per second
                
                # Update workout data
                workout_data = {
                    "repCount": rep_count,
                    "duration": duration,
                    "calories": calories,
                    "stage": stage
                }
                
                # Draw rep count on frame
                cv2.putText(frame, f'Reps: {rep_count}', (10,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Duration: {duration}s', (10,70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Calories: {calories}', (10,110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Stage: {stage}', (10,150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Display the frame
            cv2.imshow('Workout Monitoring', frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        print("Workout monitoring stopped")

@app.route('/')
def index():
    return render_template('startworkout.html')

@app.route('/api/workout/start', methods=['POST'])
def start_workout():
    global is_workout_running, workout_thread, workout_data
    
    # Check if a workout is already running
    if is_workout_running:
        return jsonify({"status": "error", "message": "Workout already in progress"})
    
    # Reset workout data
    workout_data = {
        "repCount": 0,
        "duration": 0,
        "calories": 0,
        "stage": None
    }
    
    # Get workout type from request
    data = request.json
    workout_type = data.get('workoutType', 'squat')
    
    # Start workout monitoring in a separate thread
    is_workout_running = True
    workout_thread = threading.Thread(target=workout_monitoring, args=(workout_type,))
    workout_thread.daemon = True
    workout_thread.start()
    
    return jsonify({"status": "success", "message": f"{workout_type.capitalize()} workout started"})

@app.route('/api/workout/data', methods=['GET'])
def get_workout_data():
    return jsonify(workout_data)

@app.route('/api/workout/end', methods=['POST'])
def end_workout():
    global is_workout_running, workout_data
    
    # Check if a workout is running
    if not is_workout_running:
        return jsonify({"status": "error", "message": "No workout in progress"})
    
    # Stop workout monitoring
    is_workout_running = False
    
    # Wait for thread to finish
    if workout_thread:
        workout_thread.join(timeout=1)
    
    # Return final workout data
    return jsonify({
        "status": "success",
        "workout_data": workout_data
    })

@app.route('/bone_density.html')
def bone_density():
    return render_template('bone_density.html')

@app.route("/predict", methods=["POST"])
def predict_bone_density():
    try:
        print("Received bone density prediction request")
        if df is None:
            print("Bone density dataset not available")
            return jsonify({"error": "Bone density prediction service is not available"}), 503
            
        data = request.get_json()
        print(f"Request data: {data}")
        if not data:
            print("No data received in request")
            return jsonify({"error": "No data provided"}), 400
            
        age = float(data.get("age"))
        gender = data.get("gender")
        
        print(f"Received prediction request - Age: {age}, Gender: {gender}")

        if not gender or gender not in ['male', 'female']:
            print(f"Invalid gender: {gender}")
            return jsonify({"error": "Invalid gender"}), 400

        if age < 1 or age > 120:
            print(f"Invalid age: {age}")
            return jsonify({"error": "Age must be between 1 and 120"}), 400

        if gender == 'male' and model_male:
            predicted = model_male.predict([[age]])[0]
        elif gender == 'female' and model_female:
            predicted = model_female.predict([[age]])[0]
        else:
            print(f"Model not available for gender: {gender}")
            return jsonify({"error": "Prediction model not available"}), 503

        predicted = np.clip(predicted, min_bmd, max_bmd)
        score = ((predicted - min_bmd) / (max_bmd - min_bmd)) * 100
        score = np.clip(score, 0, 100)

        print(f"Prediction successful - BMD: {predicted}, Score: {score}")
        return jsonify({
            "predicted_bmd": float(predicted),
            "bmd_score": float(score)
        })
    except Exception as e:
        print(f"Error in bone density prediction: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Print instructions
    print("Starting Workout Tracker Application")
    print("API endpoints:")
    print("  - POST /api/workout/start - Start a workout")
    print("  - GET /api/workout/data - Get current workout data")
    print("  - POST /api/workout/end - End the current workout")
    
    # Start Flask app with logging disabled
    import sys
    cli = sys.modules['flask.cli']
    cli.show_server_banner = lambda *x: None  # Hide Flask server banner
    app.run(debug=False, port=5000, use_reloader=False)  # Set debug=False to reduce logging