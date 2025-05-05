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
import torch
import base64
from io import BytesIO
from PIL import Image

# Initialize Flask application
app = Flask(__name__, static_folder='public', template_folder='public')
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

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
    try:
        if workout_type == "bicep_curl":
            # Get coordinates for both arms
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            
            # Calculate angles for both arms
            left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            # Count rep when either arm is curled (angle < 30 degrees)
            # and the other arm is extended (angle > 160 degrees)
            return (left_angle < 30 and right_angle > 160) or (right_angle < 30 and left_angle > 160)
        
        elif workout_type == "squat":
            # Get coordinates for both legs
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            
            # Calculate angles for both legs
            left_angle = calculate_angle(left_hip, left_knee, left_ankle)
            right_angle = calculate_angle(right_hip, right_knee, right_ankle)
            
            # Count rep when both knees are bent (angle < 100 degrees)
            # and hips are below knees (checking y-coordinates)
            return (left_angle < 100 and right_angle < 100 and 
                   left_hip[1] > left_knee[1] and right_hip[1] > right_knee[1])
        
        elif workout_type == "pushup":
            # Get coordinates for both arms
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            
            # Calculate angles for both arms
            left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            # Count rep when both arms are bent (angle < 90 degrees)
            # and body is parallel to ground (checking shoulder and hip alignment)
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
            shoulder_hip_alignment = abs(left_shoulder[1] - left_hip) < 0.1 and abs(right_shoulder[1] - right_hip) < 0.1
            
            return (left_angle < 90 and right_angle < 90 and shoulder_hip_alignment)
        
        return False
    except Exception as e:
        print(f"Error in exercise detection: {str(e)}")
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
    with mp_pose.Pose(
        min_detection_confidence=0.7,  # Increased confidence threshold
        min_tracking_confidence=0.7,   # Increased tracking confidence
        model_complexity=1,
        static_image_mode=False,
        enable_segmentation=False,
        smooth_segmentation=False
    ) as pose:
        
        # Initialize variables
        rep_count = 0
        stage = None
        start_time = time.time()
        last_rep_time = time.time()
        rep_cooldown = 1.0  # Minimum time between reps in seconds
        
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
                
                # Detect exercise with cooldown
                current_time = time.time()
                is_rep = detect_exercise(landmarks, workout_type)
                
                # Count reps with cooldown
                if is_rep and stage == None and (current_time - last_rep_time) > rep_cooldown:
                    stage = "down"
                elif not is_rep and stage == "down" and (current_time - last_rep_time) > rep_cooldown:
                    stage = "up"
                    rep_count += 1
                    last_rep_time = current_time
                    
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
                
                # Draw rep count and other info on frame
                cv2.putText(frame, f'Reps: {rep_count}', (10,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Duration: {duration}s', (10,70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Calories: {calories}', (10,110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Stage: {stage}', (10,150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                
                # Draw exercise-specific guidance
                if workout_type == "squat":
                    cv2.putText(frame, 'Keep knees behind toes', (10,190), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
                elif workout_type == "bicep_curl":
                    cv2.putText(frame, 'Keep elbows close to body', (10,190), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
                elif workout_type == "pushup":
                    cv2.putText(frame, 'Keep body straight', (10,190), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            
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

# Food database with nutritional information per 100g
FOOD_DATABASE = {
    'apple': {'calories': 52, 'protein': 0.3, 'carbs': 13.8, 'fat': 0.2},
    'banana': {'calories': 89, 'protein': 1.1, 'carbs': 22.8, 'fat': 0.3},
    'orange': {'calories': 47, 'protein': 0.9, 'carbs': 11.8, 'fat': 0.1},
    'chicken breast': {'calories': 165, 'protein': 31, 'carbs': 0, 'fat': 3.6},
    'rice': {'calories': 130, 'protein': 2.7, 'carbs': 28.2, 'fat': 0.3},
    'salmon': {'calories': 208, 'protein': 22, 'carbs': 0, 'fat': 13},
    'egg': {'calories': 155, 'protein': 12.6, 'carbs': 0.6, 'fat': 11.3},
    'bread': {'calories': 265, 'protein': 9, 'carbs': 49, 'fat': 3.2},
    'milk': {'calories': 42, 'protein': 3.4, 'carbs': 5, 'fat': 1},
    'yogurt': {'calories': 59, 'protein': 3.5, 'carbs': 4.7, 'fat': 3.3},
    'beef': {'calories': 250, 'protein': 26, 'carbs': 0, 'fat': 17},
    'pork': {'calories': 242, 'protein': 27, 'carbs': 0, 'fat': 14},
    'turkey': {'calories': 157, 'protein': 29, 'carbs': 0, 'fat': 3.6},
    'tuna': {'calories': 132, 'protein': 28, 'carbs': 0, 'fat': 1.2},
    'shrimp': {'calories': 99, 'protein': 24, 'carbs': 0.2, 'fat': 1.7},
    'lentils': {'calories': 116, 'protein': 9, 'carbs': 20, 'fat': 0.4},
    'chickpeas': {'calories': 164, 'protein': 8.9, 'carbs': 27.4, 'fat': 2.6},
    'black beans': {'calories': 132, 'protein': 8.9, 'carbs': 23.7, 'fat': 0.5},
    'potato': {'calories': 77, 'protein': 2, 'carbs': 17.2, 'fat': 0.1},
    'sweet potato': {'calories': 86, 'protein': 1.6, 'carbs': 20.1, 'fat': 0.1},
    'carrots': {'calories': 41, 'protein': 0.9, 'carbs': 9.6, 'fat': 0.2},
    'broccoli': {'calories': 34, 'protein': 2.8, 'carbs': 6.6, 'fat': 0.4},
    'spinach': {'calories': 23, 'protein': 2.9, 'carbs': 3.6, 'fat': 0.4},
    'bell pepper': {'calories': 31, 'protein': 1, 'carbs': 6, 'fat': 0.3},
    'mushrooms': {'calories': 22, 'protein': 3.1, 'carbs': 3.3, 'fat': 0.3},
    'avocado': {'calories': 160, 'protein': 2, 'carbs': 8.5, 'fat': 14.7},
    'almonds': {'calories': 579, 'protein': 21.2, 'carbs': 21.7, 'fat': 49.9},
    'peanut butter': {'calories': 588, 'protein': 25, 'carbs': 20, 'fat': 50},
    'honey': {'calories': 304, 'protein': 0.3, 'carbs': 82.4, 'fat': 0},
    'maple syrup': {'calories': 260, 'protein': 0, 'carbs': 67, 'fat': 0},
    'olive oil': {'calories': 884, 'protein': 0, 'carbs': 0, 'fat': 100},
    'coconut oil': {'calories': 862, 'protein': 0, 'carbs': 0, 'fat': 100},
    'butter': {'calories': 717, 'protein': 0.9, 'carbs': 0.1, 'fat': 81.1},
    'cheese': {'calories': 402, 'protein': 25, 'carbs': 1.3, 'fat': 33.1},
    'quinoa': {'calories': 120, 'protein': 4.4, 'carbs': 21.3, 'fat': 1.9},
    'oats': {'calories': 389, 'protein': 16.9, 'carbs': 66.3, 'fat': 6.9},
    'pasta': {'calories': 158, 'protein': 5.8, 'carbs': 31, 'fat': 0.9},
    'tofu': {'calories': 76, 'protein': 8, 'carbs': 1.9, 'fat': 4.8},
    'tempeh': {'calories': 192, 'protein': 20.3, 'carbs': 7.7, 'fat': 10.8},
    'seitan': {'calories': 370, 'protein': 75, 'carbs': 14, 'fat': 2},
    'chia seeds': {'calories': 486, 'protein': 17, 'carbs': 42, 'fat': 31},
    'flax seeds': {'calories': 534, 'protein': 18.3, 'carbs': 28.9, 'fat': 42.2},
    'pumpkin seeds': {'calories': 559, 'protein': 30.2, 'carbs': 10.7, 'fat': 49.1},
    'sunflower seeds': {'calories': 584, 'protein': 20.8, 'carbs': 20, 'fat': 51.5},
    'walnuts': {'calories': 654, 'protein': 15.2, 'carbs': 13.7, 'fat': 65.2},
    'cashews': {'calories': 553, 'protein': 18.2, 'carbs': 30.2, 'fat': 43.8},
    'peanuts': {'calories': 567, 'protein': 25.8, 'carbs': 16.1, 'fat': 49.2},
    'pecans': {'calories': 691, 'protein': 9.2, 'carbs': 13.9, 'fat': 72},
    'pistachios': {'calories': 562, 'protein': 20.2, 'carbs': 27.2, 'fat': 45.3},
    'blueberries': {'calories': 57, 'protein': 0.7, 'carbs': 14.5, 'fat': 0.3},
    'strawberries': {'calories': 32, 'protein': 0.7, 'carbs': 7.7, 'fat': 0.3},
    'raspberries': {'calories': 52, 'protein': 1.2, 'carbs': 11.9, 'fat': 0.7},
    'blackberries': {'calories': 43, 'protein': 1.4, 'carbs': 9.6, 'fat': 0.5},
    'grapes': {'calories': 69, 'protein': 0.6, 'carbs': 18.1, 'fat': 0.2},
    'watermelon': {'calories': 30, 'protein': 0.6, 'carbs': 7.6, 'fat': 0.2},
    'pineapple': {'calories': 50, 'protein': 0.5, 'carbs': 13.1, 'fat': 0.1},
    'mango': {'calories': 60, 'protein': 0.8, 'carbs': 15, 'fat': 0.4},
    'kiwi': {'calories': 61, 'protein': 1.1, 'carbs': 14.7, 'fat': 0.5},
    'pear': {'calories': 57, 'protein': 0.4, 'carbs': 15.2, 'fat': 0.1},
    'peach': {'calories': 39, 'protein': 0.9, 'carbs': 9.5, 'fat': 0.3},
    'plum': {'calories': 46, 'protein': 0.7, 'carbs': 11.4, 'fat': 0.3},
    'cherry': {'calories': 50, 'protein': 1, 'carbs': 12.2, 'fat': 0.3},
    'cucumber': {'calories': 15, 'protein': 0.7, 'carbs': 3.6, 'fat': 0.1},
    'tomato': {'calories': 18, 'protein': 0.9, 'carbs': 3.9, 'fat': 0.2},
    'lettuce': {'calories': 15, 'protein': 1.4, 'carbs': 2.9, 'fat': 0.2},
    'kale': {'calories': 49, 'protein': 4.3, 'carbs': 8.8, 'fat': 0.9},
    'cauliflower': {'calories': 25, 'protein': 1.9, 'carbs': 5, 'fat': 0.3},
    'brussels sprouts': {'calories': 43, 'protein': 3.4, 'carbs': 8.9, 'fat': 0.3},
    'asparagus': {'calories': 20, 'protein': 2.2, 'carbs': 3.9, 'fat': 0.2},
    'zucchini': {'calories': 17, 'protein': 1.2, 'carbs': 3.1, 'fat': 0.3},
    'eggplant': {'calories': 25, 'protein': 1, 'carbs': 6, 'fat': 0.2},
    'corn': {'calories': 86, 'protein': 3.2, 'carbs': 19, 'fat': 1.2},
    'green beans': {'calories': 31, 'protein': 1.8, 'carbs': 7, 'fat': 0.2},
    'peas': {'calories': 81, 'protein': 5.4, 'carbs': 14.5, 'fat': 0.4},
    'sweet corn': {'calories': 86, 'protein': 3.2, 'carbs': 19, 'fat': 1.2},
    'artichoke': {'calories': 47, 'protein': 3.3, 'carbs': 10.5, 'fat': 0.2},
    'beetroot': {'calories': 43, 'protein': 1.6, 'carbs': 9.6, 'fat': 0.2},
    'radish': {'calories': 16, 'protein': 0.7, 'carbs': 3.4, 'fat': 0.1},
    'turnip': {'calories': 28, 'protein': 0.9, 'carbs': 6.4, 'fat': 0.1},
    'rutabaga': {'calories': 37, 'protein': 1.2, 'carbs': 8.6, 'fat': 0.2},
    'parsnip': {'calories': 75, 'protein': 1.2, 'carbs': 18, 'fat': 0.3},
    'celery': {'calories': 16, 'protein': 0.7, 'carbs': 3, 'fat': 0.2},
    'fennel': {'calories': 31, 'protein': 1.2, 'carbs': 7.3, 'fat': 0.2},
    'leek': {'calories': 61, 'protein': 1.5, 'carbs': 14.2, 'fat': 0.3},
    'onion': {'calories': 40, 'protein': 1.1, 'carbs': 9.3, 'fat': 0.1},
    'garlic': {'calories': 149, 'protein': 6.4, 'carbs': 33.1, 'fat': 0.5},
    'ginger': {'calories': 80, 'protein': 1.8, 'carbs': 17.8, 'fat': 0.8},
    'turmeric': {'calories': 312, 'protein': 9.7, 'carbs': 67.1, 'fat': 3.2},
    'cinnamon': {'calories': 247, 'protein': 3.9, 'carbs': 80.6, 'fat': 1.2},
    'cumin': {'calories': 375, 'protein': 17.8, 'carbs': 44.2, 'fat': 22.3},
    'paprika': {'calories': 282, 'protein': 14.1, 'carbs': 53.9, 'fat': 12.9},
    'black pepper': {'calories': 251, 'protein': 10.4, 'carbs': 64, 'fat': 3.3},
    'salt': {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0},
    'sugar': {'calories': 387, 'protein': 0, 'carbs': 100, 'fat': 0},
    'brown sugar': {'calories': 380, 'protein': 0, 'carbs': 98, 'fat': 0},
    'maple syrup': {'calories': 260, 'protein': 0, 'carbs': 67, 'fat': 0},
    'honey': {'calories': 304, 'protein': 0.3, 'carbs': 82.4, 'fat': 0},
    'molasses': {'calories': 290, 'protein': 0, 'carbs': 75, 'fat': 0},
    'agave nectar': {'calories': 310, 'protein': 0.1, 'carbs': 76, 'fat': 0.5},
    'stevia': {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0},
    'coconut sugar': {'calories': 380, 'protein': 0, 'carbs': 100, 'fat': 0},
    'date sugar': {'calories': 350, 'protein': 1.8, 'carbs': 93, 'fat': 0},
    'monk fruit': {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0},
    'erythritol': {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0},
    'xylitol': {'calories': 240, 'protein': 0, 'carbs': 100, 'fat': 0},
    'sucralose': {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0},
    'aspartame': {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0},
    'saccharin': {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0},
    'acesulfame k': {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0},
    'neotame': {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0},
    'advantame': {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0},
    'alitame': {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0},
    'cyclamate': {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0},
    'thaumatin': {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0},
    'glycyrrhizin': {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0},
    'miraculin': {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0},
    'brazzein': {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0},
    'curculin': {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0},
    'mabinlin': {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0},
    'pentadin': {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0},
    'monellin': {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0},
    'osladin': {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0},
    'phyllodulcin': {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0},
    'pterocaryoside': {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0},
    'siraitia grosvenorii': {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0},
    'steviol glycoside': {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0}
}

@app.route('/api/calculate-nutrients', methods=['POST'])
def calculate_nutrients():
    try:
        data = request.get_json()
        food_items = data.get('foodItems', [])
        
        if not food_items:
            return jsonify({"error": "No food items provided"}), 400
        
        total_calories = 0
        total_protein = 0
        total_carbs = 0
        total_fat = 0
        items = {}
        
        for item in food_items:
            name = item['name'].lower()
            quantity = item['quantity']
            
            if name in FOOD_DATABASE:
                food_info = FOOD_DATABASE[name]
                # Calculate nutrients based on quantity (per 100g)
                multiplier = quantity / 100
                
                calories = round(food_info['calories'] * multiplier)
                protein = round(food_info['protein'] * multiplier, 1)
                carbs = round(food_info['carbs'] * multiplier, 1)
                fat = round(food_info['fat'] * multiplier, 1)
                
                total_calories += calories
                total_protein += protein
                total_carbs += carbs
                total_fat += fat
                
                items[name] = {
                    "calories": calories,
                    "protein": protein,
                    "carbs": carbs,
                    "fat": fat
                }
            else:
                return jsonify({"error": f"Food item '{name}' not found in database"}), 400
        
        return jsonify({
            "totalCalories": total_calories,
            "totalProtein": round(total_protein, 1),
            "totalCarbs": round(total_carbs, 1),
            "totalFat": round(total_fat, 1),
            "items": items
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Food nutrition database (simplified for demo)
FOOD_NUTRITION = {
    'apple': {
        'calories': 52,
        'protein': 0.3,
        'carbs': 13.8,
        'fat': 0.2,
        'fiber': 2.4,
        'sugar': 10.4,
        'vitamin_c': 4.6,
        'potassium': 107,
        'calcium': 6,
        'iron': 0.1
    },
    'banana': {
        'calories': 89,
        'protein': 1.1,
        'carbs': 22.8,
        'fat': 0.3,
        'fiber': 2.6,
        'sugar': 12.2,
        'vitamin_c': 8.7,
        'potassium': 358,
        'calcium': 5,
        'iron': 0.3
    },
    'chicken breast': {
        'calories': 165,
        'protein': 31,
        'carbs': 0,
        'fat': 3.6,
        'fiber': 0,
        'sugar': 0,
        'vitamin_c': 0,
        'potassium': 256,
        'calcium': 15,
        'iron': 0.7
    },
    'rice': {
        'calories': 130,
        'protein': 2.7,
        'carbs': 28.2,
        'fat': 0.3,
        'fiber': 0.4,
        'sugar': 0.1,
        'vitamin_c': 0,
        'potassium': 35,
        'calcium': 10,
        'iron': 0.2
    },
    'salmon': {
        'calories': 208,
        'protein': 22,
        'carbs': 0,
        'fat': 13,
        'fiber': 0,
        'sugar': 0,
        'vitamin_c': 3.9,
        'potassium': 363,
        'calcium': 9,
        'iron': 0.3
    },
    'egg': {
        'calories': 155,
        'protein': 12.6,
        'carbs': 0.6,
        'fat': 11.3,
        'fiber': 0,
        'sugar': 0.6,
        'vitamin_c': 0,
        'potassium': 126,
        'calcium': 56,
        'iron': 1.8
    },
    'bread': {
        'calories': 265,
        'protein': 9,
        'carbs': 49,
        'fat': 3.2,
        'fiber': 2.7,
        'sugar': 5,
        'vitamin_c': 0,
        'potassium': 115,
        'calcium': 260,
        'iron': 3.6
    },
    'milk': {
        'calories': 42,
        'protein': 3.4,
        'carbs': 5,
        'fat': 1,
        'fiber': 0,
        'sugar': 5,
        'vitamin_c': 0,
        'potassium': 150,
        'calcium': 125,
        'iron': 0.1
    },
    'yogurt': {
        'calories': 59,
        'protein': 3.5,
        'carbs': 4.7,
        'fat': 3.3,
        'fiber': 0,
        'sugar': 4.7,
        'vitamin_c': 0.5,
        'potassium': 141,
        'calcium': 121,
        'iron': 0.1
    },
    'beef': {
        'calories': 250,
        'protein': 26,
        'carbs': 0,
        'fat': 17,
        'fiber': 0,
        'sugar': 0,
        'vitamin_c': 0,
        'potassium': 318,
        'calcium': 12,
        'iron': 2.6
    },
    'pork': {
        'calories': 242,
        'protein': 27,
        'carbs': 0,
        'fat': 14,
        'fiber': 0,
        'sugar': 0,
        'vitamin_c': 0,
        'potassium': 384,
        'calcium': 19,
        'iron': 0.9
    },
    'turkey': {
        'calories': 157,
        'protein': 29,
        'carbs': 0,
        'fat': 3.6,
        'fiber': 0,
        'sugar': 0,
        'vitamin_c': 0,
        'potassium': 289,
        'calcium': 13,
        'iron': 1.1
    },
    'tuna': {
        'calories': 132,
        'protein': 28,
        'carbs': 0,
        'fat': 1.2,
        'fiber': 0,
        'sugar': 0,
        'vitamin_c': 0,
        'potassium': 441,
        'calcium': 8,
        'iron': 1.3
    },
    'shrimp': {
        'calories': 99,
        'protein': 24,
        'carbs': 0.2,
        'fat': 1.7,
        'fiber': 0,
        'sugar': 0,
        'vitamin_c': 2.2,
        'potassium': 220,
        'calcium': 70,
        'iron': 0.5
    },
    'lentils': {
        'calories': 116,
        'protein': 9,
        'carbs': 20,
        'fat': 0.4,
        'fiber': 7.9,
        'sugar': 1.8,
        'vitamin_c': 1.5,
        'potassium': 369,
        'calcium': 19,
        'iron': 3.3
    },
    'chickpeas': {
        'calories': 164,
        'protein': 8.9,
        'carbs': 27.4,
        'fat': 2.6,
        'fiber': 7.6,
        'sugar': 4.8,
        'vitamin_c': 1.3,
        'potassium': 291,
        'calcium': 49,
        'iron': 2.9
    },
    'black beans': {
        'calories': 132,
        'protein': 8.9,
        'carbs': 23.7,
        'fat': 0.5,
        'fiber': 8.7,
        'sugar': 0.3,
        'vitamin_c': 0,
        'potassium': 355,
        'calcium': 27,
        'iron': 2.1
    },
    'potato': {
        'calories': 77,
        'protein': 2,
        'carbs': 17.2,
        'fat': 0.1,
        'fiber': 2.2,
        'sugar': 0.8,
        'vitamin_c': 19.7,
        'potassium': 421,
        'calcium': 12,
        'iron': 0.8
    },
    'sweet potato': {
        'calories': 86,
        'protein': 1.6,
        'carbs': 20.1,
        'fat': 0.1,
        'fiber': 3,
        'sugar': 4.2,
        'vitamin_c': 2.4,
        'potassium': 337,
        'calcium': 30,
        'iron': 0.6
    },
    'carrots': {
        'calories': 41,
        'protein': 0.9,
        'carbs': 9.6,
        'fat': 0.2,
        'fiber': 2.8,
        'sugar': 4.7,
        'vitamin_c': 5.9,
        'potassium': 320,
        'calcium': 33,
        'iron': 0.3
    },
    'broccoli': {
        'calories': 34,
        'protein': 2.8,
        'carbs': 6.6,
        'fat': 0.4,
        'fiber': 2.6,
        'sugar': 1.7,
        'vitamin_c': 89.2,
        'potassium': 316,
        'calcium': 47,
        'iron': 0.7
    },
    'spinach': {
        'calories': 23,
        'protein': 2.9,
        'carbs': 3.6,
        'fat': 0.4,
        'fiber': 2.2,
        'sugar': 0.4,
        'vitamin_c': 28.1,
        'potassium': 558,
        'calcium': 99,
        'iron': 2.7
    },
    'bell pepper': {
        'calories': 31,
        'protein': 1,
        'carbs': 6,
        'fat': 0.3,
        'fiber': 2.1,
        'sugar': 4.2,
        'vitamin_c': 127.7,
        'potassium': 211,
        'calcium': 7,
        'iron': 0.4
    },
    'mushrooms': {
        'calories': 22,
        'protein': 3.1,
        'carbs': 3.3,
        'fat': 0.3,
        'fiber': 1,
        'sugar': 1.7,
        'vitamin_c': 2.1,
        'potassium': 318,
        'calcium': 3,
        'iron': 0.5
    },
    'avocado': {
        'calories': 160,
        'protein': 2,
        'carbs': 8.5,
        'fat': 14.7,
        'fiber': 6.7,
        'sugar': 0.7,
        'vitamin_c': 10,
        'potassium': 485,
        'calcium': 12,
        'iron': 0.6
    },
    'almonds': {
        'calories': 579,
        'protein': 21.2,
        'carbs': 21.7,
        'fat': 49.9,
        'fiber': 12.5,
        'sugar': 4.4,
        'vitamin_c': 0,
        'potassium': 733,
        'calcium': 269,
        'iron': 3.7
    },
    'peanut butter': {
        'calories': 588,
        'protein': 25,
        'carbs': 20,
        'fat': 50,
        'fiber': 6,
        'sugar': 9,
        'vitamin_c': 0,
        'potassium': 649,
        'calcium': 49,
        'iron': 1.9
    },
    'honey': {
        'calories': 304,
        'protein': 0.3,
        'carbs': 82.4,
        'fat': 0,
        'fiber': 0.2,
        'sugar': 82.1,
        'vitamin_c': 0.5,
        'potassium': 52,
        'calcium': 6,
        'iron': 0.4
    },
    'maple syrup': {
        'calories': 260,
        'protein': 0,
        'carbs': 67,
        'fat': 0,
        'fiber': 0,
        'sugar': 67,
        'vitamin_c': 0,
        'potassium': 212,
        'calcium': 102,
        'iron': 0.1
    },
    'olive oil': {
        'calories': 884,
        'protein': 0,
        'carbs': 0,
        'fat': 100,
        'fiber': 0,
        'sugar': 0,
        'vitamin_c': 0,
        'potassium': 1,
        'calcium': 1,
        'iron': 0.6
    },
    'coconut oil': {
        'calories': 862,
        'protein': 0,
        'carbs': 0,
        'fat': 100,
        'fiber': 0,
        'sugar': 0,
        'vitamin_c': 0,
        'potassium': 0,
        'calcium': 0,
        'iron': 0
    },
    'butter': {
        'calories': 717,
        'protein': 0.9,
        'carbs': 0.1,
        'fat': 81.1,
        'fiber': 0,
        'sugar': 0.1,
        'vitamin_c': 0,
        'potassium': 24,
        'calcium': 24,
        'iron': 0
    },
    'cheese': {
        'calories': 402,
        'protein': 25,
        'carbs': 1.3,
        'fat': 33.1,
        'fiber': 0,
        'sugar': 0.5,
        'vitamin_c': 0,
        'potassium': 98,
        'calcium': 721,
        'iron': 0.2
    },
    'quinoa': {
        'calories': 120,
        'protein': 4.4,
        'carbs': 21.3,
        'fat': 1.9,
        'fiber': 2.8,
        'sugar': 0.9,
        'vitamin_c': 0,
        'potassium': 172,
        'calcium': 17,
        'iron': 1.5
    },
    'oats': {
        'calories': 389,
        'protein': 16.9,
        'carbs': 66.3,
        'fat': 6.9,
        'fiber': 10.6,
        'sugar': 0,
        'vitamin_c': 0,
        'potassium': 429,
        'calcium': 54,
        'iron': 4.7
    },
    'pasta': {
        'calories': 158,
        'protein': 5.8,
        'carbs': 31,
        'fat': 0.9,
        'fiber': 1.8,
        'sugar': 0.6,
        'vitamin_c': 0,
        'potassium': 44,
        'calcium': 7,
        'iron': 1.3
    },
    'tofu': {
        'calories': 76,
        'protein': 8,
        'carbs': 1.9,
        'fat': 4.8,
        'fiber': 0.3,
        'sugar': 0.6,
        'vitamin_c': 0.1,
        'potassium': 121,
        'calcium': 350,
        'iron': 5.4
    },
    'tempeh': {
        'calories': 192,
        'protein': 20.3,
        'carbs': 7.7,
        'fat': 10.8,
        'fiber': 0,
        'sugar': 0,
        'vitamin_c': 0,
        'potassium': 412,
        'calcium': 111,
        'iron': 2.7
    },
    'seitan': {
        'calories': 370,
        'protein': 75,
        'carbs': 14,
        'fat': 2,
        'fiber': 0.6,
        'sugar': 2.9,
        'vitamin_c': 0,
        'potassium': 100,
        'calcium': 142,
        'iron': 5.2
    }
}

# Load YOLOv5 model
try:
    print("Loading YOLOv5 model...")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt')
    print("YOLOv5 model loaded successfully")
except Exception as e:
    print(f"Error loading YOLOv5 model: {str(e)}")
    model = None

@app.route('/api/detect-food', methods=['POST'])
def detect_food():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
            
        # Get the image file
        image_file = request.files['image']
        
        # Read the image
        img = Image.open(image_file)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Perform object detection
        if model is not None:
            results = model(img)
            
            # Get detections
            detections = []
            for *box, conf, cls in results.xyxy[0]:
                label = model.names[int(cls)]
                confidence = float(conf)
                
                # Lower the confidence threshold to detect more items
                if confidence > 0.3:
                    # Try to match the label with our food database
                    matched_label = None
                    
                    # Check for rice-related items
                    if 'rice' in label.lower():
                        matched_label = 'rice'
                    # Check for other food items
                    elif label in FOOD_NUTRITION:
                        matched_label = label
                    # Try to match with similar names
                    else:
                        for food_name in FOOD_NUTRITION.keys():
                            if food_name in label.lower() or label.lower() in food_name:
                                matched_label = food_name
                                break
                    
                    if matched_label:
                        detections.append({
                            'label': matched_label,
                            'confidence': confidence,
                            'nutrition': FOOD_NUTRITION[matched_label]
                        })
            
            # Draw bounding boxes on the image
            img_np = np.array(img)
            for detection in detections:
                # Get the box coordinates
                box = results.xyxy[0][detections.index(detection)][:4].cpu().numpy()
                x1, y1, x2, y2 = map(int, box)
                
                # Draw rectangle
                cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label and confidence
                label = f"{detection['label']} ({detection['confidence']:.2f})"
                cv2.putText(img_np, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Convert annotated image to base64
            _, buffer = cv2.imencode('.jpg', img_np)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return jsonify({
                'image': img_base64,
                'detections': detections
            })
        else:
            return jsonify({'error': 'Model not loaded'}), 500
            
    except Exception as e:
        print(f"Error in food detection: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/get-nutrition', methods=['POST'])
def get_nutrition():
    try:
        data = request.get_json()
        food_name = data.get('foodName', '').lower()
        quantity = float(data.get('quantity', 100))  # Default to 100g if not specified
        
        # Find the food in our database
        food_data = None
        for food_key in FOOD_NUTRITION:
            if food_name in food_key or food_key in food_name:
                food_data = FOOD_NUTRITION[food_key]
                break
        
        if not food_data:
            return jsonify({'error': 'Food not found in database'}), 404
        
        # Calculate nutrition based on quantity (assuming values are per 100g)
        scale_factor = quantity / 100.0
        nutrition_info = {
            'calories': round(food_data['calories'] * scale_factor, 1),
            'protein': round(food_data['protein'] * scale_factor, 1),
            'carbs': round(food_data['carbs'] * scale_factor, 1),
            'fat': round(food_data['fat'] * scale_factor, 1)
        }
        
        return jsonify(nutrition_info)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Print instructions
    print("Starting Workout Tracker Application")
    print("API endpoints:")
    print("  - POST /api/workout/start - Start a workout")
    print("  - GET /api/workout/data - Get current workout data")
    print("  - POST /api/workout/end - End the current workout")
    print("  - POST /api/detect-food - Analyze food image")
    
    # Start Flask app with logging disabled
    import sys
    cli = sys.modules['flask.cli']
    cli.show_server_banner = lambda *x: None  # Hide Flask server banner
    app.run(debug=False, port=5000, use_reloader=False)  # Set debug=False to reduce logging