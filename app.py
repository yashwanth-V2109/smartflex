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

# Food nutrition database (simplified for demo)
FOOD_NUTRITION = {
    'apple': {'calories': 95, 'protein': 0.5, 'carbs': 25, 'fat': 0.3},
    'banana': {'calories': 105, 'protein': 1.3, 'carbs': 27, 'fat': 0.4},
    'orange': {'calories': 62, 'protein': 1.2, 'carbs': 15, 'fat': 0.2},
    'sandwich': {'calories': 350, 'protein': 15, 'carbs': 45, 'fat': 12},
    'pizza': {'calories': 285, 'protein': 12, 'carbs': 36, 'fat': 10},
    'salad': {'calories': 150, 'protein': 5, 'carbs': 20, 'fat': 8},
    'hamburger': {'calories': 354, 'protein': 25, 'carbs': 30, 'fat': 20},
    'fries': {'calories': 365, 'protein': 4, 'carbs': 48, 'fat': 17},
    'chicken': {'calories': 335, 'protein': 31, 'carbs': 0, 'fat': 20},
    'rice': {'calories': 130, 'protein': 2.7, 'carbs': 28, 'fat': 0.3},
    'pasta': {'calories': 131, 'protein': 5, 'carbs': 25, 'fat': 1.1},
    'bread': {'calories': 79, 'protein': 2.7, 'carbs': 15, 'fat': 1},
    'egg': {'calories': 70, 'protein': 6, 'carbs': 0.6, 'fat': 5},
    'milk': {'calories': 103, 'protein': 8, 'carbs': 12, 'fat': 2.4},
    'cheese': {'calories': 113, 'protein': 7, 'carbs': 0.4, 'fat': 9},
    'yogurt': {'calories': 100, 'protein': 5, 'carbs': 7, 'fat': 3},
    'coffee': {'calories': 2, 'protein': 0.1, 'carbs': 0, 'fat': 0},
    'tea': {'calories': 2, 'protein': 0, 'carbs': 0, 'fat': 0},
    'water': {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0},
    'soda': {'calories': 150, 'protein': 0, 'carbs': 39, 'fat': 0},
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
                
                # Only process food items with confidence > 0.5
                if confidence > 0.5 and label in FOOD_NUTRITION:
                    detections.append({
                        'label': label,
                        'confidence': confidence,
                        'nutrition': FOOD_NUTRITION[label]
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