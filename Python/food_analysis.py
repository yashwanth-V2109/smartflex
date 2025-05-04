from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
import json
import base64

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Simplified food calorie database
FOOD_CALORIES = {
    'apple': {'calories': 95, 'protein': 0.5, 'carbs': 25, 'fat': 0.3},
    'banana': {'calories': 105, 'protein': 1.3, 'carbs': 27, 'fat': 0.3},
    'pizza': {'calories': 266, 'protein': 11, 'carbs': 33, 'fat': 10},
    'hamburger': {'calories': 354, 'protein': 20, 'carbs': 29, 'fat': 17},
    'salad': {'calories': 20, 'protein': 1, 'carbs': 4, 'fat': 0.2},
    'rice': {'calories': 130, 'protein': 2.7, 'carbs': 28, 'fat': 0.3},
    'chicken': {'calories': 165, 'protein': 31, 'carbs': 0, 'fat': 3.6},
    'fish': {'calories': 206, 'protein': 22, 'carbs': 0, 'fat': 12},
    'bread': {'calories': 265, 'protein': 9, 'carbs': 49, 'fat': 3.2},
    'egg': {'calories': 72, 'protein': 6.3, 'carbs': 0.6, 'fat': 4.8}
}

def estimate_calories(food_items):
    total_calories = 0
    total_protein = 0
    total_carbs = 0
    total_fat = 0
    
    for item in food_items:
        food_name = item['name'].lower()
        confidence = item['confidence']
        
        # Find matching food in database
        for db_food, nutrition in FOOD_CALORIES.items():
            if db_food in food_name:
                # Adjust values based on confidence
                total_calories += nutrition['calories'] * confidence
                total_protein += nutrition['protein'] * confidence
                total_carbs += nutrition['carbs'] * confidence
                total_fat += nutrition['fat'] * confidence
                break
    
    return {
        'calories': round(total_calories),
        'protein': round(total_protein, 1),
        'carbs': round(total_carbs, 1),
        'fat': round(total_fat, 1)
    }

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'status': 'ok', 'message': 'Flask server is running'})

@app.route('/analyze', methods=['POST'])
def analyze_food():
    try:
        # Handle file upload
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400
            
            # Save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Read and process the image
            img = cv2.imread(filepath)
            if img is None:
                return jsonify({'error': 'Invalid image file'}), 400
            
            # Clean up the uploaded file
            os.remove(filepath)
            
        # Handle base64 image
        elif 'imageData' in request.json:
            # Remove the data URL prefix if present
            image_data = request.json['imageData']
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            # Decode base64 image
            img_bytes = base64.b64decode(image_data)
            img_arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            
            if img is None:
                return jsonify({'error': 'Invalid image data'}), 400
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Perform detection
        results = model(img_rgb)
        
        # Process results
        food_items = []
        for pred in results.xyxy[0]:  # xyxy format: x1, y1, x2, y2, confidence, class
            if pred[4] > 0.3:  # Confidence threshold
                class_id = int(pred[5])
                class_name = results.names[class_id]
                confidence = float(pred[4])
                
                food_items.append({
                    'name': class_name,
                    'confidence': confidence,
                    'bbox': pred[:4].tolist()
                })
        
        # Estimate calories and nutrients
        nutrition = estimate_calories(food_items)
        
        return jsonify({
            'foodItems': food_items,
            'nutrition': nutrition
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 