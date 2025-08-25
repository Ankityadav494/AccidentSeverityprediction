#!/usr/bin/env python3
"""
Simple Web Interface for Road Accident Severity Prediction
Uses only the 10 specified columns
"""

from flask import Flask, render_template, request, jsonify
import sys
import os
sys.path.append('src')

from simple_prediction import SimpleAccidentPredictor

app = Flask(__name__)

# Initialize the predictor
predictor = None

def init_predictor():
    """Initialize the prediction model"""
    global predictor
    try:
        predictor = SimpleAccidentPredictor()
        return True
    except Exception as e:
        print(f"Error initializing predictor: {e}")
        return False

@app.route('/')
def index():
    """Main page with the simple prediction form"""
    return render_template('simple_index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get form data
        data = request.get_json()
        
        # Convert string values to appropriate types
        scenario = {}
        for key, value in data.items():
            if key in ['Year', 'Visibility Level', 'Number of Vehicles Involved']:
                try:
                    scenario[key] = float(value) if '.' in str(value) else int(value)
                except:
                    scenario[key] = 0
            else:
                scenario[key] = value
        
        # Make prediction
        if predictor is None:
            return jsonify({'error': 'Prediction model not initialized'})
        
        result = predictor.predict_severity(scenario, 'random_forest')
        
        if result is None:
            return jsonify({'error': 'Prediction failed'})
        
        # Format the response
        response = {
            'success': True,
            'prediction': result['predicted_severity'],
            'confidence_scores': result['confidence_scores'],
            'scenario': scenario,
            'interpretation': get_interpretation(result['predicted_severity'])
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'})

def get_interpretation(severity):
    """Get interpretation text for the prediction"""
    interpretations = {
        'Minor': 'This scenario suggests a low-risk accident situation with minimal damage expected. The combination of factors indicates relatively safe driving conditions with good visibility, low traffic, and favorable weather.',
        'Moderate': 'This scenario suggests a moderate-risk accident situation with some damage possible. While conditions are generally acceptable, there are factors that increase risk and require extra attention.',
        'Major': 'This scenario suggests a high-risk accident situation with significant damage possible. Multiple risk factors are present, and extreme caution is strongly advised.',
        'Fatal': 'This scenario suggests a very high-risk accident situation with severe consequences possible. Multiple high-risk factors are present, and this represents a dangerous driving situation.'
    }
    return interpretations.get(severity, 'Unable to interpret prediction.')

if __name__ == '__main__':
    # Initialize the predictor
    if init_predictor():
        print("✅ Simple prediction model loaded successfully!")
        print("🎨 Starting simple web interface...")
        print("📱 Open your browser and go to:")
        print("   🌐 http://localhost:5000")
        print("   🌐 http://127.0.0.1:5000")
        print("\n✨ Features:")
        print("   • Simple accident analysis with 10 essential features")
        print("   • Location and time information")
        print("   • Road and weather conditions")
        print("   • Real-time AI predictions")
        print("   • Beautiful animated design")
        
        # Run the server
        try:
            app.run(debug=False, host='127.0.0.1', port=5000, threaded=True)
        except Exception as e:
            print(f"Error starting server: {e}")
            print("Trying alternative port...")
            try:
                app.run(debug=False, host='127.0.0.1', port=8080, threaded=True)
                print("📱 Now try: http://127.0.0.1:8080")
            except Exception as e2:
                print(f"Failed to start server: {e2}")
    else:
        print("❌ Failed to load simple prediction model!")
