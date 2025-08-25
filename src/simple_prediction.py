#!/usr/bin/env python3
"""
Simple Prediction Module for Road Accident Severity Prediction
Uses only the 10 specified columns
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from simple_data_preprocessing import SimpleDataPreprocessor

class SimpleAccidentPredictor:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.models = {}
        self.preprocessor = None
        self.feature_columns = []
        self.label_encoders = {}
        
        # Load models and preprocessing info
        self._load_models()
        self._load_preprocessing_info()
    
    def _load_models(self):
        """Load trained models"""
        print("Loading simple trained models...")
        
        # Load models
        model_files = {
            'decision_tree': 'simple_decision_tree_model.pkl',
            'random_forest': 'simple_random_forest_model.pkl'
        }
        
        for name, filename in model_files.items():
            model_path = os.path.join(self.models_dir, filename)
            if os.path.exists(model_path):
                self.models[name] = joblib.load(model_path)
                print(f"Loaded {name} model")
            else:
                print(f"⚠️  {name} model not found")
        
        print("Simple models loaded successfully")
    
    def _load_preprocessing_info(self):
        """Load preprocessing information"""
        feature_info_path = os.path.join(self.models_dir, 'simple_feature_info.json')
        if os.path.exists(feature_info_path):
            with open(feature_info_path, 'r') as f:
                feature_info = json.load(f)
                self.feature_columns = feature_info.get('feature_columns', [])
                print(f"Loaded simple feature info: {len(self.feature_columns)} features")
        else:
            print("⚠️  Simple feature info not found, using default features")
            self.feature_columns = [
                'Year', 'Visibility Level', 'Number of Vehicles Involved',
                'Country_Encoded', 'Month_Encoded', 'Day of Week_Encoded', 'Time of Day_Encoded',
                'Urban/Rural_Encoded', 'Road Type_Encoded', 'Weather Conditions_Encoded'
            ]
        
        # Load label encoders
        encoders_path = os.path.join(self.models_dir, 'simple_label_encoders.pkl')
        if os.path.exists(encoders_path):
            self.label_encoders = joblib.load(encoders_path)
            print(f"Loaded {len(self.label_encoders)} label encoders")
        else:
            print("⚠️  Label encoders not found")
            self.label_encoders = {}
        
        # Load scaler
        scaler_path = os.path.join(self.models_dir, 'simple_scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print("Loaded scaler")
        else:
            print("⚠️  Scaler not found")
            self.scaler = None
    
    def _prepare_input_data(self, scenario):
        """Prepare input data for prediction"""
        # Create a DataFrame with the scenario data
        input_data = pd.DataFrame([scenario])
        
        # Apply encoding
        input_data = self._apply_encoding(input_data)
        
        # Apply scaling to numeric features if scaler is available
        if self.scaler is not None:
            numeric_features = ['Year', 'Visibility Level', 'Number of Vehicles Involved']
            available_numeric = [col for col in numeric_features if col in input_data.columns]
            if available_numeric:
                input_data[available_numeric] = self.scaler.transform(input_data[available_numeric])
        
        # Select and order features
        available_features = [col for col in self.feature_columns if col in input_data.columns]
        missing_features = [col for col in self.feature_columns if col not in input_data.columns]
        
        if missing_features:
            print(f"⚠️  Missing features: {missing_features}")
            # Fill missing features with default values
            for feature in missing_features:
                if 'Encoded' in feature:
                    input_data[feature] = 0
                else:
                    input_data[feature] = 0
        
        # Ensure all required features are present
        for feature in self.feature_columns:
            if feature not in input_data.columns:
                if 'Encoded' in feature:
                    input_data[feature] = 0
                else:
                    input_data[feature] = 0
        
        # Select features in the correct order
        X = input_data[self.feature_columns]
        
        return X
    
    def _apply_encoding(self, df):
        """Apply label encoding to categorical features"""
        categorical_columns = [
            'Country', 'Month', 'Day of Week', 'Time of Day', 'Urban/Rural',
            'Road Type', 'Weather Conditions'
        ]
        
        for col in categorical_columns:
            if col in df.columns:
                if col in self.label_encoders:
                    # Use saved label encoder
                    try:
                        df[f'{col}_Encoded'] = self.label_encoders[col].transform(df[col].astype(str))
                    except ValueError as e:
                        # Handle unseen values by using a default encoding
                        print(f"⚠️  Unseen value in {col}, using default encoding")
                        unique_values = df[col].unique()
                        encoding = {val: idx for idx, val in enumerate(unique_values)}
                        df[f'{col}_Encoded'] = df[col].map(encoding)
                else:
                    # Fallback to simple encoding if no saved encoder
                    unique_values = df[col].unique()
                    encoding = {val: idx for idx, val in enumerate(unique_values)}
                    df[f'{col}_Encoded'] = df[col].map(encoding)
        
        return df
    
    def predict_severity(self, scenario, model_name='random_forest'):
        """Predict accident severity for a given scenario"""
        try:
            # Validate model name
            if model_name not in self.models:
                available_models = list(self.models.keys())
                raise ValueError(f"Model '{model_name}' not found. Available models: {available_models}")
            
            # Prepare input data
            X = self._prepare_input_data(scenario)
            
            # Make prediction
            model = self.models[model_name]
            prediction = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0]
            
            # Map prediction back to severity labels
            severity_mapping = {0: 'Minor', 1: 'Moderate', 2: 'Major', 3: 'Fatal'}
            predicted_severity = severity_mapping.get(prediction, 'Unknown')
            
            # Create confidence scores
            confidence_scores = {
                'Minor': probabilities[0] if len(probabilities) > 0 else 0,
                'Moderate': probabilities[1] if len(probabilities) > 1 else 0,
                'Major': probabilities[2] if len(probabilities) > 2 else 0,
                'Fatal': probabilities[3] if len(probabilities) > 3 else 0
            }
            
            return {
                'predicted_severity': predicted_severity,
                'confidence_scores': confidence_scores,
                'model_used': model_name
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None
    
    def get_available_models(self):
        """Get list of available models"""
        return list(self.models.keys())
    
    def get_feature_importance(self, model_name='random_forest'):
        """Get feature importance for a specific model"""
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        return None

def create_sample_scenarios():
    """Create sample scenarios for testing"""
    scenarios = [
        {
            'name': 'Low Risk Scenario',
            'data': {
                'Country': 'USA',
                'Year': 2023,
                'Month': 'June',
                'Day of Week': 'Monday',
                'Time of Day': 'Morning',
                'Urban/Rural': 'Urban',
                'Road Type': 'Street',
                'Weather Conditions': 'Clear',
                'Visibility Level': 200,
                'Number of Vehicles Involved': 1
            }
        },
        {
            'name': 'High Risk Scenario',
            'data': {
                'Country': 'USA',
                'Year': 2023,
                'Month': 'December',
                'Day of Week': 'Saturday',
                'Time of Day': 'Night',
                'Urban/Rural': 'Rural',
                'Road Type': 'Highway',
                'Weather Conditions': 'Snow-covered',
                'Visibility Level': 50,
                'Number of Vehicles Involved': 3
            }
        }
    ]
    
    return scenarios

if __name__ == "__main__":
    # Test the predictor
    predictor = SimpleAccidentPredictor()
    
    # Create sample scenarios
    scenarios = create_sample_scenarios()
    
    print("🧪 Testing simple prediction system...")
    
    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        result = predictor.predict_severity(scenario['data'])
        
        if result:
            print(f"Predicted Severity: {result['predicted_severity']}")
            print("Confidence Scores:")
            for severity, confidence in result['confidence_scores'].items():
                print(f"  {severity}: {confidence:.3f}")
        else:
            print("❌ Prediction failed!")
    
    print("\n✅ Simple prediction system test completed!")
