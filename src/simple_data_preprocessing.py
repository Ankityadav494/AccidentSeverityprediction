#!/usr/bin/env python3
"""
Simple Data Preprocessing for Road Accident Severity Prediction
Uses only the 10 specified columns
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

class SimpleDataPreprocessor:
    def __init__(self, data_path='data/road_accident_dataset.csv'):
        self.data_path = data_path
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.target_column = 'Accident Severity'
        
    def load_data(self):
        """Load the accident dataset"""
        try:
            df = pd.read_csv(self.data_path)
            print(f"✅ Data loaded successfully! Shape: {df.shape}")
            
            # Select only the specified columns
            selected_columns = [
                'Country', 'Year', 'Month', 'Day of Week', 'Time of Day',
                'Urban/Rural', 'Road Type', 'Weather Conditions', 
                'Visibility Level', 'Number of Vehicles Involved', 'Accident Severity'
            ]
            
            # Check which columns exist in the dataset
            available_columns = [col for col in selected_columns if col in df.columns]
            missing_columns = [col for col in selected_columns if col not in df.columns]
            
            if missing_columns:
                print(f"⚠️  Missing columns: {missing_columns}")
            
            df = df[available_columns]
            print(f"📊 Using columns: {list(df.columns)}")
            return df
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def clean_data(self, df):
        """Clean and prepare the data"""
        print("🧹 Cleaning data...")
        
        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        print(f"   Removed {initial_rows - len(df)} duplicate rows")
        
        # Handle missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            print("   Missing values found:")
            for col, count in missing_counts[missing_counts > 0].items():
                print(f"     {col}: {count}")
            
            # Fill missing values based on column type
            for col in df.columns:
                if col == 'Accident Severity':
                    continue  # Handle target separately
                elif df[col].dtype == 'object':
                    df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
                else:
                    df[col] = df[col].fillna(df[col].median())
        
        # Convert severity to numeric categories
        severity_mapping = {
            'Minor': 0,
            'Moderate': 1, 
            'Major': 2,
            'Fatal': 3
        }
        df['Accident Severity'] = df['Accident Severity'].map(severity_mapping)
        
        # Remove rows with NaN in target variable
        df = df.dropna(subset=['Accident Severity'])
        print(f"   Final shape after cleaning: {df.shape}")
        
        print(f"✅ Data cleaning completed! Final shape: {df.shape}")
        return df
    
    def encode_categorical_features(self, df):
        """Encode categorical features"""
        print("🔤 Encoding categorical features...")
        
        categorical_columns = [
            'Country', 'Month', 'Day of Week', 'Time of Day', 'Urban/Rural',
            'Road Type', 'Weather Conditions'
        ]
        
        for col in categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_Encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                print(f"   Encoded: {col}")
        
        print("✅ Categorical encoding completed!")
        return df
    
    def select_features(self, df):
        """Select features for model training"""
        print("🎯 Selecting features...")
        
        # Define feature columns
        self.feature_columns = [
            # Original numeric features
            'Year', 'Visibility Level', 'Number of Vehicles Involved',
            
            # Encoded categorical features
            'Country_Encoded', 'Month_Encoded', 'Day of Week_Encoded', 'Time of Day_Encoded',
            'Urban/Rural_Encoded', 'Road Type_Encoded', 'Weather Conditions_Encoded'
        ]
        
        # Filter to only include columns that exist in the dataset
        available_features = [col for col in self.feature_columns if col in df.columns]
        missing_features = [col for col in self.feature_columns if col not in df.columns]
        
        if missing_features:
            print(f"⚠️  Missing features: {missing_features}")
        
        self.feature_columns = available_features
        print(f"✅ Selected {len(self.feature_columns)} features")
        return df
    
    def scale_features(self, df):
        """Scale numerical features"""
        print("📏 Scaling features...")
        
        # Select only numeric features for scaling
        numeric_features = ['Year', 'Visibility Level', 'Number of Vehicles Involved']
        
        available_numeric = [col for col in numeric_features if col in df.columns]
        
        if available_numeric:
            df[available_numeric] = self.scaler.fit_transform(df[available_numeric])
            print(f"   Scaled {len(available_numeric)} numeric features")
        
        print("✅ Feature scaling completed!")
        return df
    
    def prepare_data(self):
        """Complete data preprocessing pipeline"""
        print("🚀 Starting simple data preprocessing pipeline...")
        
        # Load data
        df = self.load_data()
        if df is None:
            return None, None, None, None
        
        # Clean data
        df = self.clean_data(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Select features
        df = self.select_features(df)
        
        # Scale features
        df = self.scale_features(df)
        
        # Prepare final dataset
        X = df[self.feature_columns]
        y = df[self.target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"✅ Data preprocessing completed!")
        print(f"   Training set: {X_train.shape}")
        print(f"   Test set: {X_test.shape}")
        print(f"   Features: {len(self.feature_columns)}")
        
        return X_train, X_test, y_train, y_test
    
    def get_feature_names(self):
        """Get list of feature names"""
        return self.feature_columns
    
    def save_preprocessing_info(self, output_dir='models'):
        """Save preprocessing information for later use"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save feature names
        feature_info = {
            'feature_columns': self.feature_columns,
            'target_column': self.target_column
        }
        
        import json
        with open(f'{output_dir}/simple_feature_info.json', 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        # Save label encoders
        if self.label_encoders:
            import joblib
            joblib.dump(self.label_encoders, f'{output_dir}/simple_label_encoders.pkl')
            print(f"✅ Label encoders saved to {output_dir}/simple_label_encoders.pkl")
        
        # Save scaler
        if self.scaler:
            import joblib
            joblib.dump(self.scaler, f'{output_dir}/simple_scaler.pkl')
            print(f"✅ Scaler saved to {output_dir}/simple_scaler.pkl")
        
        print(f"✅ Preprocessing info saved to {output_dir}/simple_feature_info.json")

if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = SimpleDataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data()
    
    if X_train is not None:
        preprocessor.save_preprocessing_info()
        print("🎉 Simple data preprocessing test completed successfully!")
    else:
        print("❌ Simple data preprocessing failed!")
