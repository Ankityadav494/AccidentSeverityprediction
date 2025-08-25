#!/usr/bin/env python3
"""
Simple Model Training for Road Accident Severity Prediction
Uses only the 10 specified columns
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from simple_data_preprocessing import SimpleDataPreprocessor

class SimpleModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.feature_importance = None
        
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train multiple models and evaluate them"""
        print("🚀 Starting simple model training...")
        
        # Define models to train
        models = {
            'decision_tree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n🔧 Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Store model
            self.models[name] = model
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        self.best_model = results[best_model_name]['model']
        
        print(f"\n🏆 Best model: {best_model_name} (Accuracy: {results[best_model_name]['accuracy']:.4f})")
        
        # Generate detailed reports
        self._generate_reports(results, y_test)
        
        # Feature importance analysis
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = self.best_model.feature_importances_
            self._plot_feature_importance(X_train.columns)
        
        return results
    
    def _generate_reports(self, results, y_test):
        """Generate detailed classification reports"""
        print("\n📊 Generating detailed reports...")
        
        for name, result in results.items():
            print(f"\n--- {name.upper()} ---")
            print("Classification Report:")
            # Get unique classes in the predictions
            unique_classes = sorted(set(result['predictions']))
            target_names = ['Minor', 'Moderate', 'Major', 'Fatal']
            # Only use target names for classes that are actually present
            available_target_names = [target_names[int(i)] for i in unique_classes if int(i) < len(target_names)]
            print(classification_report(y_test, result['predictions'], 
                                      target_names=available_target_names))
    
    def _plot_feature_importance(self, feature_names):
        """Plot feature importance"""
        if self.feature_importance is None:
            return
        
        print("📈 Plotting feature importance...")
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
        
        # Plot all features
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance for Simple Model')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        # Save plot
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/simple_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Feature importance plot saved to results/simple_feature_importance.png")
    
    def plot_confusion_matrices(self, results, y_test):
        """Plot confusion matrices for all models"""
        print("📊 Plotting confusion matrices...")
        
        n_models = len(results)
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        for idx, (name, result) in enumerate(results.items()):
            cm = confusion_matrix(y_test, result['predictions'])
            
            # Get unique classes for labels
            unique_classes = sorted(set(y_test) | set(result['predictions']))
            target_names = ['Minor', 'Moderate', 'Major', 'Fatal']
            available_target_names = [target_names[int(i)] for i in unique_classes if int(i) < len(target_names)]
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=available_target_names,
                       yticklabels=available_target_names,
                       ax=axes[idx])
            axes[idx].set_title(f'{name.replace("_", " ").title()}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/simple_confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Confusion matrices saved to results/simple_confusion_matrices.png")
    
    def save_models(self, models_dir='models'):
        """Save trained models"""
        print(f"💾 Saving models to {models_dir}...")
        
        os.makedirs(models_dir, exist_ok=True)
        
        for name, model in self.models.items():
            model_path = os.path.join(models_dir, f'simple_{name}_model.pkl')
            joblib.dump(model, model_path)
            print(f"   Saved {name} to {model_path}")
        
        # Save best model separately
        if self.best_model:
            best_model_path = os.path.join(models_dir, 'simple_best_model.pkl')
            joblib.dump(self.best_model, best_model_path)
            print(f"   Saved best model to {best_model_path}")
        
        print("✅ All models saved successfully!")

def train_and_evaluate_simple_models():
    """Main function to train and evaluate simple models"""
    print("🎯 Starting simple model training pipeline...")
    
    # Initialize preprocessor and trainer
    preprocessor = SimpleDataPreprocessor()
    trainer = SimpleModelTrainer()
    
    # Prepare data
    X_train, X_test, y_train, y_test = preprocessor.prepare_data()
    
    if X_train is None:
        print("❌ Data preparation failed!")
        return None
    
    # Save preprocessing info
    preprocessor.save_preprocessing_info()
    
    # Train models
    results = trainer.train_models(X_train, y_train, X_test, y_test)
    
    # Generate visualizations
    trainer.plot_confusion_matrices(results, y_test)
    
    # Save models
    trainer.save_models()
    
    print("\n🎉 Simple model training pipeline completed successfully!")
    return trainer

if __name__ == "__main__":
    trainer = train_and_evaluate_simple_models()
    if trainer:
        print("✅ All simple models trained and saved!")
    else:
        print("❌ Simple model training failed!")
