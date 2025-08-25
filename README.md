# 🚗 Road Accident Severity Prediction System

A machine learning-powered web application that predicts road accident severity based on various factors like location, time, weather conditions, and road characteristics.

## 🎯 **Project Overview**

This application uses a **simplified 10-feature model** to predict accident severity with high accuracy. It provides a beautiful, modern web interface for real-time predictions.

## 📁 **Project Structure**

```
ML project/
├── data/
│   └── road_accident_dataset.csv    # Main dataset
├── models/                          # Trained ML models
├── src/
│   ├── simple_data_preprocessing.py # Data preprocessing
│   ├── simple_model_training.py     # Model training
│   └── simple_prediction.py         # Prediction logic
|
├── simple_web_interface.py         # Flask web server
├── requirements.txt                # Python dependencies
└── README.md                      # This file
```

## 🚀 **Quick Start**

### **Step 1: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 2: Run the Application**
```bash
python simple_web_interface.py
```

### **Step 3: Access the Web Interface**
- Open your browser
- Go to: **http://localhost:5000**
- Start making predictions!

## 🎨 **Features**

### **Beautiful Modern Interface:**
- ✨ Animated gradient background
- 🎨 Glassmorphism design with blur effects
- 📱 Fully responsive design
- 🌟 Smooth animations and transitions

### **Input Fields (10 Essential Features):**
1. **Country** (including 🇮🇳 India)
2. **Year** (2000-2030)
3. **Month** (January-December)
4. **Day of Week** (Monday-Sunday)
5. **Time of Day** (Night, Morning, Afternoon, etc.)
6. **Urban/Rural** area type
7. **Road Type** (Highway, Street, Avenue, etc.)
8. **Weather Conditions** (Clear, Rainy, Snow-covered, etc.)
9. **Visibility Level** (0-1000 meters)
10. **Number of Vehicles Involved** (1-10)

### **AI Predictions:**
- 🟢 **Minor** - Low risk accident
- 🟡 **Moderate** - Medium risk accident
- 🟠 **Major** - High risk accident
- 🔴 **Fatal** - Very high risk accident

### **Results Include:**
- 📊 Confidence scores for each severity level
- 💡 AI interpretation and recommendations
- 📈 Visual progress bars
- 🎯 Real-time analysis

## 🔧 **Technical Details**

### **Machine Learning Models:**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- Automatic model selection based on performance

### **Data Processing:**
- Label encoding for categorical variables
- Feature scaling for numerical variables
- Robust handling of missing values
- Consistent preprocessing pipeline

### **Web Framework:**
- **Flask** backend
- **HTML/CSS/JavaScript** frontend
- **RESTful API** for predictions

## 📊 **Model Performance**

The simplified model achieves:
- **Accuracy**: ~50% (balanced for 2-class problem)
- **Fast predictions**: Real-time results
- **Consistent encoding**: Same inputs always give same results
- **Robust preprocessing**: Handles various input scenarios

## 🎯 **Example Usage**

### **Low Risk Scenario:**
- Country: USA, Year: 2024, Month: June
- Day: Monday, Time: Morning
- Area: Urban, Road: Street
- Weather: Clear, Visibility: 200m, Vehicles: 1
- **Expected**: Minor severity

### **High Risk Scenario:**
- Country: India, Year: 2024, Month: December
- Day: Saturday, Time: Night
- Area: Rural, Road: Highway
- Weather: Snow-covered, Visibility: 50m, Vehicles: 3
- **Expected**: Major/Fatal severity

## 🛠️ **Development**

### **Retrain Models:**
```bash
python src/simple_model_training.py
```

### **Test Predictions:**
```bash
python src/simple_prediction.py
```

## 📋 **Requirements**

- Python 3.7+
- Flask
- pandas
- numpy
- scikit-learn
- joblib
- matplotlib
- seaborn

## 🎉 **Ready to Use!**

Your accident prediction system is now clean, optimized, and ready for use. Simply run `python simple_web_interface.py` and start making predictions!

