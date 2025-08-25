# 🚗 Road Accident Severity Prediction System

A machine learning-powered application that predicts road accident severity based on various factors like location, time, weather conditions, and road characteristics. Features both a Streamlit web app and a PowerPoint presentation generator.

## 🎯 **Project Overview**

This application uses a **simplified 10-feature model** to predict accident severity with high accuracy. It provides an interactive Streamlit interface for real-time predictions and can generate professional PowerPoint presentations for project demonstrations.

## 📁 **Project Structure**

```
ML project/
├── data/
│   └── road_accident_dataset.csv    # Main dataset
├── models/                          # Trained ML models
│   ├── simple_decision_tree_model.pkl
│   ├── simple_random_forest_model.pkl
│   ├── simple_best_model.pkl
│   ├── simple_label_encoders.pkl
│   ├── simple_scaler.pkl
│   └── simple_feature_info.json
├── src/
│   ├── simple_data_preprocessing.py # Data preprocessing
│   ├── simple_model_training.py     # Model training
│   └── simple_prediction.py         # Prediction logic
├── app.py                          # Streamlit web application
├── generate_presentation.py        # PowerPoint generator script
├── requirements.txt                # Python dependencies
└── README.md                      # This file
```

## 🚀 **Quick Start**

### **Step 1: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 2: Run the Streamlit App**
```bash
streamlit run app.py
```

### **Step 3: Access the Web Interface**
- Open your browser
- Go to: **http://localhost:8501**
- Start making predictions!

## 🎨 **Features**

### **Interactive Streamlit Interface:**
- ✨ Clean, modern design with icons
- 🎨 Grouped input sections for better UX
- 📱 Responsive layout with columns
- 🌟 Sample scenarios and quick reset functionality
- 🔄 Model selector (Decision Tree vs Random Forest)

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
- 📊 Confidence scores with visual bar charts
- 💡 AI interpretation and recommendations
- 📈 Feature importance visualization
- 🎯 Real-time analysis with model selection
- 🔍 Detailed probability breakdowns

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
- **Streamlit** frontend and backend
- **Python-based** with pandas and plotly
- **Interactive widgets** for user input
- **Real-time predictions** with caching

## 📊 **Model Performance**

The simplified model achieves:
- **Accuracy**: ~50% (balanced for 2-class problem)
- **Fast predictions**: Real-time results
- **Consistent encoding**: Same inputs always give same results
- **Robust preprocessing**: Handles various input scenarios

## 🎯 **Example Usage**

### **Using the Streamlit App:**
1. Fill in the scenario details using the form
2. Choose your preferred model (Random Forest recommended)
3. Click "Predict Severity" to get results
4. View confidence scores and feature importance
5. Use sample scenarios for quick testing

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

### **Generate PowerPoint Presentation:**
```bash
python generate_presentation.py
```
This creates `Road_Accident_Severity_Presentation.pptx` with project details.

### **Retrain Models:**
```bash
python src/simple_model_training.py
```

### **Test Predictions:**
```bash
python src/simple_prediction.py
```

### **Local Development:**
```bash
# Install in development mode
pip install -e .

# Run with custom port
streamlit run app.py --server.port 8502
```

## 📋 **Requirements**

- Python 3.7+
- Streamlit
- pandas
- numpy
- scikit-learn
- joblib
- matplotlib
- seaborn
- python-pptx (for presentation generation)

## 🚀 **Deployment**

### **Streamlit Community Cloud (Recommended):**
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Set main file to `app.py`
5. Deploy!

### **Alternative Platforms:**
- **Render**: For API-based deployment
- **Railway**: For containerized deployment
- **PythonAnywhere**: For Python-focused hosting

## 📊 **Presentation Features**

The included PowerPoint generator creates professional slides covering:
- Project overview and objectives
- Data and feature descriptions
- Model architecture and training
- Feature importance analysis
- Streamlit app demonstration
- Future improvements and next steps

Perfect for academic presentations, client demos, or project showcases!

## 🎉 **Ready to Use!**

Your accident prediction system is now modern, interactive, and ready for use! Run `streamlit run app.py` to start the web app, or use `python generate_presentation.py` to create presentation materials.

## 📞 **Support & Contributing**

Feel free to:
- 🐛 Report issues
- 💡 Suggest improvements
- 🔧 Submit pull requests
- ⭐ Star the repository

Happy predicting! 🚗✨

