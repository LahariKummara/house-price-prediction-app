# 🏠 House Price Prediction Web App

A machine learning-based web application that predicts house prices using demographic and housing features. This app allows users to select from multiple trained regression models and get real-time price predictions via a simple web interface built using Flask.

---

## 📌 Project Overview

This project demonstrates an end-to-end machine learning pipeline:
- Data preprocessing and model training
- Evaluation of 13 regression algorithms
- Model serialization with Pickle
- Web deployment using Flask and HTML templates

---

## 🧠 ML Models Used

The following 13 regression models were trained and saved:
- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet
- Polynomial Regression
- Robust Regression (Huber Regressor)
- SGD Regressor
- ANN (MLP Regressor)
- Random Forest
- Support Vector Machine (SVR)
- K-Nearest Neighbors
- LightGBM
- XGBoost

Each model was evaluated using:
- **MAE** (Mean Absolute Error)
- **MSE** (Mean Squared Error)
- **R² Score**

Evaluation metrics are stored in `model_evaluation_results.csv`.

---

## 🛠️ Technologies Used

- **Python 3.x**
- **Pandas**
- **Scikit-learn**
- **XGBoost**
- **LightGBM**
- **Flask**
- **HTML (Jinja Templates)**
- **Pickle**

---

## 📂 Folder Structure

house-price-prediction-app/
│
├── House Price Prediction Model.py # Model training and saving script
├── House Price Prediction app.py # Flask web application
├── model_evaluation_results.csv # Model performance metrics
├── requirements.txt # List of required packages
├── README.md # Project documentation
│
├── models/ # Trained .pkl model files
│ ├── LinearRegression.pkl
│ ├── RidgeRegression.pkl
│ └── ... (total 13)
│
├── templates/ # Flask HTML templates
│ ├── index.html
│ └── results.html
│
└── screenshots/ (optional) # UI screenshots
├── home.png
└── result.png


## 🚀 How to Run the App Locally

1. **Clone the repository**

git clone https://github.com/LahariKummara/house-price-prediction-app.git
cd house-price-prediction-app

2. **Install required packages**

pip install -r requirements.txt

3. **Run the Flask app**

python "House Price Prediction app.py"

4. **Open your browser and go to**

http://127.0.0.1:5000/


✅ Acknowledgements
This project was built as part of my Data Science journey to demonstrate model training, evaluation, and deployment using Python and Flask.

















