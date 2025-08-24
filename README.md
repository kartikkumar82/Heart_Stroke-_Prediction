# 🫀 Heart Stroke Disease Prediction

This project predicts the likelihood of heart stroke disease using Machine Learning algorithms.
It is deployed with Streamlit to provide an interactive web app for users to input their health details and get predictions instantly.

 📌 Features

Cleaned and preprocessed medical dataset.

Built using Logistic Regression / Random Forest / other ML models.

Interactive Streamlit web app for user input and prediction.

Provides real-time prediction of heart stroke risk.

Easy to use and deploy.

🚀 Tech Stack

Python

Pandas, NumPy, Scikit-learn

Streamlit

Joblib (for saving/loading ML model)

📊 Dataset

The dataset used for training and testing the model is from:
Kaggle - Stroke Prediction Dataset

🛠️ Installation & Setup

Clone the repository

git clone https://github.com/your-username/heart-stroke-prediction.git
cd heart-stroke-prediction


Create virtual environment (optional but recommended)

python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows


Install dependencies

pip install -r requirements.txt


Run the Streamlit app

streamlit run app.py

📷 Screenshots
🔹 Input Form

(Add screenshot here of your Streamlit form)

🔹 Prediction Result

(Add screenshot here of the prediction output)

📂 Project Structure
heart-stroke-prediction/
│── app.py                # Streamlit app
│── LR_heart.pkl          # Trained ML model
│── requirements.txt      # Dependencies
│── heart.csv             # Dataset (if included)
│── README.md             # Documentation

📈 Model Training

Data preprocessing (handling missing values, encoding, scaling).

Splitting data into train/test sets.

Training using Logistic Regression / Random Forest.

Model evaluation with accuracy, confusion matrix, precision, recall.

Best model saved as LR_heart.pkl using joblib.

🌐 Deployment

Deployed locally using Streamlit.
You can also deploy it on:

Streamlit Cloud

 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you’d like to change.

