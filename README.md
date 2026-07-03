# 🏠 House Price Prediction MLOps Pipeline

An end-to-end Machine Learning project that predicts house prices using regression models. This project includes data ingestion, preprocessing, feature engineering, feature selection, model training, and a Flask web application for real-time predictions.

---

## 🚀 Features

- End-to-End Machine Learning Pipeline
- Data Ingestion
- Data Transformation
- Feature Engineering
- Random Forest Feature Selection
- Hyperparameter Tuning using GridSearchCV
- Model Comparison
- CatBoost Regression Model
- Flask Web Application
- Prediction Pipeline using Saved Artifacts

---

## 🛠️ Tech Stack

- Python
- Scikit-learn
- CatBoost
- XGBoost
- Pandas
- NumPy
- Flask
- Dill

---

## 📂 Project Structure

```
mlproject/
│
├── artifacts/
├── notebook/
├── src/
│   ├── components/
│   ├── pipeline/
│   ├── logger.py
│   ├── exception.py
│   └── utils.py
│
├── templates/
├── app.py
├── requirements.txt
├── setup.py
└── README.md
```

---

## 📊 Machine Learning Pipeline

```
Raw Data
     │
     ▼
Data Ingestion
     │
     ▼
Data Transformation
     │
     ▼
Feature Engineering
     │
     ▼
Random Forest Feature Selection
     │
     ▼
Model Training
     │
     ▼
Best Model Selection
     │
     ▼
Flask Prediction Pipeline
```

---

## 📈 Model Performance

| Model | R² Score |
|--------|----------|
| Random Forest | ~0.89 |
| Decision Tree | ~0.80 |
| Gradient Boosting | ~0.90 |
| Linear Regression | ~0.80 |
| XGBoost | ~0.89 |
| CatBoost | **~0.90** |
| AdaBoost | ~0.84 |

Best Performing Model: **CatBoost Regressor**

---

## 🌐 Web Application

The Flask web application allows users to enter important house features and predict the estimated house price in real time.

---

## 🔮 Future Improvements

- MLflow Experiment Tracking
- Docker Containerization
- AWS EC2 Deployment
- GitHub Actions CI/CD

---

## 👨‍💻 Author

**Jahed**