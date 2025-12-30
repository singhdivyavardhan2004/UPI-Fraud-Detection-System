# UPI Fraud Detection System

## Overview

This project presents a complete end-to-end machine learning pipeline for detecting fraudulent UPI transactions.  
It simulates a real-world fraud detection system by covering data preprocessing, feature engineering, model training, evaluation, and deployment.

The system is designed to be interpretable, scalable, and suitable for production-level environments.

---

## Features

- End-to-end fraud detection pipeline  
- Supervised machine learning models  
- Model evaluation with standard metrics  
- Flask-based REST API for real-time prediction  
- Clean and modular project structure  

---


---

## Dataset Description

The dataset simulates real-world UPI transactions and includes:

- Transaction amount  
- Sender and receiver attributes  
- Transaction category  
- Temporal and behavioral features  

Target variable:
- `0` → Legitimate transaction  
- `1` → Fraudulent transaction  

---

## Models Used

### Logistic Regression
- Baseline classification model  
- Interpretable and fast  
- Used for benchmarking  

### Random Forest Classifier
- Captures nonlinear relationships  
- Robust to noise and feature interactions  
- Provides better fraud detection accuracy  

---

## Model Evaluation

The models are evaluated using the following metrics:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC-AUC  

Evaluation results and visualizations are stored in the `results/` directory.

---
## Running the Project

### Step 1: Install Dependencies

~~~bash
pip install -r requirements.txt
~~~

### Step 2: Train the Model

~~~bash
UPI_Fraud_Detection_model.ipynb
~~~

### Step 3: Start the API

~~~bash
python app.py
~~~
## Conclusion

This project demonstrates a complete and practical implementation of a UPI fraud detection system using machine learning.  
It covers the full lifecycle of a real-world ML solution — from data preprocessing and model training to evaluation and deployment.

The system effectively identifies fraudulent transactions using supervised learning techniques and provides a scalable foundation for further enhancements such as real-time streaming, explainability, and advanced models.

---

## Future Enhancements

- Integration of advanced models such as XGBoost or LightGBM  
- Explainability using SHAP or LIME  
- Real-time transaction streaming using Kafka  
- Database integration for transaction storage  
- Cloud deployment with Docker and CI/CD pipelines  

---

## Author

**Divyavardhan Singh**  
Machine Learning | Data Science | Applied AI  

---

## License

This project is intended for educational and research purposes.  
For commercial use, please contact the author.
