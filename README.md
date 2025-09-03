## Credit Card Fraud Detection Web App
A Streamlit-based web application for detecting fraudulent credit card transactions using machine learning.

# Overview
This application provides an interactive interface for:

Exploring credit card transaction data

Visualizing patterns and relationships in the data

Training a logistic regression model for fraud detection

Making predictions on new transactions

# Live Demo

[Demo link](https://credit-card-fraud-dection-7e4ewybpeqvk2ozhecc2ht.streamlit.app/)

# Features
Data Overview: Explore dataset statistics and class distribution

Data Visualization: Interactive charts including histograms, scatter plots, and correlation matrices

Model Training: Train a logistic regression model with customizable parameters

Prediction Interface: Input transaction details to get real-time fraud predictions

Responsive Design: Clean, modern UI with custom styling

# Dataset
The application uses a synthetic dataset that mimics the structure of real credit card transaction data, containing:

Time: Number of seconds elapsed between this transaction and the first transaction

V1-V28: Principal components obtained from PCA transformation

Amount: Transaction amount

Class: 0 for legitimate, 1 for fraudulent transactions

# Installation
Clone the repository:

bash
git clone https://github.com/subhadipsinha722133/credit-card-fraud-detection.git
cd credit-card-fraud-detection
Install required packages:

bash
pip install streamlit pandas numpy matplotlib seaborn plotly scikit-learn
Usage
Run the application with:

bash
streamlit run app.py
The application will open in your default web browser at http://localhost:8501.

# How to Use
Data Overview: Start by exploring the dataset structure and statistics

Data Visualization: Examine patterns through interactive visualizations

Model Training: Train the logistic regression model with your preferred parameters

Make Predictions: Use the trained model to detect fraudulent transactions

# Model Details
The application uses a Logistic Regression model trained on a balanced dataset (equal number of legitimate and fraudulent transactions). Key metrics displayed:

Training and test accuracy

Confusion matrices

Classification report with precision, recall, and F1-score

# File Structure
text
credit-card-fraud-detection/  <br>
├── app.py                 # Main Streamlit application  <br>
├── README.md              # Project documentation  <br>
└── requirements.txt       # Python dependencies   <br>
# Dependencies
- streamlit
- pandas
- numpy
- matplotlib
- seaborn
- plotly
- scikit-learn

# Contributing
Fork the repository

Create a feature branch (git checkout -b feature/amazing-feature)

Commit your changes (git commit -m 'Add amazing feature')

Push to the branch (git push origin feature/amazing-feature)

Open a Pull Request

# Future Enhancements
Integration with real credit card transaction datasets

Additional machine learning models (Random Forest, XGBoost, Neural Networks)

Real-time data processing capabilities

Advanced feature engineering options

Model comparison and selection interface

# Disclaimer
This is a demonstration application. For production use:

Use actual transaction data with proper security measures

Implement additional validation and security protocols

Consider more sophisticated model architectures

Ensure compliance with financial regulations and data privacy laws

# License
This project is licensed under the MIT License - see the LICENSE file for details.

# Acknowledgments
Inspired by the Kaggle Credit Card Fraud Detection dataset


Built with Streamlit, an open-source app framework for Machine Learning and Data Science projects
