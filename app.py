import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .highlight {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .positive {
        color: #2ecc71;
    }
    .negative {
        color: #e74c3c;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<h1 class="main-header">💳 Credit Card Fraud Detection System</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose a page", 
                               ["Home", "Data Overview", "Data Visualization", "Model Training", "Make Predictions"])

# Load data
@st.cache_data
def load_data():
    # For demo purposes, we'll create a sample dataset
    # In a real app, you would load from the actual CSV file
    np.random.seed(42)
    n_samples = 1000
    
    # Create synthetic data similar to the credit card fraud dataset
    data = {
        'Time': np.random.uniform(0, 172000, n_samples),
        'Amount': np.random.exponential(100, n_samples),
        'Class': np.random.choice([0, 1], n_samples, p=[0.98, 0.02])
    }
    
    # Add V1-V28 features (normally distributed)
    for i in range(1, 29):
        data[f'V{i}'] = np.random.normal(0, 1, n_samples)
    
    df = pd.DataFrame(data)
    return df

credit_card_data = load_data()

# Home page
if app_mode == "Home":
    st.markdown("""
    ## Welcome to the Credit Card Fraud Detection System
    
    This application helps detect fraudulent credit card transactions using machine learning.
    
    ### Features:
    - **Data Overview**: Explore the dataset and understand its structure
    - **Data Visualization**: Visualize distributions and relationships in the data
    - **Model Training**: Train a logistic regression model on the data
    - **Make Predictions**: Use the trained model to detect fraudulent transactions
    
    ### How to use:
    1. Start by exploring the dataset in the **Data Overview** section
    2. Visualize the data patterns in the **Data Visualization** section
    3. Train the model in the **Model Training** section
    4. Make predictions on new data in the **Make Predictions** section
    """)
    
    st.image("https://github.com/subhadipsinha722133/Credit-Card-Fraud-Dection/blob/main/ccfd-scaled.jpeg", 
             use_column_width=True, caption="Credit Card Fraud Detection")

# Data Overview page
elif app_mode == "Data Overview":
    st.markdown('<h2 class="sub-header">Dataset Overview</h2>', unsafe_allow_html=True)
    
    # Show dataset
    st.dataframe(credit_card_data.head(10))
    
    # Dataset info
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Dataset Shape")
        st.write(f"Number of rows: {credit_card_data.shape[0]}")
        st.write(f"Number of columns: {credit_card_data.shape[1]}")
        
        st.markdown("#### Column Information")
        st.write(credit_card_data.dtypes.astype(str))
    
    with col2:
        st.markdown("#### Class Distribution")
        class_counts = credit_card_data['Class'].value_counts()
        st.write(f"Legitimate transactions (Class 0): {class_counts[0]}")
        st.write(f"Fraudulent transactions (Class 1): {class_counts[1]}")
        
        fig = px.pie(values=class_counts.values, names=['Legitimate', 'Fraudulent'],
                     title='Transaction Class Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistical summary
    st.markdown("#### Statistical Summary")
    st.dataframe(credit_card_data.describe())

# Data Visualization page
elif app_mode == "Data Visualization":
    st.markdown('<h2 class="sub-header">Data Visualization</h2>', unsafe_allow_html=True)
    
    # Select visualization type
    viz_type = st.selectbox("Select Visualization Type", 
                           ["Transaction Amount Distribution", "Time vs Amount", 
                            "Feature Correlation", "Class Comparison"])
    
    if viz_type == "Transaction Amount Distribution":
        fig = px.histogram(credit_card_data, x='Amount', nbins=50, 
                          title='Distribution of Transaction Amounts')
        st.plotly_chart(fig, use_container_width=True)
        
        # Log scale option
        log_scale = st.checkbox("Log Scale")
        if log_scale:
            fig = px.histogram(credit_card_data, x='Amount', nbins=50, 
                              title='Distribution of Transaction Amounts (Log Scale)',
                              log_x=True)
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Time vs Amount":
        fig = px.scatter(credit_card_data, x='Time', y='Amount', color='Class',
                        title='Time vs Amount by Transaction Class')
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Feature Correlation":
        # Calculate correlation matrix for a subset of features
        features_to_correlate = ['Time', 'Amount', 'V1', 'V2', 'V3', 'V4', 'V5', 'Class']
        corr_matrix = credit_card_data[features_to_correlate].corr()
        
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                       title='Correlation Matrix of Selected Features')
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Class Comparison":
        selected_feature = st.selectbox("Select Feature to Compare", 
                                       ['Time', 'Amount', 'V1', 'V2', 'V3', 'V4', 'V5'])
        
        fig = px.box(credit_card_data, x='Class', y=selected_feature, 
                    title=f'{selected_feature} Distribution by Class')
        st.plotly_chart(fig, use_container_width=True)

# Model Training page
elif app_mode == "Model Training":
    st.markdown('<h2 class="sub-header">Model Training</h2>', unsafe_allow_html=True)
    
    # Create a balanced dataset (similar to your code)
    legit = credit_card_data[credit_card_data.Class == 0]
    fraud = credit_card_data[credit_card_data.Class == 1]
    
    # Check if we have enough samples
    if len(fraud) == 0:
        st.error("No fraudulent transactions found in the dataset!")
        st.stop()
    
    n_fraud = len(fraud)
    legit_sample = legit.sample(n=n_fraud, random_state=42)
    new_dataset = pd.concat([legit_sample, fraud], axis=0)
    
    # Prepare features and target
    X = new_dataset.drop(columns='Class', axis=1)
    Y = new_dataset['Class']
    
    # Split the data
    test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)
    random_state = st.number_input("Random State", 0, 100, 2)
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, stratify=Y, random_state=random_state
    )
    
    st.write(f"Training set size: {X_train.shape[0]}")
    st.write(f"Test set size: {X_test.shape[0]}")
    
    # Train model
    if st.button("Train Model"):
        with st.spinner("Training the model..."):
            model = LogisticRegression()
            model.fit(X_train, Y_train)
            
            # Make predictions
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            # Calculate accuracy
            train_accuracy = accuracy_score(Y_train, train_pred)
            test_accuracy = accuracy_score(Y_test, test_pred)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### Training Results")
                st.markdown(f"<p class='highlight'>Accuracy: <span class='positive'>{train_accuracy:.4f}</span></p>", 
                           unsafe_allow_html=True)
                
                # Confusion matrix for training
                cm_train = confusion_matrix(Y_train, train_pred)
                fig = px.imshow(cm_train, text_auto=True, 
                               labels=dict(x="Predicted", y="Actual", color="Count"),
                               x=['Legitimate', 'Fraudulent'], y=['Legitimate', 'Fraudulent'],
                               title="Training Confusion Matrix")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("##### Test Results")
                st.markdown(f"<p class='highlight'>Accuracy: <span class='positive'>{test_accuracy:.4f}</span></p>", 
                           unsafe_allow_html=True)
                
                # Confusion matrix for test
                cm_test = confusion_matrix(Y_test, test_pred)
                fig = px.imshow(cm_test, text_auto=True, 
                               labels=dict(x="Predicted", y="Actual", color="Count"),
                               x=['Legitimate', 'Fraudulent'], y=['Legitimate', 'Fraudulent'],
                               title="Test Confusion Matrix")
                st.plotly_chart(fig, use_container_width=True)
            
            # Classification report
            st.markdown("##### Classification Report")
            report = classification_report(Y_test, test_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
            
            # Save model to session state for prediction page
            st.session_state.model = model
            st.success("Model trained successfully and saved for predictions!")

# Make Predictions page
elif app_mode == "Make Predictions":
    st.markdown('<h2 class="sub-header">Make Predictions</h2>', unsafe_allow_html=True)
    
    if 'model' not in st.session_state:
        st.warning("Please train the model first in the 'Model Training' section.")
    else:
        st.info("Enter the transaction details to check for potential fraud.")
        
        # Create input form
        col1, col2 = st.columns(2)
        
        with col1:
            time = st.number_input("Time", min_value=0.0, max_value=200000.0, value=0.0)
            amount = st.number_input("Amount", min_value=0.0, max_value=5000.0, value=0.0)
            v1 = st.number_input("V1", value=0.0)
            v2 = st.number_input("V2", value=0.0)
            v3 = st.number_input("V3", value=0.0)
            v4 = st.number_input("V4", value=0.0)
        
        with col2:
            v5 = st.number_input("V5", value=0.0)
            v6 = st.number_input("V6", value=0.0)
            v7 = st.number_input("V7", value=0.0)
            v8 = st.number_input("V8", value=0.0)
            v9 = st.number_input("V9", value=0.0)
            v10 = st.number_input("V10", value=0.0)
        
        # Create feature array
        features = [time, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10]
        # Add remaining V features with default value 0
        features.extend([0.0] * 18)  # For V11 to V28
        features.append(amount)
        
        # Make prediction
        if st.button("Predict"):
            features_array = np.array(features).reshape(1, -1)
            prediction = st.session_state.model.predict(features_array)
            probability = st.session_state.model.predict_proba(features_array)
            
            if prediction[0] == 0:
                st.markdown(f"<h3 class='positive'>✅ Legitimate Transaction (Probability: {probability[0][0]:.4f})</h3>", 
                           unsafe_allow_html=True)
            else:
                st.markdown(f"<h3 class='negative'>🚨 Fraudulent Transaction (Probability: {probability[0][1]:.4f})</h3>", 
                           unsafe_allow_html=True)
            
            # Show probability chart
            fig = px.bar(x=['Legitimate', 'Fraudulent'], y=probability[0], 
                        labels={'x': 'Class', 'y': 'Probability'},
                        title='Prediction Probability')
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("### About")
st.markdown("""
This credit card fraud detection system uses machine learning to identify potentially fraudulent transactions.
The model is trained on historical transaction data and can flag suspicious activity in real-time.

**Note:** This is a demonstration application. In a real-world scenario, you would:
1. Use the actual credit card fraud dataset
2. Implement more sophisticated models
3. Add additional validation and security measures

""")
