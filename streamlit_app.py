
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px

# Page config
st.set_page_config(
    page_title="Network Anomaly Detection System",
    page_icon="🛡️",
    layout="wide"
)

# Title
st.title("🛡️ Network Anomaly Detection System")
st.markdown("**DDoS Attack Detection using Machine Learning**")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Model Selection")
model_choice = st.sidebar.radio(
    "Choose Detection Model:",
    ["Autoencoder (Best)", "Isolation Forest v2"]
)

st.sidebar.markdown("---")
st.sidebar.header("📊 Project Info")
st.sidebar.info("""
**Dataset:** CIC-IDS2017  
**Samples:** 225,745  
**Features:** 78  
**Classes:** BENIGN, DDoS  

**Best Model:** Autoencoder  
- F1-Score: 75.11%
- ROC-AUC: 84.57%
- Precision: 94.24%
""")

# Load models
@st.cache_resource
def load_models():
    # Load Autoencoder
    autoencoder = tf.keras.models.load_model(
        r'C:\Users\spraj\anomaly_detection\models\autoencoder_model.keras',
        compile=False
    )
    with open(r'C:\Users\spraj\anomaly_detection\models\autoencoder_threshold.pkl', 'rb') as f:
        auto_threshold = pickle.load(f)
    
    # Load Isolation Forest
    with open(r'C:\Users\spraj\anomaly_detection\models\isolation_forest_v2.pkl', 'rb') as f:
        iso_forest = pickle.load(f)
    with open(r'C:\Users\spraj\anomaly_detection\models\iso_threshold_v2.pkl', 'rb') as f:
        iso_threshold = pickle.load(f)
    
    # Load scaler
    with open(r'C:\Users\spraj\anomaly_detection\processed_data\scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    return autoencoder, auto_threshold, iso_forest, iso_threshold, scaler

autoencoder, auto_threshold, iso_forest, iso_threshold, scaler = load_models()

# Load test data
@st.cache_data
def load_test_data():
    with open(r'C:\Users\spraj\anomaly_detection\processed_data\X_test_scaled.pkl', 'rb') as f:
        X_test = pickle.load(f)
    with open(r'C:\Users\spraj\anomaly_detection\processed_data\y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)
    return X_test, y_test

X_test, y_test = load_test_data()

# Main content
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Test Samples", f"{len(X_test):,}")
with col2:
    st.metric("BENIGN Traffic", f"{(y_test == 0).sum():,}")
with col3:
    st.metric("DDoS Attacks", f"{(y_test == 1).sum():,}")

st.markdown("---")

# Prediction section
st.header("🔍 Live Detection Demo")

sample_size = st.slider("Number of samples to analyze:", 100, 5000, 1000, step=100)

if st.button("🚀 Run Detection", type="primary"):
    with st.spinner("Analyzing network traffic..."):
        # Get sample
        X_sample = X_test.iloc[:sample_size].values if hasattr(X_test, 'iloc') else X_test[:sample_size]
        y_sample = y_test[:sample_size]
        
        if model_choice == "Autoencoder (Best)":
            # Autoencoder prediction
            X_reconstructed = autoencoder.predict(X_sample, verbose=0)
            reconstruction_error = np.mean(np.square(X_sample - X_reconstructed), axis=1)
            predictions = (reconstruction_error > auto_threshold).astype(int)
            scores = reconstruction_error
            
        else:
            # Isolation Forest prediction
            scores = iso_forest.decision_function(X_sample)
            predictions = (scores < iso_threshold).astype(int)
            scores = -scores
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        accuracy = accuracy_score(y_sample, predictions)
        precision = precision_score(y_sample, predictions, zero_division=0)
        recall = recall_score(y_sample, predictions, zero_division=0)
        f1 = f1_score(y_sample, predictions, zero_division=0)
        
        # Display results
        st.success("✅ Detection Complete!")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{accuracy*100:.2f}%")
        col2.metric("Precision", f"{precision*100:.2f}%")
        col3.metric("Recall", f"{recall*100:.2f}%")
        col4.metric("F1-Score", f"{f1*100:.2f}%")
        
        # Prediction breakdown
        st.markdown("### 📈 Detection Results")
        col1, col2 = st.columns(2)
        
        with col1:
            detected_benign = (predictions == 0).sum()
            detected_ddos = (predictions == 1).sum()
            
            fig = go.Figure(data=[go.Pie(
                labels=['BENIGN', 'DDoS'],
                values=[detected_benign, detected_ddos],
                marker_colors=['#2ecc71', '#e74c3c'],
                hole=0.4
            )])
            fig.update_layout(title="Predicted Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            actual_benign = (y_sample == 0).sum()
            actual_ddos = (y_sample == 1).sum()
            
            fig = go.Figure(data=[go.Pie(
                labels=['BENIGN', 'DDoS'],
                values=[actual_benign, actual_ddos],
                marker_colors=['#3498db', '#f39c12'],
                hole=0.4
            )])
            fig.update_layout(title="Actual Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        # Anomaly score distribution
        st.markdown("### 📊 Anomaly Score Distribution")
        
        df_viz = pd.DataFrame({
            'Anomaly Score': scores,
            'Actual Label': ['BENIGN' if y == 0 else 'DDoS' for y in y_sample],
            'Predicted': ['BENIGN' if p == 0 else 'DDoS' for p in predictions]
        })
        
        fig = px.histogram(
            df_viz, 
            x='Anomaly Score', 
            color='Actual Label',
            nbins=50,
            color_discrete_map={'BENIGN': '#2ecc71', 'DDoS': '#e74c3c'},
            title="Anomaly Score by Actual Label"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Top anomalies table
        st.markdown("### 🚨 Top 10 Detected Anomalies")
        top_indices = np.argsort(scores)[-10:][::-1]
        
        top_anomalies = pd.DataFrame({
            'Sample ID': top_indices,
            'Anomaly Score': scores[top_indices],
            'Actual Label': ['DDoS' if y_sample[i] == 1 else 'BENIGN' for i in top_indices],
            'Prediction': ['DDoS' if predictions[i] == 1 else 'BENIGN' for i in top_indices],
            'Status': ['✅ Correct' if predictions[i] == y_sample[i] else '❌ Incorrect' for i in top_indices]
        })
        
        st.dataframe(top_anomalies, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>Network Anomaly Detection System</strong></p>
    <p>Developed using TensorFlow, Scikit-learn, and Streamlit</p>
</div>
""", unsafe_allow_html=True)
