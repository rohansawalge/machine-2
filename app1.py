import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import tensorflow as tf

# Load pre-trained model (modify the path accordingly)
@st.cache_resource
def load_autoencoder():
    model = tf.keras.models.load_model("autoencoder_model.h5")  # Change to your saved model path
    return model

autoencoder = load_autoencoder()

# Streamlit UI
st.title("🔍 CBN Landis Machine - AI-Based Failure Prediction Dashboard")
st.markdown("Upload sensor data to detect anomalies and potential failures.")

# File upload
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### 📊 Uploaded Data Sample:", df.head())

    # Preprocess data
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)

    # Get reconstruction error
    reconstructions = autoencoder.predict(df_scaled)
    mse = np.mean(np.square(df_scaled - reconstructions), axis=1)

    # Set threshold (adjust based on training analysis)
    threshold = np.percentile(mse, 95)
    anomalies = mse > threshold

    df["Anomaly"] = anomalies
    df["Reconstruction Error"] = mse

    # Display results
    st.write("### 📌 Anomaly Detection Results:")
    st.dataframe(df)

    # Plot results
    st.write("### 📈 Reconstruction Error Distribution:")
    fig, ax = plt.subplots()
    ax.hist(mse, bins=50, color='blue', alpha=0.7)
    ax.axvline(threshold, color='red', linestyle='dashed', linewidth=2, label="Anomaly Threshold")
    ax.set_xlabel("Reconstruction Error")
    ax.set_ylabel("Frequency")
    ax.set_title("Autoencoder Error Distribution")
    ax.legend()
    st.pyplot(fig)

    # Show anomalies only
    st.write("### ❌ Detected Anomalies:")
    st.dataframe(df[df["Anomaly"] == True])
