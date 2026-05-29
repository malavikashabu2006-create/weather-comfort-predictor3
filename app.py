import streamlit as st
import pickle
import numpy as np
import os

# 1. PAGE SETUP & STYLING
st.set_page_config(page_title="Weather Comfort App", page_icon="🌡️", layout="centered")

st.markdown("<h1 style='text-align: center; color: #1F4E78;'>🌡️ Outdoor Comfort Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #566573;'>A clean, simple app to see how weather affects outdoor comfort levels using your trained AI models.</p>", unsafe_allow_html=True)
st.markdown("---")

# 2. LOAD TRAINED ASSETS SAFELY WITH LIVE TROUBLESHOOTER
@st.cache_resource
def load_ml_assets():
    with open('linear_regression_model.pkl', 'rb') as f:
        lr = pickle.load(f)
    with open('knn_regression_model.pkl', 'rb') as f:
        knn = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        sc = pickle.load(f)
    return lr, knn, sc

try:
    linear_model, knn_model, data_scaler = load_ml_assets()
    st.sidebar.success("⚡ Models & Scaler Loaded Successfully!")
except FileNotFoundError:
    st.error("❌ Error: Missing model files in the directory!")
    st.markdown("### 🔍 Live Troubleshooter Dashboard")
    st.write("The app cannot find one of your required files. Let's see what files are actually inside your GitHub repository right now:")
    current_files = os.listdir('.')
    st.code(current_files)
    st.stop()

# 3. SIDEBAR CONFIGURATION
st.sidebar.header("⚙️ Model Settings")
selected_model = st.sidebar.radio("Choose Model Architecture", ["Linear Regression", "KNN Regressor"])

# 4. USER INPUT CONTROLS
st.subheader("🌡️ Core Metric Input")
temperature = st.slider("Select Temperature (°C)", min_value=10.0, max_value=45.0, value=25.0, step=0.5)

# Hidden advanced controls to keep UI clean
with st.expander("🛠️ Advanced Weather Settings (Pre-filled with Safe Defaults)"):
    humidity = st.slider("Relative Humidity (%)", min_value=30, max_value=100, value=70)
    wind_speed = st.slider("Wind Speed (km/h)", min_value=0.0, max_value=50.0, value=15.0)
    air_quality = st.slider("Air Quality Index (AQI)", min_value=0, max_value=300, value=100)
    rainfall = st.slider("Rainfall (mm)", min_value=0.0, max_value=50.0, value=0.0)
    uv_index = st.slider("UV Index", min_value=0.0, max_value=12.0, value=5.0)
    visibility = st.slider("Visibility (km)", min_value=1.0, max_value=15.0, value=8.0)
    
    weather_type = st.selectbox("Weather Condition", ["Sunny", "Cloudy", "Rainy"])
    comfort_feeling = st.selectbox("General Feeling Context", ["Comfortable", "Cold", "Hot", "Humid"])
    season = st.selectbox("Current Season", ["Summer", "Winter", "Monsoon", "Spring"])

# 5. EXACT ALPHABETICAL LABEL ENCODING MAPS (Matching Sklearn behavior)
weather_map = {"Cloudy": 0, "Rainy": 1, "Sunny": 2}
comfort_map = {"Cold": 0, "Comfortable": 1, "Hot": 2, "Humid": 3}
season_map = {"Monsoon": 0, "Spring": 1, "Summer": 2, "Winter": 3}

weather_encoded = weather_map[weather_type]
comfort_encoded = comfort_map[comfort_feeling]
season_encoded = season_map[season]

# 6. INFERENCE PREDICTION LOGIC WITH AUTODETECT FEATURE ORDER
st.markdown("---")

# Exact alignment matching training columns: Weather_Type -> ComfortLevel -> Season
full_features = [
    temperature, humidity, wind_speed, air_quality, 
    rainfall, uv_index, visibility, 
    weather_encoded, comfort_encoded, season_encoded
]

numerical_features = [
    temperature, humidity, wind_speed, air_quality, 
    rainfall, uv_index, visibility
]

expected_features = data_scaler.n_features_in_
model_to_use = linear_model if selected_model == "Linear Regression" else knn_model

# Handle column transformations based on scaler constraints
if expected_features == 10:
    input_vector = np.array([full_features])
    scaled_input = data_scaler.transform(input_vector)
    predicted_score = model_to_use.predict(scaled_input)[0]
elif expected_features == 7:
    input_vector_num = np.array([numerical_features])
    scaled_numerical = data_scaler.transform(input_vector_num)
    scaled_input = np.hstack([scaled_numerical, [[weather_encoded, comfort_encoded, season_encoded]]])
    predicted_score = model_to_use.predict(scaled_input)[0]
else:
    st.error(f"⚠️ Unexpected Scaler configuration: expected {expected_features} features.")
    st.stop()

# Post-processing prediction constraints
final_score = max(0.0, min(10.0, predicted_score))

if final_score >= 8.0:
    status_label = "🌟 Excellent / Highly Comfortable"
    status_color = "#1D8348"
elif final_score >= 5.0:
    status_label = "🙂 Moderate / Manageable Comfort"
    status_color = "#D4AC0D"
else:
    status_label = "🥵 Poor / Uncomfortable Conditions"
    status_color = "#922B21"

# DISPLAY RESULTS CARD
st.markdown(f"""
<div style='background-color: #F4F6F7; padding: 20px; border-radius: 10px; border-left: 6px solid {status_color};'>
    <h3 style='color: #1F4E78; margin-top: 0;'>🎯 Live Predictive Analysis:</h3>
    <p style='font-size: 18px;'><b>Predicted Comfort Score:</b> {final_score:.2f} / 10.0</p>
    <p style='font-size: 18px;'><b>Evaluated Comfort Level:</b> <span style='color: {status_color}; font-weight: bold;'>{status_label}</span></p>
</div>
""", unsafe_allow_html=True)