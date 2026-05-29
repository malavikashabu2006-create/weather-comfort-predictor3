import streamlit as st
import pickle
import numpy as np
import os

# 1. PAGE SETUP & STYLING
st.set_page_config(page_title="Weather Comfort App", page_icon="🌡️", layout="centered")

st.markdown("<h1 style='text-align: center; color: #1F4E78;'>🌡️ Outdoor Comfort Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #566573;'>A clean, simple app to see how temperature affects outdoor comfort levels using your trained AI models.</p>", unsafe_allow_html=True)
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
    
    # This code reads and prints exactly what files are present on the server
    current_files = os.listdir('.')
    st.write("**Files currently found on your web server:**")
    st.code(current_files)
    
    st.markdown("""
    **💡 Look closely at the list in the box above to see what is wrong:**
    1. **Are the files hidden inside a folder?** If you see something like `['weather_app', 'README.md']`, it means your files are tucked inside a folder. You need to upload your files directly to the main repository page next to `README.md` instead of inside a subfolder.
    2. **Is there a double extension or typo?** Check if a file is accidentally named something like `scaler.pkl.txt`, `scaler (1).pkl`, or `linear_regression_model.pkl.txt`. It must match the names below perfectly.
    """)
    st.stop()

# 3. SIDEBAR CONFIGURATION
st.sidebar.header("⚙️ Model Settings")
selected_model = st.sidebar.radio("Choose Model Architecture", ["Linear Regression", "KNN Regressor"])

# 4. SIMPLE BASIC USER INPUT
st.subheader("🌡️ Core Metric Input")
temperature = st.slider("Select Temperature (°C)", min_value=10.0, max_value=45.0, value=25.0, step=0.5)

# Hide other metrics in an expander so the UI stays simple and uncluttered
with st.expander("🛠️ Advanced Weather Settings (Pre-filled with Safe Defaults)"):
    humidity = st.slider("Relative Humidity (%)", min_value=30, max_value=100, value=70)
    wind_speed = st.slider("Wind Speed (km/h)", min_value=0.0, max_value=50.0, value=15.0)
    air_quality = st.slider("Air Quality Index (AQI)", min_value=0, max_value=300, value=100)
    rainfall = st.slider("Rainfall (mm)", min_value=0.0, max_value=50.0, value=0.0)
    uv_index = st.slider("UV Index", min_value=0.0, max_value=12.0, value=5.0)
    visibility = st.slider("Visibility (km)", min_value=1.0, max_value=15.0, value=8.0)
    
    weather_type = st.selectbox("Weather Condition", ["Sunny", "Cloudy", "Rainy"])
    season = st.selectbox("Current Season", ["Winter", "Monsoon", "Summer", "Autumn"])
    comfort_feeling = st.selectbox("General Feeling Context", ["Cold", "Comfortable", "Hot", "Humid"])

# 5. CATEGORICAL ENCODING MAPS
weather_map = {"Cloudy": 0, "Rainy": 1, "Sunny": 2}
season_map = {"Winter": 0, "Monsoon": 1, "Summer": 2, "Autumn": 3}
comfort_map = {"Cold": 0, "Comfortable": 1, "Hot": 2, "Humid": 3}

weather_encoded = weather_map[weather_type]
season_encoded = season_map[season]
comfort_encoded = comfort_map[comfort_feeling]

# 6. INFERENCE PREDICTION LOGIC WITH AUTO-FEATURE DETECTION
st.markdown("---")

full_features = [
    temperature, humidity, wind_speed, air_quality, 
    rainfall, uv_index, visibility, 
    weather_encoded, season_encoded, comfort_encoded
]

numerical_features = [
    temperature, humidity, wind_speed, air_quality, 
    rainfall, uv_index, visibility
]

expected_features = data_scaler.n_features_in_
model_to_use = linear_model if selected_model == "Linear Regression" else knn_model

if expected_features == 10:
    input_vector = np.array([full_features])
    scaled_input = data_scaler.transform(input_vector)
    predicted_score = model_to_use.predict(scaled_input)[0]
elif expected_features == 7:
    input_vector_num = np.array([numerical_features])
    scaled_numerical = data_scaler.transform(input_vector_num)
    scaled_input = np.hstack([scaled_numerical, [[weather_encoded, season_encoded, comfort_encoded]]])
    predicted_score = model_to_use.predict(scaled_input)[0]

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

st.markdown(f"""
<div style='background-color: #F4F6F7; padding: 20px; border-radius: 10px; border-left: 6px solid {status_color};'>
    <h3 style='color: #1F4E78; margin-top: 0;'>🎯 Live Predictive Analysis:</h3>
    <p style='font-size: 18px;'><b>Predicted Comfort Score:</b> {final_score:.2f} / 10.0</p>
    <p style='font-size: 18px;'><b>Evaluated Comfort Level:</b> <span style='color: {status_color}; font-weight: bold;'>{status_label}</span></p>
</div>
""", unsafe_allow_html=True)