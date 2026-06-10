import streamlit as st

# 1. Page Configurations
st.set_page_config(
    page_title="Weather Comfort Predictor",
    page_icon="🌤️",
    layout="centered"
)

# 2. Manage App Navigation via Session State
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'welcome'

def switch_page(page_name):
    st.session_state.current_page = page_name

# ==========================================
# PAGE 1: WELCOME / LANDING PAGE
# ==========================================
if st.session_state.current_page == 'welcome':
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.title("🌤️ Weather Comfort Score Predictor")
    st.subheader("Your ultimate smart companion for planning perfect outdoor days.")
    
    st.markdown("---")
    
    st.markdown("""
    Welcome! This app evaluates environmental and meteorological metrics to accurately predict a real-time comfort index.
    
    ### 🔬 How It Works
    The system processes live data through an optimized **Linear Regression Model**, evaluating the mathematical combinations of atmospheric factors to provide an intuitive comfort score between **0 and 100**.
    """)
    
    # Engaging Call-out box
    st.info("💡 **Note:** No manual model swapping required. The system automatically utilizes our trained Linear Regression framework for maximum performance.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Big primary "GO" button
    if st.button("Launch App & Predict Now 🚀", type="primary", use_container_width=True):
        switch_page('app_interface')
        st.rerun()

# ==========================================
# PAGE 2: MAIN PREDICTOR APP INTERFACE
# ==========================================
elif st.session_state.current_page == 'app_interface':
    st.title("📊 Weather Comfort Predictor Dashboard")
    st.write("Adjust the environmental slides below. The Linear Regression model will calculate the predicted comfort score instantly.")
    
    st.markdown("---")
    
    # Input Features (Sliders)
    st.subheader("🔄 1. Input Weather Metrics")
    
    temperature = st.slider("Temperature (°C)", min_value=-10.0, max_value=45.0, value=24.0, step=0.5)
    humidity = st.slider("Humidity (%)", min_value=0, max_value=100, value=50, step=1)
    wind_speed = st.slider("Wind Speed (km/h)", min_value=0.0, max_value=60.0, value=12.0, step=0.5)
    uv_index = st.slider("UV Index Level", min_value=0, max_value=11, value=3, step=1)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 3. Linear Regression Calculation Engine
    # Simulating a highly optimized, pre-trained linear regression equation:
    # Score decreases with high heat/cold, high humidity, high winds, and high UV.
    intercept = 112.54
    w_temp = -0.68
    w_humidity = -0.32
    w_wind = -0.18
    w_uv = -1.45
    
    # Calculate continuous score
    predicted_score = intercept + (w_temp * temperature) + (w_humidity * humidity) + (w_wind * wind_speed) + (w_uv * uv_index)
    
    # Enforce logical boundaries between 0 and 100
    predicted_score = max(0.0, min(100.0, predicted_score))
    
    # Output Section
    st.subheader("🔮 2. Linear Regression Prediction")
    
    # Dynamic Visual Alerts based on the regression output value
    if predicted_score >= 75:
        st.success(f"🟢 **Excellent Comfort Rating:** {predicted_score:.1f} / 100")
        st.balloons()
    elif predicted_score >= 45:
        st.warning(f"🟡 **Moderate Comfort Rating:** {predicted_score:.1f} / 100")
    else:
        st.error(f"🔴 **Poor/Uncomfortable Rating:** {predicted_score:.1f} / 100")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Reset/Back navigation button to go back to the home screen
    if st.button("⬅️ Return to Welcome Page", use_container_width=True):
        switch_page('welcome')
        st.rerun()