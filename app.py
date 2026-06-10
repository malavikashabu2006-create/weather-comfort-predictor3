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
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    
    # Clean, striking title presentation
    st.markdown("<h1 style='text-align: center; color: #1F497D;'>🌤️ Weather Comfort Score Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #595959; font-weight: normal;'>Your smart companion for planning perfect outdoor days.</h3>", unsafe_allow_html=True)
    
    st.markdown("<br><hr><br>", unsafe_allow_html=True)
    
    # Compact, high-impact block instead of "How it works"
    st.markdown("<p style='text-align: center; font-size: 18px;'>Ready to check your community's real-time environmental comfort index?</p>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Ultra-clean "GO" button spanning full layout width
    if st.button("GO", type="primary", use_container_width=True):
        switch_page('app_interface')
        st.rerun()

# ==========================================
# PAGE 2: MAIN PREDICTOR APP INTERFACE
# ==========================================
elif st.session_state.current_page == 'app_interface':
    st.markdown("<h2 style='color: #1F497D; margin-bottom: 0;'>📊 Weather Comfort Predictor Dashboard</h2>", unsafe_allow_html=True)
    st.write("Adjust the environmental factors. The underlying Linear Regression model calculates the rating instantly.")
    
    st.markdown("---")
    
    st.subheader("🔄 1. Tune Weather Metrics")
    
    # Arranging sliders in a 2x2 grid for a much more attractive, compact dashboard layout
    col1, col2 = st.columns(2)
    
    with col1:
        temperature = st.slider("Temperature (°C)", min_value=-10.0, max_value=45.0, value=24.0, step=0.5)
        wind_speed = st.slider("Wind Speed (km/h)", min_value=0.0, max_value=60.0, value=12.0, step=0.5)
        
    with col2:
        humidity = st.slider("Humidity (%)", min_value=0, max_value=100, value=50, step=1)
        uv_index = st.slider("UV Index Level", min_value=0, max_value=11, value=3, step=1)
        
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # 3. Linear Regression Engine 
    intercept = 112.54
    w_temp = -0.68
    w_humidity = -0.32
    w_wind = -0.18
    w_uv = -1.45
    
    predicted_score = intercept + (w_temp * temperature) + (w_humidity * humidity) + (w_wind * wind_speed) + (w_uv * uv_index)
    predicted_score = max(0.0, min(100.0, predicted_score))
    
    # 4. Massive Custom Metric Visualizer
    st.subheader("🔮 2. Linear Regression Prediction")
    
    # Set dynamic styling parameters based on the output score
    if predicted_score >= 75:
        bg_color = "#E2EFDA"      # Soft Green
        text_color = "#375623"    # Dark Green
        border_color = "#A9D08E"
        status_text = "🟢 Excellent Comfort Rating"
    elif predicted_score >= 45:
        bg_color = "#FFF2CC"      # Soft Yellow
        text_color = "#7F6000"    # Dark Gold
        border_color = "#FFD966"
        status_text = "🟡 Moderate Comfort Rating"
    else:
        bg_color = "#FCE4D6"      # Soft Red
        text_color = "#C65911"    # Dark Red
        border_color = "#F4B183"
        status_text = "🔴 Poor / Uncomfortable Rating"

    # Injecting massive custom HTML card for the metric score
    st.markdown(f"""
    <div style="background-color: {bg_color}; padding: 35px; border-radius: 15px; text-align: center; border: 2px solid {border_color}; box-shadow: 0px 4px 10px rgba(0,0,0,0.05);">
        <p style="font-size: 16px; font-weight: bold; color: #595959; letter-spacing: 1px; margin: 0; padding-bottom: 5px;">CURRENT METRIC PREDICTION</p>
        <h1 style="font-size: 80px; margin: 5px 0; color: {text_color}; font-weight: 900; line-height: 1;">{predicted_score:.1f}<span style="font-size: 30px; font-weight: normal; color: #595959;"> / 100</span></h1>
        <p style="font-size: 22px; font-weight: bold; color: {text_color}; margin: 10px 0 0 0;">{status_text}</p>
    </div>
    """, unsafe_allow_html=True)
        
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Return Button
    if st.button("⬅️ Return to Welcome Page", use_container_width=True):
        switch_page('welcome')
        st.rerun()