import streamlit as st
import numpy as np
import joblib
import vonage

# Load the pre-trained model
model = joblib.load('flood_model.pkl')

# Vonage client setup for sending SMS
# client = 0 Key and secret
# sms = vonage.Sms(client)

# Phone numbers to notify in case of flood risk
phone_numbers = []

# Function to simulate fetching system-provided data
def fetch_system_data():
    return {'rainfall': 100.0, 'water_level': 5.0, 'wind_speed': 10.0, 'ndwi': 0.4}

# Streamlit app title
st.title('Flood Prediction and SMS Notification Dashboard')

# Radio button to choose data source
data_source = st.radio("Choose Data Source", ('Use System Data', 'Enter Data Manually'))

# Handling data input based on selected source
if data_source == 'Use System Data':
    data = fetch_system_data()
    rainfall = data['rainfall']
    water_level = data['water_level']
    wind_speed = data['wind_speed']
    ndwi = data['ndwi']
    st.write(f"System Data: Rainfall: {rainfall} mm, Water Level: {water_level} m, Wind Speed: {wind_speed} m/s, NDWI: {ndwi}")
else:
    rainfall = st.number_input('Rainfall (mm)', min_value=0.0, step=0.1)
    water_level = st.number_input('Water Level (m)', min_value=0.0, step=0.1)
    wind_speed = st.number_input('Wind Speed (m/s)', min_value=0.0, step=0.1)
    ndwi = st.number_input('NDWI', min_value=-1.0, max_value=1.0, step=0.01)

# Predict button and flood risk prediction
if st.button('Predict Flood Risk'):
    input_data = np.array([[ndwi, rainfall, water_level, wind_speed]])
    prediction = model.predict(input_data)
    result = 'Flood Expected' if prediction == 1 else 'No Flood'
    st.write(f'Prediction: {result}')

    # If flood is expected, send SMS notifications
    if prediction == 1:
        for phone_number in phone_numbers:
            responseData = sms.send_message({
                "from": "FloodAlert",
                "to": phone_number,
                "text": "Warning: A flood risk has been detected in your area. Please take necessary precautions."
            })

            if responseData["messages"][0]["status"] == "0":
                st.write(f"Message sent successfully to {phone_number}.")
            else:
                st.write(f"Message to {phone_number} failed with error: {responseData['messages'][0]['error-text']}")
