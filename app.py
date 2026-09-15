import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('logi.sav')

st.title('Delivery Delay Prediction')

st.write("Enter the features below to predict delivery delay:")

# Define input fields based on x.columns
# 'Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
# 'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age',
# 'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency',
# 'Warehouse_Processing_Time'

delivery_distance = st.slider('Delivery Distance', 0.0, 100.0, 20.0)
traffic_congestion = st.slider('Traffic Congestion (1-5)', 1, 5, 3)
weather_condition = st.slider('Weather Condition (1-5)', 1, 5, 3)
delivery_slot = st.slider('Delivery Slot (1-3)', 1, 3, 2)
driver_experience = st.slider('Driver Experience (years)', 0, 20, 5)
num_stops = st.slider('Number of Stops', 0, 10, 2)
vehicle_age = st.slider('Vehicle Age (years)', 0, 10, 3)
road_condition_score = st.slider('Road Condition Score (1-5)', 1, 5, 3)
package_weight = st.slider('Package Weight', 0.0, 200.0, 120.0)
fuel_efficiency = st.slider('Fuel Efficiency', 0.0, 30.0, 12.0)
warehouse_processing_time = st.slider('Warehouse Processing Time', 0, 150, 120)

# Create a DataFrame from user inputs
input_data = pd.DataFrame([{
    'Delivery_Distance': delivery_distance,
    'Traffic_Congestion': traffic_congestion,
    'Weather_Condition': weather_condition,
    'Delivery_Slot': delivery_slot,
    'Driver_Experience': driver_experience,
    'Num_Stops': num_stops,
    'Vehicle_Age': vehicle_age,
    'Road_Condition_Score': road_condition_score,
    'Package_Weight': package_weight,
    'Fuel_Efficiency': fuel_efficiency,
    'Warehouse_Processing_Time': warehouse_processing_time
}])

# Make prediction
if st.button('Predict Delivery Delay'):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.error('Prediction: Delivery is LIKELY to be Delayed')
    else:
        st.success('Prediction: Delivery is LIKELY to be On Time')
    
    st.write(f"Probability of Delay: {prediction_proba[0][1]:.2f}")
    st.write(f"Probability of On Time: {prediction_proba[0][0]:.2f}")
