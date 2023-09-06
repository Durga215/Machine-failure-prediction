# Import necessary libraries
import pandas as pd
import numpy as np
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = tf.keras.models.load_model('model.h5')  # Replace with the actual path to your trained model

# Streamlit app
st.title('Machine Failure Prediction')

# Input fields for user to enter data
st.sidebar.header('Enter Data')
type_value = st.sidebar.selectbox('Type', ['Type_1', 'Type_2', 'Type_3'])  # Replace with actual types
air_temp = st.sidebar.number_input('Air Temperature [K]')
process_temp = st.sidebar.number_input('Process Temperature [K]')
rotational_speed = st.sidebar.number_input('Rotational Speed [rpm]')
torque = st.sidebar.number_input('Torque [Nm]')
tool_wear = st.sidebar.number_input('Tool Wear [min]')
twf = st.sidebar.selectbox('TWF', [0, 1])
hdf = st.sidebar.selectbox('HDF', [0, 1])
pwf = st.sidebar.selectbox('PWF', [0, 1])
osf = st.sidebar.selectbox('OSF', [0, 1])
rnf = st.sidebar.selectbox('RNF', [0, 1])

# Create a DataFrame with user input
user_input = pd.DataFrame({
    'Type': [type_value],
    'Air temperature [K]': [air_temp],
    'Process temperature [K]': [process_temp],
    'Rotational speed [rpm]': [rotational_speed],
    'Torque [Nm]': [torque],
    'Tool wear [min]': [tool_wear],
    'TWF': [twf],
    'HDF': [hdf],
    'PWF': [pwf],
    'OSF': [osf],
    'RNF': [rnf]
})

# Function to preprocess user input
def preprocess_user_input(input_data):
    # Create a new LabelEncoder for 'Type' if it wasn't saved during training
    le = LabelEncoder()
    input_data['Type'] = le.fit_transform(input_data['Type'])
    return input_data

# Preprocess the user input
preprocessed_input = preprocess_user_input(user_input)

# Standardize the input data (feature scaling)
scaler = StandardScaler()
preprocessed_input_scaled = scaler.fit_transform(preprocessed_input)

# Make predictions
if st.sidebar.button('Predict'):
    prediction = model.predict(preprocessed_input_scaled)

    # Display the results
    st.subheader('Predicted Machine Failure:')
    if prediction > 0.5:
        st.write('Yes')
    else:
        st.write('No')
