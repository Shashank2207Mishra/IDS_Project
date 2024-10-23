import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("Flight Price Predictor: Predict the Price of Domestic Flights")

# Load dataset and pipeline
dataset = pickle.load(open("dataset.pkl", "rb"))
pipe = pickle.load(open("pipeline_1.pkl", "rb"))

# User input selection
Airline = st.selectbox("Select Airline to Travel", dataset['Airline'].unique())
Source = st.selectbox("Select Origin", dataset['Source'].unique())
Destination = st.selectbox("Select Destination", dataset['Destination'].unique())
day = st.selectbox("Select day of Journey", np.arange(1, 32))
month = st.selectbox("Select Month Of Journey", np.arange(1, 13))
year = 2024
Total_Stops = st.selectbox("Select total Number of Stops", dataset['Total_Stops'].unique())
Dep_hr = st.selectbox("Select Time (hr) For the Journey", dataset['Dep_hr'].sort_values().unique())
Dep_Minz = st.selectbox("Select Time (min) for the journey", np.arange(10, 70, 10))
Arrival_hr = st.selectbox("Select Time (hr) of the arrival", dataset['Arrival_hr'].sort_values().unique())
Arrival_minz = st.selectbox("Select Time (min) of the arrival", np.arange(10, 70, 10))
Total_Duration_hrs = st.selectbox("Select Total Duration in hrs", np.arange(2,100,0.5))
Additional_Info = st.selectbox("Any other additional Info about the Flight", dataset['Additional_Info'].unique())

# Initialize a variable to track whether the prediction has been made
predicted_price = 0
st.subheader("Click Here to Predict Approximate Price (Note plus minus Rs 400 is tolerance limit")

# Button to trigger prediction
if st.button("Predict Price"):
    # Prepare query as DataFrame
    query = pd.DataFrame({
        'Airline': [Airline],
        'Source': [Source],
        'Destination': [Destination],
        'day': [day],
        'month': [month],
        'year': [year],
        'Total_Stops': [Total_Stops],
        'Dep_hr': [Dep_hr],
        'Dep_Minz': [Dep_Minz],
        'Arrival_hr': [Arrival_hr],
        'Arrival_minz': [Arrival_minz],
        'Total_Duration_hrs': [Total_Duration_hrs],
        'Additional_Info': [Additional_Info]
    })

    # Predict and update the predicted price
    predicted_price = pipe.predict(query)[0]

# Display the price - default 0 or predicted value
if predicted_price == 0:
    st.title("Predicted Price: 0")
else:
    st.title(f"Predicted Best  Price Should be Rs: {predicted_price+0.41*predicted_price}")
