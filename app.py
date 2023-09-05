



import streamlit as st
import numpy as np



import pandas as pd
import pickle




    




def check_churn(seq1):
    loaded_model=load_curr_model()
    
    k=loaded_model.predict(seq1.reshape((1,-1)))
    if(k==0):
        return "Will not churn"
    if(k==1):
        return "Will Churn"
    
    
    
st.title("Churn Prediction")
st.sidebar.header("User Input Features")

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://editor.analyticsvidhya.com/uploads/17047What-stops-customer-churn-Having-a-centralized-data-hub-does-and-heres-why.jpeg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 
    
@st.cache_data()
def load_curr_model():
    with st.spinner('Model is being loaded..'):
        with open('decision_tree_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    
   

    # Get user input
age = st.sidebar.slider("Age", min_value=18, max_value=100, step=1)

# Subscription Length (in months)
subscription_length = st.sidebar.slider("Subscription Length (Months)", min_value=1, max_value=60, step=1)

# Monthly Bill
monthly_bill = st.sidebar.number_input("Monthly Bill", min_value=0.0, step=1.0)

# Total Usage (in GB)
total_usage_gb = st.sidebar.number_input("Total Usage (GB)", min_value=0.0, step=1.0)

# Gender (Male)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

# Location
location_options = ["Houston", "Los Angeles", "Miami", "New York","Chicago"]
selected_location = st.sidebar.selectbox("Location", location_options)

# Convert gender and location to binary values (1 or 0)
gender_male_encoded = 1 if gender == "Male" else 0
location_encoded = [1 if selected_location == loc else 0 for loc in location_options]

# Create a button to trigger the prediction
if st.sidebar.button("Predict"):
    # Prepare the user input as a feature vector
    user_input = [age, subscription_length, monthly_bill, total_usage_gb, gender_male_encoded] + location_encoded[:-1]

    # Make a prediction
    prediction = check_churn(np.asarray(user_input).reshape((1,-1)))

    # Display the prediction result
    st.write("Churn Prediction:", prediction)
    

st.write("\n\n By Abhishek Ambast.")






