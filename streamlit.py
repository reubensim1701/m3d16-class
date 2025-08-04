import streamlit as st
import pickle
import pandas as pd

# Load the trained model
with open('car_insurance_prediction.sav', 'rb') as file:
    model = pickle.load(file)

st.set_page_config(page_title="Car Insurance Claim Prediction", layout="centered")

st.title("🚗 Car Insurance Claim Prediction")

st.sidebar.header("Input Customer Information")

# Sidebar inputs
age = st.sidebar.selectbox("Age", ['16-25', '26-39', '40-64', '65+'])
gender = st.sidebar.selectbox("Gender", ['female', 'male'])
race = st.sidebar.selectbox("Race", ['majority', 'minority'])
driving_experience = st.sidebar.selectbox("Driving Experience", ['0-9y', '10-19y', '20-29y', '30y+'])
education = st.sidebar.selectbox("Education", ['none', 'high school', 'university'])
income = st.sidebar.selectbox("Income", ['poverty', 'working class', 'middle class', 'upper class'])
vehicle_ownership = st.sidebar.selectbox("Vehicle Ownership", ['no', 'yes'])
vehicle_year = st.sidebar.selectbox("Vehicle Year", ['before 2015', 'after 2015'])
married = st.sidebar.selectbox("Married", ['no', 'yes'])
children = st.sidebar.selectbox("Children", ['no', 'yes'])
city = st.sidebar.selectbox("City", ['santa rosa', 'oviedo', 'san diego', 'baltimore'])
region = st.sidebar.selectbox("Region", ['west', 'south', 'nan'])
state = st.sidebar.selectbox("State", ['california', 'florida', 'nan', 'maryland'])

credit_score = st.sidebar.slider("Credit Score", 0.05, 0.97, 0.52)
annual_mileage = st.sidebar.number_input("Annual Mileage", min_value=2000, max_value=22000, value=12000)
speeding_violations = st.sidebar.number_input("Speeding Violations", min_value=0, max_value=22, value=0)
past_accidents = st.sidebar.number_input("Past Accidents", min_value=0, max_value=15, value=0)

# Prepare input for prediction
input_dict = {
    'AGE': age,
    'GENDER': gender,
    'RACE': race,
    'DRIVING_EXPERIENCE': driving_experience,
    'EDUCATION': education,
    'INCOME': income,
    'VEHICLE_OWNERSHIP': vehicle_ownership,
    'VEHICLE_YEAR': vehicle_year,
    'MARRIED': married,
    'CHILDREN': children,
    'CITY': city,
    'REGION': region,
    'STATE': state,
    'CREDIT_SCORE': credit_score,
    'ANNUAL_MILEAGE': annual_mileage,
    'SPEEDING_VIOLATIONS': speeding_violations,
    'PAST_ACCIDENTS': past_accidents
}

input_df = pd.DataFrame([input_dict])

st.write("### Prediction Result")

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0][1]
        st.success(f"**Prediction:** {'Claim' if prediction == 1 else 'No Claim'}\n\n**Probability of Claim:** {proba:.2%}")
    else:
        st.success(f"**Prediction:** {'Claim' if prediction == 1 else 'No Claim'}")
else:
    st.info("Enter the details in the sidebar and click **Predict** to see the result.")

st.markdown("---")
st.caption("Model and app by your team. Powered by Streamlit.")