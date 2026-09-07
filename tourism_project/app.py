import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="🌴",
    layout="wide"
)

# Load trained model
MODEL_PATH = "tourism_model.pkl"
model = joblib.load(MODEL_PATH)

# Title
st.title("🌴 Tourism Package Purchase Prediction")
st.write(
    "Enter the customer details below to predict whether the customer "
    "is likely to purchase the tourism package."
)

st.divider()

# Customer information
st.subheader("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    city_tier = st.selectbox(
        "City Tier",
        [1, 2, 3]
    )

    occupation = st.selectbox(
        "Occupation",
        ["Salaried", "Free Lancer", "Small Business", "Large Business"]
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    marital_status = st.selectbox(
        "Marital Status",
        ["Single", "Divorced", "Married", "Unmarried"]
    )

with col2:
    type_of_contact = st.selectbox(
        "Type of Contact",
        ["Self Enquiry", "Company Invited"]
    )

    duration_of_pitch = st.number_input(
        "Duration of Pitch (minutes)",
        min_value=0,
        value=15
    )

    number_of_person_visiting = st.number_input(
        "Number of Persons Visiting",
        min_value=1,
        value=2
    )

    number_of_followups = st.number_input(
        "Number of Followups",
        min_value=0,
        value=3
    )

    product_pitched = st.selectbox(
        "Product Pitched",
        ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]
    )

with col3:
    preferred_property_star = st.number_input(
        "Preferred Property Star",
        min_value=1,
        max_value=5,
        value=3
    )

    number_of_trips = st.number_input(
        "Number of Trips",
        min_value=0,
        value=2
    )

    passport = st.selectbox(
        "Passport",
        [0, 1]
    )

    pitch_satisfaction_score = st.number_input(
        "Pitch Satisfaction Score",
        min_value=1,
        max_value=5,
        value=3
    )

    own_car = st.selectbox(
        "Own Car",
        [0, 1]
    )

    number_of_children_visiting = st.number_input(
        "Number of Children Visiting",
        min_value=0,
        value=0
    )

    designation = st.selectbox(
        "Designation",
        ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
    )

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=0,
        value=25000
    )

# Prediction button
st.divider()

if st.button("Predict Package Purchase", type="primary"):

    input_data = pd.DataFrame([{
        "Age": age,
        "TypeofContact": type_of_contact,
        "CityTier": city_tier,
        "DurationOfPitch": duration_of_pitch,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": number_of_person_visiting,
        "NumberOfFollowups": number_of_followups,
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": preferred_property_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": number_of_trips,
        "Passport": passport,
        "PitchSatisfactionScore": pitch_satisfaction_score,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": number_of_children_visiting,
        "Designation": designation,
        "MonthlyIncome": monthly_income
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(
            f"Customer is likely to purchase the tourism package. "
            f"Probability: {probability:.2%}"
        )
    else:
        st.warning(
            f"Customer is unlikely to purchase the tourism package. "
            f"Probability: {probability:.2%}"
        )