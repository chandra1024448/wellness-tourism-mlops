import streamlit as st
import pandas as pd
import pickle

st.title("🏖️ Tourism Package Purchase Prediction")
st.write("Predict whether a customer will buy a travel package.")

# Load model
with open("rf_random_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

# UI Inputs
Age = st.number_input("Age", 18, 80, 30)
TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
DurationOfPitch = st.number_input("Duration of Pitch (minutes)", 0, 100, 10)
Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number of Person Visiting", 1, 5, 2)
NumberOfFollowups = st.number_input("Number of Followups", 1, 6, 3)
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
PreferredPropertyStar = st.selectbox("Preferred Property Star", [3, 4, 5])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Unmarried", "Divorced"])
NumberOfTrips = st.number_input("Number of Trips", 0, 20, 2)
Passport = st.selectbox("Passport", ["Yes", "No"])
PitchSatisfactionScore = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
OwnCar = st.selectbox("Own Car", ["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", 0, 5, 0)
Designation = st.selectbox("Designation Level", [0,1,2,3,4])
MonthlyIncome = st.number_input("Monthly Income", 0, 1000000, 25000)

# Apply LabelEncoder Mappings

contact_map = {"Company Invited": 0, "Self Enquiry": 1}
occupation_map = {"Free Lancer": 0, "Large Business": 1, "Salaried": 2, "Small Business": 3}
gender_map = {"Female": 0, "Male": 1}
product_map = {"Basic": 0, "Deluxe": 1, "King": 2, "Standard": 3, "Super Deluxe": 4}
marital_map = {"Divorced": 0, "Married": 1, "Single": 2, "Unmarried": 3}
passport_map = {"No": 0, "Yes": 1}
car_map = {"No": 0, "Yes": 1}

# Build dataframe
input_df = pd.DataFrame({
    "Unnamed: 0": [0],
    "CustomerID": [0],
    "Age": [Age],
    "TypeofContact": [contact_map[TypeofContact]],
    "CityTier": [CityTier],
    "DurationOfPitch": [DurationOfPitch],
    "Occupation": [occupation_map[Occupation]],
    "Gender": [gender_map[Gender]],
    "NumberOfPersonVisiting": [NumberOfPersonVisiting],
    "NumberOfFollowups": [NumberOfFollowups],
    "ProductPitched": [product_map[ProductPitched]],
    "PreferredPropertyStar": [PreferredPropertyStar],
    "MaritalStatus": [marital_map[MaritalStatus]],
    "NumberOfTrips": [NumberOfTrips],
    "Passport": [passport_map[Passport]],
    "PitchSatisfactionScore": [PitchSatisfactionScore],
    "OwnCar": [car_map[OwnCar]],
    "NumberOfChildrenVisiting": [NumberOfChildrenVisiting],
    "Designation": [Designation],
    "MonthlyIncome": [MonthlyIncome]
})

st.write("### Input Summary")
st.write(input_df)

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.success("🎉 Customer is likely to PURCHASE the package.")
    else:
        st.error("❌ Customer is NOT likely to purchase the package.")