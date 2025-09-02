import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -----------------------------
# Load model and preprocessors
# -----------------------------
@st.cache_resource
def load_artifacts():
    # These should be pickled after training (model, encoder, scaler)
    model = pickle.load(open("../models/rforest_tuned.pkl", "rb"))
    ohe = pickle.load(open("../encoders/one_hot_encoder.pkl", "rb"))
    scaler = pickle.load(open("../scalers/standard_scaler.pkl", "rb"))
    return model, ohe, scaler

model, ohe, scaler = load_artifacts()

# -----------------------------
# Streamlit App UI
# -----------------------------
st.title("🏠 NYC Airbnb Rental Price Predictor")
st.write("Enter the details of the listing to predict the nightly rental price.")

# Dropdowns and inputs
col1, col2 = st.columns(2)

with col1:
    host_is_superhost = st.selectbox("Is Host Superhost?", ["True", "False"])
    neighbourhood_group = st.selectbox(
        "Neighbourhood Group",
        ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
    )
    room_type = st.selectbox(
        "Room Type",
        ["Entire home/apt", "Hotel room", "Private room", "Shared room"]
    )
    latitude = st.number_input("Latitude", value=40.7128, format="%.6f")
    longitude = st.number_input("Longitude", value=-74.0060, format="%.6f")
    accommodates = st.number_input("Accommodates", min_value=1, value=2)

with col2:
    bedrooms = st.number_input("Bedrooms", min_value=0, value=1)
    maximum_nights = st.number_input("Maximum Nights", min_value=1, value=365, step=1)
    availability_365 = st.number_input("Availability (days per year)", min_value=0, max_value=365, value=180)
    number_of_reviews = st.number_input("Number of Reviews", min_value=0, value=10)
    calculated_host_listings_count = st.number_input("Calculated Host Listings Count", min_value=1, value=1)
    total_amenities = st.number_input("Total Amenities", min_value=0, value=5)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Price"):
    # Build input dataframe
    input_df = pd.DataFrame({
        "host_is_superhost": [host_is_superhost],
        "neighbourhood_group_cleansed": [neighbourhood_group],
        "room_type": [room_type],
        "latitude": [latitude],
        "longitude": [longitude],
        "accommodates": [accommodates],
        "bedrooms": [bedrooms],
        "maximum_nights": [maximum_nights],
        "availability_365": [availability_365],
        "number_of_reviews": [number_of_reviews],
        "calculated_host_listings_count": [calculated_host_listings_count],
        "total_amenities": [total_amenities]
    })

    # Apply same preprocessing as training
    # Log transforms
    for col in ["number_of_reviews", "calculated_host_listings_count", "accommodates"]:
        input_df[col] = np.log1p(input_df[col])  # log(1+x) to handle zeros

    # One-hot encode categorical columns
    cat_cols = ["host_is_superhost", "neighbourhood_group_cleansed", "room_type"]
    input_ohe = ohe.transform(input_df[cat_cols])
    input_ohe_df = pd.DataFrame(input_ohe, columns=ohe.get_feature_names_out(cat_cols))

    # Combine with numeric features
    num_cols = [c for c in input_df.columns if c not in cat_cols]
    input_num_df = input_df[num_cols].reset_index(drop=True)
    input_full = pd.concat([input_ohe_df, input_num_df], axis=1)

    # Scale
    input_scaled = scaler.transform(input_full)

    # Predict
    prediction = model.predict(input_scaled)

    st.success(f"💰 Estimated Rental Price: ${prediction[0]:.2f} per night")
