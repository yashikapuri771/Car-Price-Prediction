import streamlit as st
import pandas as pd
import pickle

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Car Price Predictor",
    layout="centered"
)

# ================= THEME TOGGLE =================
dark_mode = st.toggle("🌙 Dark Mode")

# ================= THEME CSS =================
if dark_mode:
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0f172a;
            color: #e5e7eb;
        }

        h1 {
            color: #f8fafc;
        }

        p, label {
            color: #cbd5f5;
        }

        div[data-testid="stSelectbox"] > div,
        div[data-testid="stNumberInput"] > div {
            background-color: #1e293b;
            color: #f8fafc;
            border-radius: 10px;
            border: 1px solid #334155;
        }

        div[data-testid="stButton"] > button {
            background-color: #2563eb;
            color: white;
            border-radius: 10px;
            padding: 10px 26px;
            border: none;
            font-weight: 600;
            font-size: 16px;
        }

        div[data-testid="stButton"] > button:hover {
            background-color: #1d4ed8;
        }

        .stAlert {
            background-color: #052e16 !important;
            color: #bbf7d0 !important;
            border-radius: 10px;
            border: 1px solid #166534;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #ffffff;
            color: #0f172a;
        }

        h1 {
            color: #0f172a;
        }

        p, label {
            color: #334155;
        }

        div[data-testid="stSelectbox"] > div,
        div[data-testid="stNumberInput"] > div {
            background-color: #f8fafc;
            border-radius: 10px;
            border: 1px solid #e2e8f0;
        }

        div[data-testid="stButton"] > button {
            background-color: #2563eb;
            color: white;
            border-radius: 10px;
            padding: 10px 26px;
            border: none;
            font-weight: 600;
            font-size: 16px;
        }

        div[data-testid="stButton"] > button:hover {
            background-color: #1d4ed8;
        }

        .stAlert {
            background-color: #ecfdf5 !important;
            color: #065f46 !important;
            border-radius: 10px;
            border: 1px solid #bbf7d0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# ================= LOAD DATA & MODEL =================
data = pd.read_csv("Data/quikr_car.csv")

with open("LinearRegressionModel.pkl", "rb") as f:
    model = pickle.load(f)

# ================= TITLE =================
st.title("Car Price Predictor")
st.write(
    "Predict the resale price of a car based on company, model, "
    "year, fuel type and kilometres driven."
)

st.markdown("---")

# ================= INPUTS =================
companies = sorted(data["company"].dropna().astype(str).unique())
fuels = sorted(data["fuel_type"].dropna().astype(str).unique())

years = sorted(
    pd.to_numeric(data["year"], errors="coerce")
    .dropna()
    .astype(int)
    .unique(),
    reverse=True
)

company = st.selectbox("Select Company", companies)

models = sorted(
    data[data["company"] == company]["name"]
    .dropna()
    .astype(str)
    .unique()
)

car_name = st.selectbox("Select Car Model", models)

year = st.selectbox("Select Year of Purchase", years)

fuel = st.selectbox("Select Fuel Type", fuels)

kms = st.number_input(
    "Enter Kilometres Driven",
    min_value=0,
    step=500
)

st.markdown("---")

# ================= PREDICTION =================
if st.button("Predict Price"):
    input_df = pd.DataFrame({
        "name": [car_name],
        "company": [company],
        "fuel_type": [fuel],
        "year": [year],
        "kms_driven": [kms]
    })

    price = model.predict(input_df)[0]
    st.success(f"💰 Estimated Car Price: ₹ {int(price):,}")
    st.caption("Price is an estimate based on historical data.")
