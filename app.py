import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import datetime

st.title("💧 Water Quality Pollutants Predictor")

# === Load dataset from local file (NO file upload) ===
@st.cache_data
def load_data():
    df = pd.read_csv("PB_All_2000_2021.csv", sep=';')
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y', errors='coerce')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    return df

df = load_data()

pollutants = ['NH4', 'BSK5', 'Suspended', 'O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']
df = df.dropna(subset=pollutants, how='all')

# Feature/Target split
X = df[['id', 'year', 'month']]
y = df[pollutants]

# Impute missing values
imputer = SimpleImputer(strategy='mean')
y_imputed = imputer.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_imputed, test_size=0.2, random_state=42)

# Model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
model = MultiOutputRegressor(rf)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluation
r2_scores = r2_score(y_test, y_pred, multioutput='raw_values')
rmse_scores = np.sqrt(mean_squared_error(y_test, y_pred, multioutput='raw_values'))

results = pd.DataFrame({
    'Pollutant': pollutants,
    'R² Score': r2_scores,
    'RMSE': rmse_scores
}).sort_values(by='R² Score', ascending=False)

st.subheader("📊 Model Performance")
st.dataframe(results.style.format({'R² Score': "{:.3f}", 'RMSE': "{:.2f}"}))

# Optional: Prediction
st.subheader("🔍 Predict Pollutants")
id_input = st.number_input("Station ID", min_value=1, value=1)
year_input = st.number_input("Year", min_value=2000, max_value=2100, value=2023)
month_input = st.number_input("Month", min_value=1, max_value=12, value=6)

if st.button("Predict"):
    input_df = pd.DataFrame([[id_input, year_input, month_input]], columns=['id', 'year', 'month'])
    prediction = model.predict(input_df)[0]

    pred_df = pd.DataFrame({'Pollutant': pollutants, 'Predicted Value': prediction})
    st.dataframe(pred_df.style.format({'Predicted Value': "{:.3f}"}))
