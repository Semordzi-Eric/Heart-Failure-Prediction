import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("C:/Users/DELL/Desktop/python/Classification/Heart_failure_project/Data/heart.csv")
    return df

# Data Processing
def preprocess_data(df):
    df_encoded = pd.get_dummies(df, drop_first=True)
    X = df_encoded.drop('HeartDisease', axis=1)
    y = df_encoded['HeartDisease']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42), scaler, X.columns

#  App
def main():
    st.title("Heart Disease Prediction App")

   
    df = load_data()
    st.write("Data Sample:")
    st.dataframe(df.head())

    
    (x_train, x_test, y_train, y_test), scaler, feature_columns = preprocess_data(df)
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(x_train, y_train)

    # User Inputs
    st.sidebar.header("Input Features")
    user_input = {
        'Age': st.sidebar.number_input("Age", 1, 120, 30),
        'Sex': 1 if st.sidebar.selectbox("Sex", ["M", "F"]) == "M" else 0,
        'RestingBP': st.sidebar.number_input("Resting Blood Pressure", 0, 200, 120),
        'Cholesterol': st.sidebar.number_input("Cholesterol", 0, 600, 200),
        'FastingBS': st.sidebar.selectbox("Fasting Blood Sugar > 120mg/dl", [0, 1]),
        'MaxHR': st.sidebar.number_input("Max Heart Rate", 60, 220, 100),
        'Oldpeak': st.sidebar.number_input("Oldpeak", value=1.0),
        'ChestPainType_ATA': 0,
        'ChestPainType_NAP': 0,
        'ChestPainType_TA': 0,
        'RestingECG_Normal': 0,
        'RestingECG_ST': 0,
        'ExerciseAngina_Y': 0,
        'ST_Slope_Flat': 0,
        'ST_Slope_Up': 0
    }

    # Categorical inputs
    cp = st.sidebar.selectbox("Chest Pain Type", ["ASY", "ATA", "NAP", "TA"])
    ecg = st.sidebar.selectbox("Resting ECG", ["LVH", "Normal", "ST"])
    angina = st.sidebar.selectbox("Exercise Angina", ["N", "Y"])
    slope = st.sidebar.selectbox("ST Slope", ["Down", "Flat", "Up"])

    if cp != "ASY":
        user_input[f"ChestPainType_{cp}"] = 1
    if ecg != "LVH":
        user_input[f"RestingECG_{ecg}"] = 1
    if angina == "Y":
        user_input["ExerciseAngina_Y"] = 1
    if slope != "Down":
        user_input[f"ST_Slope_{slope}"] = 1

    # A DataFrame
    user_df = pd.DataFrame([user_input])
    # Ensuring all feature columns exist in user_df
    for col in feature_columns:
        if col not in user_df.columns:
            user_df[col] = 0
    user_df = user_df[feature_columns]

    # Scaling user input
    user_scaled = scaler.transform(user_df)

    # Prediction
    prediction = model.predict(user_scaled)[0]
    st.subheader("Prediction Result:")
    st.success("Heart Disease Detected" if prediction == 1 else "No Heart Disease Detected")

if __name__ == "__main__":
    main()
