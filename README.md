
HEART DISEASE PREDICTION APP

This is a **Streamlit web application** that predicts the likelihood of **heart disease** using a machine learning model (XGBoost Classifier) trained on clinical data.



FEATURES

* Easy-to-use web interface (Streamlit)
* Predicts heart disease risk based on:

  * Age, Sex, Chest Pain Type
  * Resting Blood Pressure
  * Cholesterol, Fasting Blood Sugar
  * Resting ECG, Max Heart Rate
  * Exercise Angina, Oldpeak, ST Slope
* Real-time predictions
* Preprocessing includes scaling and one-hot encoding



DATASET

The dataset used is from [Heart Disease UCI](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction).


MODEL

The model used is:

* **XGBoost Classifier** (`XGBClassifier`)
* Trained with:

  * 80/20 train-test split
  * StandardScaler for feature scaling
  * One-hot encoding for categorical variables

INSTALLATION

1. Clone the repo:


2. Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

> `requirements.txt` should include:
>
> ```
> streamlit
> pandas
> xgboost
> scikit-learn
> matplotlib
> seaborn
> ```



RUN APP

```bash
streamlit run app.py
```

Then open the app in your browser at `http://localhost:8501`


EXAMPLE PREDICTION

You’ll be asked to enter patient information on the sidebar, and the app will instantly predict if they are at risk of heart disease.


