from flask import Flask, render_template, request
import joblib
import pandas as pd
import os
import pickle
import numpy as np
from src.api_utils import get_flight_info, get_weather
from app.visualizations import (
    plot_delay_distribution,
    plot_airline_delays,
    plot_weather_vs_delay,
    plot_airport_delay_trends,
    plot_feature_importance
)

app = Flask(__name__)

# Load models and artifacts
xgb_model = joblib.load('models/xgboost_model.pkl')
xgb_reg_model = joblib.load("models/xgboost_regressor_model.pkl")

with open('models/feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

with open('models/label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

WEATHER_API_KEY = '83942d880e99433aa0b162804251006'
training_df = pd.read_csv('data/flight_weather_balanced.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_realtime', methods=['POST'])
def predict():
    flight_number = request.form['flight_number']

    # Get flight and weather data (no date required)
    flight_data = get_flight_info(flight_number)
    if not flight_data:
        return "Flight not found. Try examples like AI101, EK500, or BA143."

    dep_weather = get_weather(flight_data['departure_code'], WEATHER_API_KEY)
    arr_weather = get_weather(flight_data['arrival_code'], WEATHER_API_KEY)
    if not dep_weather or not arr_weather:
        return "Weather data unavailable."

    dep_hour, dep_minute = map(int, flight_data['scheduled_departure'].split(':'))

    input_data = {
        'Airline': flight_data['airline'],
        'Dep_Airport': flight_data['departure_code'],
        'Arr_Airport': flight_data['arrival_code'],
        'Dep_Weather': dep_weather['condition'],
        'Arr_Weather': arr_weather['condition'],
        'Dep_Temp': dep_weather['temperature'],
        'Dep_Humidity': dep_weather['humidity'],
        'Dep_WindSpeed': dep_weather['wind_kph'],
        'Dep_Visibility': dep_weather['vis_km'],
        'Dep_RainfallRate': dep_weather['precip_mm'],
        'Dep_SnowAccumulation': dep_weather['snow_cm'],
        'Arr_Temp': arr_weather['temperature'],
        'Arr_Humidity': arr_weather['humidity'],
        'Arr_WindSpeed': arr_weather['wind_kph'],
        'Arr_Visibility': arr_weather['vis_km'],
        'Arr_RainfallRate': arr_weather['precip_mm'],
        'Arr_SnowAccumulation': arr_weather['snow_cm'],
        'Dep_Hour': dep_hour,
        'Dep_Min': dep_minute
    }

    df_input = pd.DataFrame([input_data])

    # Encode categorical features using label encoders
    for col in ['Airline', 'Dep_Airport', 'Arr_Airport', 'Dep_Weather', 'Arr_Weather']:
        le = label_encoders.get(col)
        if le:
            val = df_input[col].iloc[0]
            if val not in le.classes_:
                le.classes_ = np.append(le.classes_, val)
            df_input[col] = le.transform([val])[0]  # assign scalar, not array

    # Reindex and scale features
    df_input_encoded = df_input.reindex(columns=feature_columns, fill_value=0)
    df_input_scaled = pd.DataFrame(scaler.transform(df_input_encoded), columns=feature_columns)

    # Predict delay status and delay time
    delay_status = xgb_model.predict(df_input_scaled)[0]
    delay_minutes = xgb_reg_model.predict(df_input_scaled)[0]
    delay_minutes = np.clip(delay_minutes, 0, 600)

    status = "Delayed" if delay_status == 1 or delay_minutes >= 15 else "On-Time"
    delay_reasons = get_delay_reasons(dep_weather, arr_weather)

    # Generate visualizations
    plot_delay_distribution(training_df)
    plot_airline_delays(training_df)
    plot_weather_vs_delay(training_df)
    plot_airport_delay_trends(training_df)
    plot_feature_importance(xgb_model, feature_columns)  # ✅ Corrected


    return render_template('result.html',
        status=status,
        delay_minutes=round(delay_minutes),
        flight_number=flight_number,
        airline=flight_data['airline'],
        dep_weather=dep_weather,
        arr_weather=arr_weather,
        delay_reasons=delay_reasons,
        dep_code=flight_data['departure_code'],
        arr_code=flight_data['arrival_code']
    )

def get_delay_reasons(dep_weather, arr_weather):
    reasons = []
    if dep_weather['vis_km'] < 2:
        reasons.append("Low visibility at departure")
    if dep_weather['wind_kph'] > 40:
        reasons.append("Strong winds at departure")
    if dep_weather['precip_mm'] > 20:
        reasons.append("Heavy rainfall at departure")
    if dep_weather['temperature'] < -20 or dep_weather['temperature'] > 40:
        reasons.append("Extreme temperature at departure")

    if arr_weather['vis_km'] < 2:
        reasons.append("Low visibility at arrival")
    if arr_weather['wind_kph'] > 40:
        reasons.append("Strong winds at arrival")
    if arr_weather['precip_mm'] > 20:
        reasons.append("Heavy rainfall at arrival")
    if arr_weather['temperature'] < -20 or arr_weather['temperature'] > 40:
        reasons.append("Extreme temperature at arrival")

    return reasons if reasons else ["Weather conditions are normal"]

if __name__ == '__main__':
    app.run(debug=True)
