import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("data/flight_weather_balanced.csv")

# Feature engineering
df[['Dep_Hour', 'Dep_Min']] = df['Scheduled_Departure'].str.split(':', expand=True).astype(int)
df.drop(columns=['Scheduled_Departure'], inplace=True)

# Clip unrealistic delay values (e.g., > 300 minutes)
df['Delay_Minutes'] = df['Delay_Minutes'].clip(upper=300)

# Encode categorical features
categorical_cols = ['Airline', 'Dep_Airport', 'Arr_Airport', 'Dep_Weather', 'Arr_Weather']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Normalize numeric features
num_cols = ['Dep_Temp', 'Dep_Humidity', 'Dep_WindSpeed', 'Dep_Visibility',
            'Dep_RainfallRate', 'Dep_SnowAccumulation', 'Arr_Temp', 'Arr_Humidity',
            'Arr_WindSpeed', 'Arr_Visibility', 'Arr_RainfallRate', 'Arr_SnowAccumulation',
            'Dep_Hour', 'Dep_Min']
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Features and target
X = df.drop(columns=['Delay_Status', 'Delay_Minutes'])
y = df['Delay_Minutes']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost regressor
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

# Predict and clip outputs
y_pred = model.predict(X_test)
y_pred = np.clip(y_pred, 0, 300)

# Evaluate
print("✅ MAE:", mean_absolute_error(y_test, y_pred))
print("✅ R2 Score:", r2_score(y_test, y_pred))

# Save model and encoders
with open("models/xgboost_regressor_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

with open("models/feature_columns.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model, encoders, and scaler saved successfully.")
