import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
df = pd.read_csv("data/flight_weather_balanced.csv")

# Create plots directory
os.makedirs("static/plots", exist_ok=True)

# Feature engineering
df[['Dep_Hour', 'Dep_Min']] = df['Scheduled_Departure'].str.split(':', expand=True).astype(int)
df.drop(columns=['Scheduled_Departure'], inplace=True)

# Encode categorical features
categorical_cols = ['Airline', 'Dep_Airport', 'Arr_Airport', 'Dep_Weather', 'Arr_Weather']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Normalize numerical columns
numerical_cols = [
    'Dep_Temp', 'Dep_Humidity', 'Dep_WindSpeed', 'Dep_Visibility', 'Dep_RainfallRate', 'Dep_SnowAccumulation',
    'Arr_Temp', 'Arr_Humidity', 'Arr_WindSpeed', 'Arr_Visibility', 'Arr_RainfallRate', 'Arr_SnowAccumulation',
    'Dep_Hour', 'Dep_Min'
]
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Features and target
X = df.drop(columns=['Delay_Status', 'Delay_Minutes'])
y = df['Delay_Status']

# Save column names
feature_columns = X.columns.tolist()

# Compute class imbalance
neg, pos = np.bincount(y)
scale_pos_weight = neg / pos

# Cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracies, f1_scores = [], []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    y_pred = model.predict(X.iloc[val_idx])
    acc = accuracy_score(y.iloc[val_idx], y_pred)
    f1 = f1_score(y.iloc[val_idx], y_pred)

    print(f"Fold {fold} - Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    accuracies.append(acc)
    f1_scores.append(f1)

# Final model training
final_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
final_model.fit(X, y)

# Save feature importance
xgb.plot_importance(final_model)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("static/plots/feature_importance.png")
plt.close()

# Save confusion matrix
y_pred_full = final_model.predict(X)
cm = confusion_matrix(y, y_pred_full)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Full Data)")
plt.tight_layout()
plt.savefig("static/plots/confusion_matrix.png")
plt.close()

# ✅ Save everything needed for inference
with open("models/xgboost_model.pkl", "wb") as f:
    pickle.dump(final_model, f)
with open("models/label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)
with open("models/feature_columns.pkl", "wb") as f:
    pickle.dump(feature_columns, f)
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\n✅ Model training complete.")
print(f"Average Accuracy: {np.mean(accuracies):.4f}")
print(f"Average F1 Score: {np.mean(f1_scores):.4f}")
