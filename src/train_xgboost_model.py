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

df = pd.read_csv("data/flight_weather_balanced.csv")

# Extract departure hour and minute
df[['Dep_Hour', 'Dep_Min']] = df['Scheduled_Departure'].str.split(':', expand=True).astype(int)
df.drop(columns=['Scheduled_Departure'], inplace=True)

# Encode categorical columns
categorical_cols = ['Airline', 'Dep_Airport', 'Arr_Airport', 'Dep_Weather', 'Arr_Weather']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and Target
X = df.drop(columns=['Delay_Status', 'Delay_Minutes'])
y = df['Delay_Status']

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
feature_columns = X.columns.tolist()  # Save column names before converting

# Compute scale_pos_weight for imbalance
neg, pos = np.bincount(y)
scale_pos_weight = neg / pos

# Cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracies, f1_scores = [], []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled, y), start=1):
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    print(f"\nFold {fold} Results:")
    print("Accuracy:", acc)
    print("F1 Score:", f1)
    print(classification_report(y_val, y_pred))

    accuracies.append(acc)
    f1_scores.append(f1)

# Final model on all data
final_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
final_model.fit(X_scaled, y)

# Save plots
os.makedirs("static/plots", exist_ok=True)
xgb.plot_importance(final_model)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("static/plots/feature_importance.png")
plt.close()

y_pred_full = final_model.predict(X_scaled)
cm = confusion_matrix(y, y_pred_full)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("static/plots/confusion_matrix.png")
plt.close()

# Save artifacts
os.makedirs("models", exist_ok=True)
with open("models/xgboost_model.pkl", "wb") as f:
    pickle.dump(final_model, f)

with open("models/label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

with open("models/feature_columns.pkl", "wb") as f:
    pickle.dump(feature_columns, f)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\n✅ Model training complete.")
print(f"Avg Accuracy: {np.mean(accuracies):.4f}, Avg F1 Score: {np.mean(f1_scores):.4f}")