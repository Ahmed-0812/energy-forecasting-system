import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ================= LOAD DATA =================
df = pd.read_csv('data/energy_data.csv')

# ================= PREPROCESSING =================
df['date'] = pd.to_datetime(df['date'])

# Feature engineering (important)
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# Drop unnecessary columns
df = df.drop(['date', 'country'], axis=1)

print("Data Preview:")
print(df.head())

# ================= FEATURE SELECTION =================
# Use only strong features (better performance)
X = df[['avg_temperature', 'humidity', 'co2_emission',
        'industrial_activity_index', 'energy_price',
        'month', 'day']]

y = df['energy_consumption']

# ================= TRAIN TEST SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================= MODEL =================
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# ================= EVALUATION =================
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("Mean Absolute Error (MAE):", mae)
print("R2 Score:", r2)

# ================= VISUALIZATION =================
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Energy Consumption")
plt.ylabel("Predicted Energy Consumption")
plt.title("Actual vs Predicted")
plt.show()

# ================= SAVE MODEL =================
joblib.dump(model, 'model/model.pkl')
print("\nModel saved successfully!")