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

# Extract features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# Drop unnecessary columns (safe)
df = df.drop(['date', 'country'], axis=1)

# Handle missing values (important for deployment)
df = df.fillna(df.mean(numeric_only=True))

print("Data Preview:")
print(df.head())

# ================= FEATURE SELECTION =================
# ⚠️ MUST MATCH app.py EXACT ORDER
features = [
    'avg_temperature',
    'humidity',
    'co2_emission',
    'industrial_activity_index',
    'energy_price',
    'month',
    'day'
]

X = df[features]
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
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Energy Consumption")
plt.ylabel("Predicted Energy Consumption")
plt.title("Actual vs Predicted")
plt.show()

# ================= SAVE MODEL =================
joblib.dump(model, 'model/model.pkl')

print("\nModel saved successfully!")