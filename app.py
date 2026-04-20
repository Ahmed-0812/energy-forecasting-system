import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Energy Forecasting App", layout="wide")

# ================= CUSTOM UI =================
st.markdown("""
    <style>
    .main {background-color: #0E1117; color: white;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 10px;}
    </style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.title("⚡ Smart Energy Demand Forecasting")
st.markdown("### 📌 Predict energy consumption using Machine Learning")

st.info("This app predicts energy consumption based on environmental and industrial factors.")

# ================= SIDEBAR =================
st.sidebar.header("🔧 Input Parameters")

temp = st.sidebar.number_input("Average Temperature", value=25.0)
humidity = st.sidebar.number_input("Humidity", value=50.0)
co2 = st.sidebar.number_input("CO2 Emission", value=300.0)
industry = st.sidebar.number_input("Industrial Activity Index", value=50.0)
price = st.sidebar.number_input("Energy Price", value=100.0)
month = st.sidebar.slider("Month", 1, 12, 6)
day = st.sidebar.slider("Day", 1, 31, 15)

# ================= LOAD DATA =================
df = pd.read_csv('data/energy_data.csv')

# ================= PREPROCESSING =================
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# ================= TRAIN MODEL INSIDE APP =================
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

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# ================= PREDICTION =================
if st.button("🔍 Predict Energy Consumption"):
    input_data = pd.DataFrame({
        'avg_temperature': [temp],
        'humidity': [humidity],
        'co2_emission': [co2],
        'industrial_activity_index': [industry],
        'energy_price': [price],
        'month': [month],
        'day': [day]
    })

    prediction = model.predict(input_data)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("⚡ Predicted Energy", f"{prediction[0]:.2f}")

    with col2:
        st.metric("🌡️ Temperature Used", temp)

    # Bar chart
    fig_bar, ax = plt.subplots()
    ax.bar(["Predicted"], [prediction[0]])
    ax.set_ylabel("Energy Consumption")
    st.pyplot(fig_bar)

# ================= BASIC GRAPH =================
st.subheader("📈 Energy Consumption Trend")

fig_line_basic, ax2 = plt.subplots()
ax2.plot(df['energy_consumption'].head(100))
ax2.set_xlabel("Samples")
ax2.set_ylabel("Energy")
st.pyplot(fig_line_basic)

# ================= ADVANCED CHARTS =================
st.subheader("📊 Advanced Data Visualizations")

fig_line = px.line(df.head(200), y='energy_consumption',
                   title="Energy Consumption Over Time")
st.plotly_chart(fig_line, use_container_width=True)

fig_hist = px.histogram(df, x='energy_consumption', nbins=30,
                        title="Energy Consumption Distribution")
st.plotly_chart(fig_hist, use_container_width=True)

fig_scatter = px.scatter(df.head(200),
                         x='avg_temperature',
                         y='energy_consumption',
                         title="Temperature vs Energy Consumption")
st.plotly_chart(fig_scatter, use_container_width=True)