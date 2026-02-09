import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="Mobile Price Predictor",
    page_icon="📱",
    layout="wide"
)

# -----------------------------------
# TITLE SECTION
# -----------------------------------
st.markdown(
    """
    <h1 style='text-align:center; color:#4CAF50;'>📱 Mobile Price Prediction App</h1>
    <h4 style='text-align:center;'>KNN Algorithm • Real-World ML Project</h4>
    """,
    unsafe_allow_html=True
)

st.write("---")

# -----------------------------------
# DATASET
# -----------------------------------
data = {
    'battery_power': [800, 1000, 1200, 1500, 1800, 2200, 2500, 3000, 3500, 4000],
    'ram': [1, 2, 2, 3, 4, 4, 6, 8, 8, 12],
    'internal_memory': [8, 16, 16, 32, 64, 64, 128, 128, 256, 256],
    'camera': [5, 8, 12, 13, 16, 20, 32, 48, 50, 64],
    'price_range': [0, 0, 1, 1, 1, 2, 2, 3, 3, 3]
}

df = pd.DataFrame(data)

# -----------------------------------
# SIDEBAR INPUT
# -----------------------------------
st.sidebar.header("📊 Enter Mobile Features")

battery = st.sidebar.slider("🔋 Battery Power (mAh)", 800, 4000, 2500)
ram = st.sidebar.slider("🧠 RAM (GB)", 1, 12, 6)
memory = st.sidebar.slider("💾 Internal Memory (GB)", 8, 256, 128)
camera = st.sidebar.slider("📷 Camera (MP)", 5, 64, 32)

# -----------------------------------
# MODEL TRAINING
# -----------------------------------
X = df.drop("price_range", axis=1)
y = df["price_range"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# -----------------------------------
# PREDICTION
# -----------------------------------
user_data = [[battery, ram, memory, camera]]
user_data_scaled = scaler.transform(user_data)
prediction = knn.predict(user_data_scaled)

price_map = {
    0: "💸 Low Cost",
    1: "💰 Medium Cost",
    2: "💎 High Cost",
    3: "👑 Very High Cost"
}

# -----------------------------------
# MAIN DISPLAY
# -----------------------------------
st.subheader("📌 Prediction Result")

st.success(f"Predicted Price Category: **{price_map[prediction[0]]}**")

accuracy = accuracy_score(y_test, knn.predict(X_test))
st.info(f"🎯 Model Accuracy: **{accuracy * 100:.2f}%**")

# -----------------------------------
# CONFUSION MATRIX
# -----------------------------------
st.write("---")
st.subheader("📊 Model Performance")

cm = confusion_matrix(y_test, knn.predict(X_test))

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, cmap="Greens", fmt="d", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig)

# -----------------------------------
# FOOTER
# -----------------------------------
st.markdown(
    """
    <hr>
    <p style='text-align:center;'>
    🚀 Built with ❤️ using Streamlit & KNN <br>
    📘 Beginner Friendly • Real-World Project
    </p>
    """,
    unsafe_allow_html=True
)
