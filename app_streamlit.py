import streamlit as st
import pickle
import plotly.express as px
import time
from streamlit_option_menu import option_menu

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="CreditGuard ML", layout="wide")

# ---------------- LOADING SCREEN ----------------
with st.spinner("Loading CreditGuard ML..."):
    time.sleep(1)

# ---------------- CSS ----------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

.title {
    font-size: 60px;
    text-align: center;
    color: #FFD700;
    font-weight: bold;
}

.card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    box-shadow: 0px 0px 20px rgba(0,0,0,0.3);
}

.stButton>button {
    background: linear-gradient(90deg, #ff8c00, #ff2e63);
    color: white;
    font-size: 18px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))

# ---------------- NAVBAR ----------------
selected = option_menu(
    menu_title=None,
    options=["Overview", "EDA", "Models", "Results", "Live Demo"],
    icons=["house", "bar-chart", "cpu", "trophy", "play"],
    orientation="horizontal"
)

# ---------------- TITLE ----------------
st.markdown("<div class='title'>🚀 CreditGuard ML</div>", unsafe_allow_html=True)

# ================= OVERVIEW =================
if selected == "Overview":
    st.subheader("📊 Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.markdown("<div class='card'><h2>92.6%</h2><p>Accuracy</p></div>", unsafe_allow_html=True)
    col2.markdown("<div class='card'><h2>92.3%</h2><p>Precision</p></div>", unsafe_allow_html=True)
    col3.markdown("<div class='card'><h2>91.7%</h2><p>Recall</p></div>", unsafe_allow_html=True)
    col4.markdown("<div class='card'><h2>92.0%</h2><p>F1 Score</p></div>", unsafe_allow_html=True)

    st.divider()

    st.info("✔ Best Model: LightGBM | ✔ AUC Score: 0.97 | ✔ Fast Prediction")

# ================= EDA =================
elif selected == "EDA":
    st.subheader("📊 Exploratory Data Analysis")

    labels = ["Good Risk", "Bad Risk"]
    values = [2278, 1522]

    fig = px.pie(values=values, names=labels, title="Risk Distribution",
                 color_discrete_sequence=["green", "red"])
    st.plotly_chart(fig, use_container_width=True)

# ================= MODELS =================
elif selected == "Models":
    st.subheader("🧠 Model Comparison")

    models = ["RandomForest", "XGBoost", "LightGBM"]
    acc = [85, 91, 92]

    fig = px.bar(x=models, y=acc, color=acc, title="Model Accuracy")
    st.plotly_chart(fig, use_container_width=True)

# ================= RESULTS =================
elif selected == "Results":
    st.subheader("🏆 Final Results")

    st.image("confusion_matrix.png", caption="Confusion Matrix")

    st.metric("Accuracy", "92.6%")
    st.metric("F1 Score", "92.0%")

# ================= LIVE DEMO =================
elif selected == "Live Demo":
    st.subheader("🚀 Live Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 18, 100)
        income = st.number_input("Income", 1000, 200000)

        if st.button("Run Prediction"):
            pred = model.predict([[age, income]])
            prob = model.predict_proba([[age, income]])[0][1]

            st.progress(prob)

            if pred[0] == 1:
                st.error(f"High Risk ❌ ({prob*100:.2f}%)")
            else:
                st.success(f"Low Risk ✅ ({prob*100:.2f}%)")

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.image("confusion_matrix.png")
        st.markdown("</div>", unsafe_allow_html=True)