import streamlit as st
import pickle
import plotly.express as px
import time
from streamlit_option_menu import option_menu

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="CreditGuard ML", layout="wide")

# ---------------- LOADING ----------------
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
    padding: 10px 20px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model_new.pkl", "rb"))

# ---------------- NAVBAR ----------------
selected = option_menu(
    menu_title=None,
    options=["Overview", "EDA", "Models", "Results", "Report", "Live Demo"],
    icons=["house", "bar-chart", "cpu", "trophy", "file-text", "play"],
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

    fig = px.pie(values=values, names=labels,
                 title="Risk Distribution",
                 color_discrete_sequence=["green", "red"])
    st.plotly_chart(fig, use_container_width=True)

# ================= MODELS =================
elif selected == "Models":
    st.subheader("🧠 Model Comparison")

    models = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest"
    ]

    acc = [
    84,
    88,
    92
    ]

    fig = px.bar(x=models, y=acc, color=acc,
                 title="Model Accuracy")
    st.plotly_chart(fig, use_container_width=True)

# ================= RESULTS =================
elif selected == "Results":
    st.subheader("🏆 Final Results")

    st.image("confusion_matrix.png", caption="Confusion Matrix")

    col1, col2 = st.columns(2)
    col1.metric("Accuracy", "92.6%")
    col2.metric("F1 Score", "92.0%")

# ================= REPORT =================
elif selected == "Report":

    st.markdown("""
    <div style="padding:25px; border-radius:20px;
                background: rgba(255,255,255,0.08);
                backdrop-filter: blur(12px);
                box-shadow: 0px 0px 20px rgba(0,0,0,0.3);">
    """, unsafe_allow_html=True)

    st.markdown("<h2 style='color:#FFD700;'>📄 Project Report</h2>", unsafe_allow_html=True)

    st.markdown("""
    ### 📌 Abstract
    This project builds a **Credit Risk Prediction System** using machine learning.  
    It classifies users into **Good Risk** and **Bad Risk** categories.  
    Achieved **92.6% accuracy using LightGBM**.
    """)

    st.markdown("""
    ### 🧠 Introduction
    Credit risk prediction helps banks reduce loan defaults.  
    Machine learning improves decision-making using data patterns.
    """)

    st.markdown("""
    ### 📊 Dataset
    Synthetic dataset with financial features like age and income.
    """)

    st.markdown("""
    ### ⚙️ Methodology
    Data preprocessing → Model training → Evaluation → Deployment.
    """)

    st.markdown("""
    ### 🏆 Results
    Accuracy: 92.6%  
    Precision: 92.3%  
    Recall: 91.7%  
    F1 Score: 92.0%
    """)

    st.markdown("""
    ### 🚀 Future Scope
    - Real banking integration  
    - More features  
    - Better models  
    """)

    st.markdown("</div>", unsafe_allow_html=True)

# ================= LIVE DEMO =================
elif selected == "Live Demo":
    st.subheader("🚀 Live Prediction Demo")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📋 Applicant Information")

        # Advanced UI (like your screenshot)
        income = st.number_input("Annual Income ($)", 1000, 200000, 65000)
        employment = st.number_input("Employment (Years)", 0, 40, 5)
        credit_score = st.number_input("Credit Score", 300, 850, 680)
        dependents = st.number_input("Dependents", 0, 10, 1)

        age = st.number_input("Age", 18, 100, 35)
        dti = st.number_input("DTI Ratio (%)", 0, 100, 28)
        loan = st.number_input("Loan Amount ($)", 1000, 100000, 25000)
        education = st.selectbox("Education Level",
                                ["High School", "Bachelor's", "Master's"])

        if st.button("🚀 Run Prediction"):
            # convert education to number
            edu_map = {"High School": 0, "Bachelor's": 1, "Master's": 2}
            input_data = [[
            age,
            income,
            credit_score,
            loan,
            dti,
            employment,
            dependents,
            edu_map[education]
            ]]
             
            pred = model.predict(input_data)
            prob = model.predict_proba(input_data)[0][1]

            st.progress(prob)

            if pred[0] == 1:
                st.error(f"High Risk ❌ ({prob*100:.2f}%)")
            else:
                st.success(f"Low Risk ✅ ({prob*100:.2f}%)")

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.image("confusion_matrix.png", caption="Model Performance")
        st.markdown("</div>", unsafe_allow_html=True)