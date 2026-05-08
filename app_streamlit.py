import streamlit as st
import pickle
import plotly.express as px
import pandas as pd
import time
from streamlit_option_menu import option_menu

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="CreditGuard ML",
    layout="wide"
)

# ---------------- LOADING SCREEN ----------------
with st.spinner("Loading CreditGuard ML..."):
    time.sleep(1)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

/* TITLE */
.title {
    font-size: 65px;
    text-align: center;
    color: #FFD700;
    font-weight: bold;
    margin-bottom: 10px;
}

/* SUBTITLE */
.subtitle {
    text-align: center;
    color: #d3d3d3;
    font-size: 18px;
    margin-bottom: 25px;
}

/* CARDS */
.card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 20px;
    text-align: center;
    box-shadow: 0px 0px 20px rgba(0,0,0,0.3);
}

/* BUTTONS */
.stButton>button {
    background: linear-gradient(90deg, #ff8c00, #ff2e63);
    color: white;
    font-size: 18px;
    border-radius: 12px;
    padding: 12px 24px;
    border: none;
    width: 100%;
}

/* INPUT BOXES */
.stNumberInput input {
    background-color: rgba(255,255,255,0.06);
    color: white;
}

/* SELECT BOX */
.stSelectbox div[data-baseweb="select"] {
    color: black;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model_new.pkl", "rb"))

# ---------------- NAVBAR ----------------
selected = option_menu(
    menu_title=None,
    options=[
        "Overview",
        "EDA",
        "Models",
        "Results",
        "Report",
        "Live Demo"
    ],
    icons=[
        "house",
        "bar-chart",
        "cpu",
        "trophy",
        "file-text",
        "play"
    ],
    orientation="horizontal"
)

# ---------------- TITLE ----------------
st.markdown(
    "<div class='title'>🚀 CreditGuard ML</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Applied Machine Learning Project | Credit Risk Prediction System</div>",
    unsafe_allow_html=True
)

# =====================================================
# OVERVIEW
# =====================================================
if selected == "Overview":

    st.subheader("📊 Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.markdown("""
    <div class='card'>
    <h2>92%</h2>
    <p>Accuracy</p>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown("""
    <div class='card'>
    <h2>91%</h2>
    <p>Precision</p>
    </div>
    """, unsafe_allow_html=True)

    col3.markdown("""
    <div class='card'>
    <h2>90%</h2>
    <p>Recall</p>
    </div>
    """, unsafe_allow_html=True)

    col4.markdown("""
    <div class='card'>
    <h2>91%</h2>
    <p>F1 Score</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.info("""
    ✔ Multiple ML Models Used  
    ✔ Real-Time Prediction System  
    ✔ Streamlit Cloud Deployment  
    ✔ Random Forest Selected As Best Model
    """)

# =====================================================
# EDA
# =====================================================
elif selected == "EDA":

    st.subheader("📊 Exploratory Data Analysis")

    labels = ["Low Risk", "High Risk"]
    values = [620, 380]

    fig = px.pie(
        values=values,
        names=labels,
        title="Credit Risk Distribution",
        color_discrete_sequence=["green", "red"]
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ### 📌 Observations
    - Most applicants belong to low-risk category
    - High DTI ratio increases risk
    - Low credit score strongly affects predictions
    """)

# =====================================================
# MODELS
# =====================================================
elif selected == "Models":

    st.subheader("🧠 Machine Learning Models")

    st.markdown("""
    This project compares multiple machine learning algorithms
    to identify the best-performing model.
    """)

    models = [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ]

    accuracies = [
        84,
        88,
        92
    ]

    fig = px.bar(
        x=models,
        y=accuracies,
        color=accuracies,
        text=accuracies,
        title="Model Accuracy Comparison"
    )

    fig.update_layout(
        title_x=0.3,
        xaxis_title="Models",
        yaxis_title="Accuracy (%)"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.success("""
    🏆 Random Forest achieved highest performance
    and was selected as the final deployed model.
    """)

# =====================================================
# RESULTS
# =====================================================
elif selected == "Results":

    st.subheader("🏆 Final Results & Evaluation")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            "confusion_matrix.png",
            caption="Confusion Matrix"
        )

    with col2:
        st.image(
            "roc_curve.png",
            caption="ROC Curve"
        )

    st.divider()

    st.subheader("📊 Evaluation Metrics")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Accuracy", "92%")
    c2.metric("Precision", "91%")
    c3.metric("Recall", "90%")
    c4.metric("F1 Score", "91%")

    st.divider()

    st.subheader("⭐ Feature Importance")

    st.image(
        "feature_importance.png",
        caption="Feature Importance Graph"
    )

# =====================================================
# REPORT
# =====================================================
elif selected == "Report":

    st.markdown("""
    <div style="
        padding:25px;
        border-radius:20px;
        background: rgba(255,255,255,0.08);
        backdrop-filter: blur(12px);
        box-shadow: 0px 0px 20px rgba(0,0,0,0.3);
    ">
    """, unsafe_allow_html=True)

    st.markdown("""
    ## 📄 Project Report

    ### 📌 Introduction
    CreditGuard ML is a machine learning-based system
    developed for predicting credit risk of loan applicants.

    ### 🎯 Objective
    To reduce loan default risks using machine learning
    algorithms and intelligent prediction systems.

    ### 📊 Dataset
    The project uses financial and demographic features:
    - Age
    - Income
    - Credit Score
    - Loan Amount
    - DTI Ratio
    - Employment Years

    ### 🧠 Models Used
    - Logistic Regression
    - Decision Tree
    - Random Forest

    ### ⚙️ Technologies
    - Python
    - Streamlit
    - Scikit-learn
    - Plotly
    - Pandas

    ### 🏆 Final Result
    Random Forest achieved highest accuracy and
    delivered best classification performance.

    ### 🚀 Deployment
    Application deployed using Streamlit Cloud.
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# LIVE DEMO
# =====================================================
elif selected == "Live Demo":

    st.subheader("🚀 Live Prediction Demo")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 📋 Applicant Information")

        income = st.number_input(
            "Annual Income ($)",
            1000,
            200000,
            65000
        )

        employment = st.number_input(
            "Employment (Years)",
            0,
            40,
            5
        )

        credit_score = st.number_input(
            "Credit Score",
            300,
            850,
            680
        )

        dependents = st.number_input(
            "Dependents",
            0,
            10,
            1
        )

        age = st.number_input(
            "Age",
            18,
            100,
            35
        )

        dti = st.number_input(
            "DTI Ratio (%)",
            0,
            100,
            28
        )

        loan = st.number_input(
            "Loan Amount ($)",
            1000,
            100000,
            25000
        )

        education = st.selectbox(
            "Education Level",
            ["High School", "Bachelor's", "Master's"]
        )

        if st.button("🚀 Run Prediction"):

            edu_map = {
                "High School": 0,
                "Bachelor's": 1,
                "Master's": 2
            }

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

            st.progress(float(prob))

            if pred[0] == 1:
                st.error(
                    f"High Risk ❌ ({prob*100:.2f}%)"
                )
            else:
                st.success(
                    f"Low Risk ✅ ({prob*100:.2f}%)"
                )

    with col2:

        st.markdown("""
        <div class='card'>
        """, unsafe_allow_html=True)

        st.image(
            "feature_importance.png",
            caption="Feature Importance"
        )

        st.markdown("""
        </div>
        """, unsafe_allow_html=True)