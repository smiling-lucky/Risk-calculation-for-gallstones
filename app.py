import streamlit as st
try:
    import pkg_resources
except ImportError:
    pass
# Monkey patch for xgboost 1.6.2 issue with pkg_resources in some environments
import sys
if 'pkg_resources' not in sys.modules:
    import setuptools.pkg_resources as pkg_resources
    sys.modules['pkg_resources'] = pkg_resources

import pandas as pd

print("App is starting...")

import numpy as np
import joblib
import json
import os

# Page configuration
st.set_page_config(
    page_title="Gallbladder Stone Risk Calculator",
    page_icon="🏥",
    layout="centered"
)

# Load resources
@st.cache_resource
def load_model():
    # Get the directory where the current script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "best_xgboost_calibrated_model.pkl")
    
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        return None
    return joblib.load(model_path)

@st.cache_data
def load_stats():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    stats_path = os.path.join(current_dir, "feature_stats.json")
    
    if not os.path.exists(stats_path):
        st.error(f"Stats file not found at: {stats_path}")
        return None
    with open(stats_path, 'r', encoding='utf-8') as f:
        return json.load(f)

model = load_model()
stats = load_stats()

# Custom CSS styles
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    .result-container {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    .risk-low { color: #28a745; font-weight: bold; }
    .risk-high { color: #dc3545; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

def main():
    st.title("🏥 Gallbladder Stone Risk Calculator")
    st.markdown("---")
    st.markdown("Please enter your physiological indicators below. The system will estimate your risk of gallbladder stones based on our machine learning model.")

    if model is None or stats is None:
        return

    # Create input form
    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            # Age
            age_min = int(stats['Age']['min'])
            age_max = int(stats['Age']['max'])
            age_default = int(stats['Age']['mean'])
            if age_default < 60: age_default = 60
            age = st.number_input("Age", min_value=0, max_value=120, value=age_default, step=1, format="%d", help="Please enter your age (60-120)")
            if age < 60:
                st.error("Age must be between 60 and 120.")
            
            # Gender
            # Based on stats: 1 (Male), 2 (Female)
            gender_options = {1: "Male", 2: "Female"}
            gender_val = st.selectbox("Gender", options=[1, 2], format_func=lambda x: gender_options[x], help="Please select your gender")

            # WHtR
            whtr_mean = stats['WHtR']['mean']
            whtr = st.number_input("Waist-to-Height Ratio (WHtR)", min_value=0.0, max_value=2.0, value=whtr_mean, step=0.01, help="Formula: WHtR = WC / Height\nWhere:\n- WC = Waist Circumference (cm)\n- Height = Height (cm)")

            # LDL
            ldl_mean = stats['LDL']['mean']
            ldl = st.number_input("Low-Density Lipoprotein (LDL)", min_value=0.0, max_value=500.0, value=ldl_mean, step=1.0, help="Unit: mg/dL")

        with col2:
            # Eosinophils
            eos_mean = stats['Eosinophils']['mean']
            eos = st.number_input("Eosinophils", min_value=0.0, max_value=10.0, value=eos_mean, step=0.01, help="Unit: 10^9/L")

            # MLR
            mlr_mean = stats['MLR']['mean']
            mlr = st.number_input("Monocyte-to-Lymphocyte Ratio (MLR)", min_value=0.0, max_value=10.0, value=mlr_mean, step=0.01, help="Formula: MLR = Monocyte Count / Lymphocyte Count")

            # ABSI
            absi_mean = stats['ABSI']['mean']
            absi = st.number_input("A Body Shape Index (ABSI)", min_value=0.0, max_value=0.2, value=absi_mean, format="%.4f", step=0.0001, help="Formula: ABSI = WC / (BMI^(2/3) * Height^(1/2))\nWhere:\n- WC = Waist Circumference (m)\n- BMI = Body Mass Index (kg/m^2)\n- Height = Height (m)")

    if st.button("Calculate Risk Probability"):
        # Preprocess inputs
        # 1. Gender encoding
        # Gender_2 = 1 if Female (2), 0 if Male (1)
        gender_2 = 1 if gender_val == 2 else 0

        # 2. Standardization
        # Formula: (x - mean) / std
        def standardize(val, feature_name):
            mean = stats[feature_name]['mean']
            std = stats[feature_name]['std']
            return (val - mean) / std

        age_std = standardize(age, 'Age')
        whtr_std = standardize(whtr, 'WHtR')
        ldl_std = standardize(ldl, 'LDL')
        eos_std = standardize(eos, 'Eosinophils')
        mlr_std = standardize(mlr, 'MLR')
        absi_std = standardize(absi, 'ABSI')

        # 3. Construct feature vector
        # Order: ['Age', 'WHtR', 'LDL', 'Eosinophils', 'MLR', 'ABSI', 'Gender_2']
        input_data = pd.DataFrame([[
            age_std,
            whtr_std,
            ldl_std,
            eos_std,
            mlr_std,
            absi_std,
            gender_2
        ]], columns=['Age', 'WHtR', 'LDL', 'Eosinophils', 'MLR', 'ABSI', 'Gender_2'])

        # Predict
        try:
            # Get probability of positive class (1)
            prob = model.predict_proba(input_data)[0][1]
            prob_percent = prob * 100
            
            # Optimal threshold from optimization results
            optimal_threshold = 0.1480 

            st.markdown("---")
            st.subheader("Evaluation Results")
            
            # Display result with color coding based on optimal threshold
            if prob < optimal_threshold:
                risk_level = "Low Risk"
                risk_class = "risk-low"
            else:
                risk_level = "High Risk"
                risk_class = "risk-high"

            st.markdown(f"""
                <div class="result-container">
                    <h3 style='text-align: center;'>Estimated Gallbladder Stone Risk: <span class='{risk_class}'>{prob_percent:.2f}%</span></h3>
                    <p style='text-align: center; font-size: 1.2em;'>Risk Level: <span class='{risk_class}'>{risk_level}</span></p>
                </div>
            """, unsafe_allow_html=True)

            if prob >= optimal_threshold:
                st.warning(f"⚠️ **Warning:** Your estimated risk exceeds {optimal_threshold:.2%}. It is highly recommended to visit a hospital for a comprehensive medical evaluation.")

            # Medical advice
            st.info("""
            **💡 Note:**
            - This calculator is based on a machine learning model for academic research and risk reference only.
            - The results do not replace professional clinical diagnosis by a physician.
            - If your risk assessment is high, please consult a medical professional for a comprehensive examination.
            """)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray; font-size: 0.8em;'>"
        "© 2026 Gallbladder Stone Risk Assessment System | For Academic Reference Only"
        "</div>", 
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
