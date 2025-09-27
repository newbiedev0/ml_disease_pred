import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings("ignore")

@st.cache_resource
def load_model(file_path):
    try:
        model = joblib.load(file_path)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {file_path}. Please ensure it is in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model from {file_path}: {e}")
        return None

class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor
        self.bounds = {}
        self.feature_cols = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X) 
        self.feature_cols = X_df.columns.tolist() 
        
        for col in self.feature_cols:
            Q1 = X_df[col].quantile(0.25)
            Q3 = X_df[col].quantile(0.75)
            IQR = Q3 - Q1
            self.bounds[col] = {
                'lower': Q1 - (self.factor * IQR),
                'upper': Q3 + (self.factor * IQR)
            }
        return self

    def transform(self, X, y=None):
        X_df = pd.DataFrame(X, columns=self.feature_cols)
        
        for col in self.feature_cols:
            if col in self.bounds:
                lower_bound = self.bounds[col]['lower']
                upper_bound = self.bounds[col]['upper']
                
                X_df[col] = np.where(X_df[col] < lower_bound, lower_bound, X_df[col])
                X_df[col] = np.where(X_df[col] > upper_bound, upper_bound, X_df[col])
        
        return X_df.values 
    
    
parkinsons_model = load_model('parkinsons_gbc_pipeline.pkl')
ckd_model = load_model('pipeline_kidneydces.pkl')
liver_model = load_model('pipeline_logistic_liver.pkl')


def predict_parkinsons(model, data):
    feature_cols = [
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
        'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
        'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
        'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',
        'spread1', 'spread2', 'D2', 'PPE'
    ]
    input_df = pd.DataFrame([data], columns=feature_cols)
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][prediction]
    return prediction, proba

def predict_ckd(model, data):
    feature_cols = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 
                    'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 
                    'dm', 'cad', 'appet', 'pe', 'ane']
    input_df = pd.DataFrame([data], columns=feature_cols)
    prediction = model.predict(input_df)[0]
    if model.classes_[1] in ['ckd', 1]:
        proba = model.predict_proba(input_df)[0][1]
    else:
        proba = model.predict_proba(input_df)[0][0]
        
    return prediction, proba

def predict_liver(model, data):
    feature_cols = [
        'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 
        'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 
        'Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio'
    ]
    input_df = pd.DataFrame([data], columns=feature_cols)
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][prediction]
    return prediction, proba


def page_home():
    st.title("🩺 Medical Diagnostic Prediction Suite")
    st.markdown("""
        Welcome to the diagnostic suite! This application uses three distinct machine learning models 
        to provide preliminary predictions for **Parkinson's Disease**, **Chronic Kidney Disease (CKD)**, 
        and **Liver Disease**.
        
        **Select a test to begin.**
    """)

def page_parkinsons():
    st.title("🎙️ Parkinson's Disease Prediction")
    st.markdown("Enter the 22 required voice signal measurements.")
    
    if parkinsons_model is None:
        return

    with st.expander("Enter Voice Signal Measurements"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Core Frequency Measures**")
            fo = st.number_input("MDVP:Fo(Hz) (Avg Freq)", min_value=50.0, max_value=250.0, value=150.0)
            fhi = st.number_input("MDVP:Fhi(Hz) (High Freq)", min_value=100.0, max_value=600.0, value=200.0)
            flo = st.number_input("MDVP:Flo(Hz) (Low Freq)", min_value=50.0, max_value=250.0, value=100.0)
            
            st.markdown("**Jitter Measures**")
            jitter_per = st.number_input("MDVP:Jitter(%)", format="%.8f", value=0.005)
            jitter_abs = st.number_input("MDVP:Jitter(Abs)", format="%.8f", value=0.00005)
            rap = st.number_input("MDVP:RAP", format="%.8f", value=0.0035)
            ppq = st.number_input("MDVP:PPQ", format="%.8f", value=0.004)
            jitter_ddp = st.number_input("Jitter:DDP", format="%.8f", value=0.01)

        with col2:
            st.markdown("**Shimmer Measures**")
            mdvp_shimmer = st.number_input("MDVP:Shimmer", format="%.8f", value=0.03)
            mdvp_shimmer_db = st.number_input("MDVP:Shimmer(dB)", format="%.8f", value=0.3)
            shimmer_apq3 = st.number_input("Shimmer:APQ3", format="%.8f", value=0.012)
            shimmer_apq5 = st.number_input("Shimmer:APQ5", format="%.8f", value=0.018)
            mdvp_apq = st.number_input("MDVP:APQ", format="%.8f", value=0.02)
            shimmer_dda = st.number_input("Shimmer:DDA", format="%.8f", value=0.02)

        with col3:
            st.markdown("**Noise & Non-Linear Measures**")
            nhr = st.number_input("NHR (Noise/Harmonics)", format="%.5f", value=0.015)
            hnr = st.number_input("HNR (Harmonics/Noise)", value=25.0)
            rpde = st.number_input("RPDE", format="%.6f", value=0.5)
            dfa = st.number_input("DFA", format="%.6f", value=0.7)
            
            st.markdown("**Non-Linear Dynamic Complexity**")
            spread1 = st.number_input("spread1", format="%.6f", value=-4.5)
            spread2 = st.number_input("spread2", format="%.6f", value=0.2)
            d2 = st.number_input("D2", format="%.6f", value=2.5)
            ppe = st.number_input("PPE", format="%.6f", value=0.2)

    data = [fo, fhi, flo, jitter_per, jitter_abs, rap, ppq, jitter_ddp, mdvp_shimmer, mdvp_shimmer_db, 
            shimmer_apq3, shimmer_apq5, mdvp_apq, shimmer_dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]

    if st.button("Predict Parkinson's Status"):
        prediction, proba = predict_parkinsons(parkinsons_model, data)
        if prediction == 1:
            st.error(f"🔴 Prediction: Likely to have Parkinson's Disease (Confidence: {proba:.2%})")
        else:
            st.success(f"🟢 Prediction: Not likely to have Parkinson's Disease (Confidence: {proba:.2%})")
        


def page_ckd():
    st.title("🩸 Chronic Kidney Disease Prediction")
    st.markdown("Enter 24 clinical parameters.")

    if ckd_model is None:
        return

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Numeric Inputs")
        age = st.number_input("Age", min_value=1, max_value=100, value=45)
        bp = st.number_input("Blood Pressure (bp)", value=80.0)
        sg = st.number_input("Specific Gravity (sg)", format="%.3f", value=1.015)
        al = st.number_input("Albumin (al)", min_value=0, max_value=5, value=1)
        su = st.number_input("Sugar (su)", min_value=0, max_value=5, value=0)
        bgr = st.number_input("Blood Glucose Random (bgr)", value=120.0)
        bu = st.number_input("Blood Urea (bu)", value=30.0)
        sc = st.number_input("Serum Creatinine (sc)", value=1.2)
        sod = st.number_input("Sodium (sod)", value=140.0)
        pot = st.number_input("Potassium (pot)", value=4.0)
        hemo = st.number_input("Hemoglobin (hemo)", value=14.0)
        pcv = st.number_input("Packed Cell Volume (pcv)", value=42.0)
        wc = st.number_input("White Blood Cell Count (wc)", value=7000)
        rc = st.number_input("Red Blood Cell Count (rc)", value=5.0)

    with col2:
        st.subheader("Categorical Inputs")
        rbc = st.selectbox("Red Blood Cells (rbc)", ['normal', 'abnormal'])
        pc = st.selectbox("Pus Cell (pc)", ['normal', 'abnormal'])
        pcc = st.selectbox("Pus Cell Clumps (pcc)", ['present', 'notpresent'])
        ba = st.selectbox("Bacteria (ba)", ['present', 'notpresent'])
        htn = st.selectbox("Hypertension (htn)", ['yes', 'no'])
        dm = st.selectbox("Diabetes Mellitus (dm)", ['yes', 'no'])
        cad = st.selectbox("Coronary Artery Disease (cad)", ['yes', 'no'])
        appet = st.selectbox("Appetite (appet)", ['good', 'poor'])
        pe = st.selectbox("Pedal Edema (pe)", ['yes', 'no'])
        ane = st.selectbox("Anemia (ane)", ['yes', 'no'])

    data = [age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane]

    if st.button("Predict CKD Status"):
        prediction, proba = predict_ckd(ckd_model, data)
        if prediction in ['ckd', 1]:
            st.error(f"🔴 Prediction: Likely to have Chronic Kidney Disease (Confidence: {proba:.2%})")
        else:
            st.success(f"🟢 Prediction: Not likely to have Chronic Kidney Disease (Confidence: {1 - proba:.2%})")
        


def page_liver():
    st.title("💛 Liver Disease Prediction")
    st.markdown("Enter 10 blood test and demographic parameters.")

    if liver_model is None:
        return

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=100, value=45)
        gender = st.selectbox("Gender", ['Male', 'Female'])
        total_bilirubin = st.number_input("Total Bilirubin", value=1.2)
        direct_bilirubin = st.number_input("Direct Bilirubin", value=0.3)
        alk_phos = st.number_input("Alkaline Phosphotase", value=200)

    with col2:
        ala_trans = st.number_input("Alamine Aminotransferase", value=20)
        asp_trans = st.number_input("Aspartate Aminotransferase", value=20)
        total_protiens = st.number_input("Total Protiens", value=7.0)
        albumin = st.number_input("Albumin", value=3.5)
        a_g_ratio = st.number_input("Albumin and Globulin Ratio", format="%.3f", value=1.0)
    
    data = [age, gender, total_bilirubin, direct_bilirubin, alk_phos, 
            ala_trans, asp_trans, total_protiens, albumin, a_g_ratio]

    if st.button("Predict Liver Disease Status"):
        prediction, proba = predict_liver(liver_model, data)
        if prediction == 1:
            st.error(f"🔴 Prediction: Likely to have Liver Disease (Confidence: {proba:.2%})")
        else:
            st.success(f"🟢 Prediction: Not likely to have Liver Disease (Confidence: {1 - proba:.2%})")
        


def main():
    st.set_page_config(layout="wide")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Test", ["Home", "Parkinson's Disease", "CKD", "Liver Disease"])

    if page == "Home":
        page_home()
    elif page == "Parkinson's Disease":
        page_parkinsons()
    elif page == "CKD":
        page_ckd()
    elif page == "Liver Disease":
        page_liver()

if __name__ == "__main__":
    main()