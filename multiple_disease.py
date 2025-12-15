import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.set_page_config(page_title="Multiple Disease Prediction", layout="wide")

# -------------------------------------------------------------
# Load ML Models & Scalers
# -------------------------------------------------------------
kidney_model = pickle.load(open("kidney_model.pkl", "rb"))
kidney_scaler = pickle.load(open("kidney_scaler.pkl", "rb"))

liver_model = pickle.load(open("liver_model.pkl", "rb"))
liver_scaler = pickle.load(open("liver_scaler.pkl", "rb"))

parkinson_model = pickle.load(open("parkinson_model.pkl", "rb"))
parkinson_scaler = pickle.load(open("parkinson_scaler.pkl", "rb"))

# -------------------------------------------------------------
# Sidebar Navigation
# -------------------------------------------------------------
st.sidebar.title("🧠 Multiple Disease Prediction System")
menu = st.sidebar.radio(
    "Choose Prediction",
    ["Kidney Disease", "Liver Disease", "Parkinson's Disease"]
)

# -------------------------------------------------------------
# KIDNEY DISEASE PAGE
# -------------------------------------------------------------
if menu == "Kidney Disease":
    st.title("🩺 Kidney Disease Prediction")

    # DATAFRAME ONLY (No extra columns)
    with st.expander("📘 Kidney Dataset Preview", expanded=False):
        df_kidney = pd.read_csv("D:\VS Code\kidney_disease_cleaned.csv") 
        st.dataframe(df_kidney) 

    # FORM INSIDE EXPANDER
    with st.expander("📝 Enter Patient Details", expanded=False):

        # ------------------- ROW 1 -------------------
        r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns(5)

        age = r1c1.number_input("Age")
        bp = r1c2.number_input("Blood Pressure")
        sg = r1c3.number_input("Specific Gravity")
        al = r1c4.number_input("Albumin")
        rbc = r1c5.selectbox("Red Blood Cells", ["Normal", "Abnormal"])

        # ------------------- ROW 2 -------------------
        r2c1, r2c2, r2c3, r2c4, r2c5 = st.columns(5)

        pc = r2c1.selectbox("Pus Cell", ["Normal", "Abnormal"])
        bgr = r2c2.number_input("Blood Glucose Random")
        bu = r2c3.number_input("Blood Urea")
        sc = r2c4.number_input("Serum Creatinine")
        sod = r2c5.number_input("Sodium")

        # ------------------- ROW 3 -------------------
        r3c1, r3c2, r3c3, r3c4, r3c5 = st.columns(5)

        hemo = r3c1.number_input("Hemoglobin")
        pcv = r3c2.number_input("Packed Cell Volume")
        wc = r3c3.number_input("White Blood Cell Count")
        rc = r3c4.number_input("Red Blood Cell Count")
        htn = r3c5.selectbox("Hypertension", ["No", "Yes"])

        # ------------------- ROW 4 -------------------
        r4c1, r4c2, r4c3, r4c4, r4c5 = st.columns(5)

        dm = r4c1.selectbox("Diabetes Mellitus", ["No", "Yes"])
        appet = r4c2.selectbox("Appetite", ["Good", "Poor"])
        pe = r4c3.selectbox("Pedal Edema", ["No", "Yes"])

        # empty cols for alignment
        r4c4.write("")
        r4c5.write("")

    # Encoding
    rbc = 1 if rbc == "Abnormal" else 0
    pc = 1 if pc == "Abnormal" else 0
    htn = 1 if htn == "Yes" else 0
    dm = 1 if dm == "Yes" else 0
    appet = 1 if appet == "Good" else 0
    pe = 1 if pe == "Yes" else 0

    # Prediction Button
    if st.button("🔍 Predict Kidney Disease"):
        features = np.array([[age, bp, sg, al, rbc, pc, bgr, bu, sc,
                              sod, hemo, pcv, wc, rc, htn, dm, appet, pe]])

        scaled = kidney_scaler.transform(features)
        result = kidney_model.predict(scaled)

        if result[0] == 1:
            st.error("⚠️ High Probability of Kidney Disease!")
        else:
            st.success("✅ No Kidney Disease Detected.")



# -------------------------------------------------------------
# LIVER DISEASE PAGE
# -------------------------------------------------------------
elif menu == "Liver Disease":
    st.title("🩸 Liver Disease Prediction")

    with st.expander("📘 Liver Dataset Preview", expanded=False):
        df_liver = pd.read_csv("D:\Guvi-Projects\Multiple Disease Project - 4\indian_liver_patient - indian_liver_patient.csv")
        st.dataframe(df_liver)

    with st.expander("📝 Enter Patient Details", expanded=False):

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age")
            gender = st.selectbox("Gender", ["Male", "Female"])
            tb = st.number_input("Total Bilirubin")
            db = st.number_input("Direct Bilirubin")

        with col2:
            alkphos = st.number_input("Alkaline Phosphotase")
            sgot = st.number_input("Aspartate Aminotransferase")
            sgpt = st.number_input("Alamine Aminotransferase")
            tp = st.number_input("Total Proteins")

        with col3:
            alb = st.number_input("Albumin")
            agr = st.number_input("Albumin/Globulin Ratio")

    gender = 1 if gender == "Male" else 0

    if st.button("🔍 Predict Liver Disease"):
        features = np.array([[age, gender, tb, db, alkphos,
                              sgpt, sgot, tp, alb, agr]])

        scaled = liver_scaler.transform(features)
        result = liver_model.predict(scaled)

        if result[0] == 1:
            st.error("⚠️ High Probability of Liver Disease!")
        else:
            st.success("✅ No Liver Disease Detected.")



# -------------------------------------------------------------
# PARKINSON'S PAGE
# -------------------------------------------------------------
elif menu == "Parkinson's Disease":
    st.title("🧠 Parkinson’s Disease Prediction")

    with st.expander("📘 Parkinson Dataset Preview", expanded=False):
        df_parkinson = pd.read_csv("D:\Guvi-Projects\Multiple Disease Project - 4\parkinsons - parkinsons.csv")
        st.dataframe(df_parkinson)

    with st.expander("🎤 Enter Patient Voice Parameters", expanded=False):

        tab1, tab2, tab3 = st.tabs(["Jitter Features", "Shimmer Features", "Nonlinear Features"])

        with tab1:
            fo = st.number_input("MDVP:Fo(Hz)")
            fhi = st.number_input("MDVP:Fhi(Hz)")
            flo = st.number_input("MDVP:Flo(Hz)")
            jitter = st.number_input("MDVP:Jitter(%)")
            jitter_abs = st.number_input("MDVP:Jitter(Abs)")
            rap = st.number_input("MDVP:RAP")
            ppq = st.number_input("MDVP:PPQ")
            ddp = st.number_input("Jitter:DDP")

        with tab2:
            shimmer = st.number_input("MDVP:Shimmer")
            shimmer_db = st.number_input("MDVP:Shimmer(dB)")
            apq3 = st.number_input("Shimmer:APQ3")
            apq5 = st.number_input("Shimmer:APQ5")
            mdvp_apq = st.number_input("MDVP:APQ")
            dda = st.number_input("Shimmer:DDA")

        with tab3:
            nhr = st.number_input("NHR")
            hnr = st.number_input("HNR")
            rpde = st.number_input("RPDE")
            dfa = st.number_input("DFA")
            spread1 = st.number_input("Spread1")
            spread2 = st.number_input("Spread2")
            d2 = st.number_input("D2")
            ppe = st.number_input("PPE")

    if st.button("🔍 Predict Parkinson’s Disease"):
        features = np.array([[fo, fhi, flo, jitter, jitter_abs, rap, ppq, ddp,
                              shimmer, shimmer_db, apq3, apq5, mdvp_apq, dda,
                              nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]])

        scaled = parkinson_scaler.transform(features)
        result = parkinson_model.predict(scaled)

        if result[0] == 1:
            st.error("⚠️ High Probability of Parkinson's Disease!")
        else:
            st.success("✅ No Parkinson’s Disease Detected.")
