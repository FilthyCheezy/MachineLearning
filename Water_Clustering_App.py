import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering

# --- Page Setup ---
st.set_page_config(page_title="Water Quality Clustering", layout="centered")

st.markdown("""
    <style>
    .main > div {
        max-width: 1550px;
        margin: auto;
        background-color: #e4eefd;
        border-radius: 12px;
    }
    .prediction-text {
        text-align: center;
        font-size: 1.4em;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>💧 Water Quality Clustering System</h1>", unsafe_allow_html=True)

# --- Session state ---
if "predicted_cluster" not in st.session_state:
    st.session_state.predicted_cluster = None

# --- Prediction Display (with columns) ---
def display_water_quality(prediction=None):
    col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 1, 1, 2, 1, 1, 1])
    with col4:
        if prediction == "Clean":
            st.image("2.png", width = 10, use_container_width=True)
            st.markdown("<div class='prediction-text'>🔵 Predicted Water Quality: <strong>Clean</strong></div>", unsafe_allow_html=True)
        elif prediction == "Dirty":
            st.image("3.png", width = 10, use_container_width=True)
            st.markdown("<div class='prediction-text'>🔴 Predicted Water Quality: <strong>Dirty</strong></div>", unsafe_allow_html=True)
        else:
            st.image("1.png", width = 10, use_container_width=True)
            st.markdown("<div class='prediction-text'>⚠️ <strong>Unable to classify</strong></div>", unsafe_allow_html=True)

# Show prediction
display_water_quality(st.session_state.predicted_cluster)

# --- Form UI ---
with st.form("input_form"):
    st.subheader("🧪 Adjust Water Quality Parameters")

    def slider_with_input(label, min_val, max_val, default_val, step=0.1):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"<div style='font-size:1.1em; font-weight:600;'>{label}</div>", unsafe_allow_html=True)
            val_slider = st.slider("", min_value=min_val, max_value=max_val, value=default_val, step=step)
        with col2:
            val_input = st.number_input(f"{label} input", min_value=min_val, max_value=max_val, value=val_slider, step=step, label_visibility="collapsed")
        return val_input

    col1, col2 = st.columns(2)
    with col1:
        Temp = slider_with_input("Temp (°C)", 0.0, 50.0, 25.0)
        DO = slider_with_input("D.O. (mg/l)", 0.0, 15.0, 7.0)
        PH = slider_with_input("PH", 0.0, 14.0, 7.0)
        Conductivity = slider_with_input("CONDUCTIVITY (µmhos/cm)", 0.0, 2000.0, 500.0)
    with col2:
        BOD = slider_with_input("B.O.D. (mg/l)", 0.0, 30.0, 5.0)
        Nitrate = slider_with_input("NITRATENAN N+ NITRITENANN (mg/l)", 0.0, 20.0, 2.0)
        FecalColiform = slider_with_input("FECAL COLIFORM (MPN/100ml)", 0.0, 10000.0, 1000.0)
        TotalColiform = slider_with_input("TOTAL COLIFORM (MPN/100ml)Mean", 0.0, 50000.0, 2000.0)

    submitted = st.form_submit_button("🔍 Predict Water Quality")

# --- Prediction Logic ---
if submitted:
    user_input = pd.DataFrame([[Temp, DO, PH, Conductivity, BOD, Nitrate, FecalColiform, TotalColiform]],
        columns=['Temp', 'D.O. (mg/l)', 'PH', 'CONDUCTIVITY (µmhos/cm)',
                 'B.O.D. (mg/l)', 'NITRATENAN N+ NITRITENANN (mg/l)',
                 'FECAL COLIFORM (MPN/100ml)', 'TOTAL COLIFORM (MPN/100ml)Mean'])

    clean_center = np.array([25.0, 7.0, 7.2, 300.0, 2.0, 1.0, 100.0, 1000.0])
    dirty_center = np.array([30.0, 4.5, 6.5, 800.0, 6.0, 3.0, 5000.0, 20000.0])
    reference = pd.DataFrame([clean_center, dirty_center], columns=user_input.columns)

    combined = pd.concat([reference, user_input], ignore_index=True)
    scaler = StandardScaler()
    scaled_combined = scaler.fit_transform(combined)
    pca = PCA(n_components=0.90)
    pca_combined = pca.fit_transform(scaled_combined)

    model = SpectralClustering(
        n_clusters=2,
        gamma=1.0,
        affinity='rbf',
        assign_labels='kmeans',
        random_state=42
    )
    labels = model.fit_predict(pca_combined)

    reference_labels = labels[:2].tolist()
    user_label = labels[-1]

    if len(set(reference_labels)) == 2:
        if reference.loc[0, 'D.O. (mg/l)'] > reference.loc[1, 'D.O. (mg/l)']:
            label_map = {reference_labels[0]: 'Clean', reference_labels[1]: 'Dirty'}
        else:
            label_map = {reference_labels[0]: 'Dirty', reference_labels[1]: 'Clean'}
        prediction = label_map.get(user_label, "Unknown")
    else:
        prediction = "Unknown"

    st.session_state.predicted_cluster = prediction
    st.rerun()
