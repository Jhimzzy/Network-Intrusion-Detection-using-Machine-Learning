import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

st.set_page_config(
    page_title="Network Intrusion Detection System",
    page_icon="🔒",
    layout="wide"
)

ARTIFACT_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "rf_pipeline.joblib")
FEATURE_NAMES_PATH = os.path.join(ARTIFACT_DIR, "feature_names.joblib")

EXPECTED_FEATURES = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
    "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean",
    "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Fwd URG Flags",
    "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std",
    "Packet Length Variance", "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
    "PSH Flag Count", "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Header Length.1", "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets",
    "Subflow Bwd Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean", "Active Std",
    "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min"
]


@st.cache_resource
def load_or_train_model():
    """
    Load the saved pipeline from disk if it exists.
    Otherwise train from the artifact CSVs and cache to disk.
    """
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURE_NAMES_PATH):
        pipeline = joblib.load(MODEL_PATH)
        feature_names = joblib.load(FEATURE_NAMES_PATH)
        return pipeline, feature_names

    else:
        st.info("No saved model found. Training now from artifact files — this may take a minute...")

    X_train = pd.read_csv(os.path.join(ARTIFACT_DIR, "X_train_resampled.csv"))
    y_train = pd.read_csv(os.path.join(ARTIFACT_DIR, "y_train_resampled.csv")).squeeze()

    feature_names = X_train.columns.tolist()

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(max_depth=5, random_state=42, n_jobs=-1))
    ])

    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(feature_names, FEATURE_NAMES_PATH)

    return pipeline, feature_names


def predict(pipeline, feature_names, input_df):
    """Run prediction and return labels and probabilities."""
    input_aligned = input_df[feature_names]
    predictions = pipeline.predict(input_aligned)
    probabilities = pipeline.predict_proba(input_aligned)
    return predictions, probabilities


def render_manual_input_form(feature_names):
    """
    Renders feature inputs in a 4-per-row grid layout.
    Returns a single-row DataFrame with the entered values.
    """
    st.subheader("Enter Feature Values")
    st.caption("Fill in all fields then click Predict below.")

    values = {}
    cols_per_row = 4

    # Chunk the feature list into groups of 4
    for row_start in range(0, len(feature_names), cols_per_row):
        row_features = feature_names[row_start: row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, feat in zip(cols, row_features):
            with col:
                values[feat] = st.number_input(feat, value=0.0, key=f"feat_{feat}")

    return pd.DataFrame([values])


def check_csv_compatibility(uploaded_df, feature_names):
    """Check whether the uploaded CSV has all required columns."""
    uploaded_cols = set(uploaded_df.columns.tolist())
    required_cols = set(feature_names)
    missing = required_cols - uploaded_cols
    extra = uploaded_cols - required_cols
    return len(missing) == 0, missing, extra


def render_results(predictions, probabilities, pipeline):
    """Display prediction results and probability breakdown."""
    st.subheader("Prediction Results")

    results_df = pd.DataFrame({
        "Row": range(1, len(predictions) + 1),
        "Predicted Class": predictions
    })

    def highlight_attack(val):
        if val == "BENIGN":
            return "background-color: #d4edda; color: #155724"
        return "background-color: #f8d7da; color: #721c24"

    styled = results_df.style.map(highlight_attack, subset=["Predicted Class"])
    st.dataframe(styled, use_container_width=True)

    st.subheader("Class Probability Breakdown")
    prob_df = pd.DataFrame(probabilities, columns=pipeline.classes_)
    prob_df.insert(0, "Row", range(1, len(prob_df) + 1))
    st.dataframe(
        prob_df.style.format({c: "{:.4f}" for c in pipeline.classes_}),
        use_container_width=True
    )

    # Bar chart only makes sense for a single record
    if len(predictions) == 1:
        st.subheader("Probability Distribution")
        prob_series = pd.Series(probabilities[0], index=pipeline.classes_).sort_values(ascending=False)
        st.bar_chart(prob_series)


# Header
st.title("🔒 Network Intrusion Detection System")
st.markdown("**Real-Time Traffic Classification using Random Forest**")

cover_image_path = "images/cover-image.jpg"
if os.path.exists(cover_image_path):
    st.image(cover_image_path, use_container_width=True)

st.markdown("---")

# Load or train the model
try:
    pipeline, feature_names = load_or_train_model()
    st.success("Model loaded and ready.")
except FileNotFoundError as e:
    st.error(
        f"Could not find required artifact files: {e}. "
        "Make sure the 'artifacts/' folder contains X_train_resampled.csv and y_train_resampled.csv."
    )
    st.stop()

# Sidebar for navigation and controls
with st.sidebar:
    st.header("Controls")
    st.markdown("Use the options below to configure how you want to provide input data.")
    st.markdown("---")

    input_mode = st.radio(
        "Input Mode",
        ["Manual Entry", "Upload CSV"],
        help="Manual Entry lets you type in individual feature values. Upload CSV lets you batch-predict from a file."
    )

    st.markdown("---")
    st.markdown("**Model Info**")
    st.markdown(f"Features expected: `{len(feature_names)}`")
    st.markdown(f"Classes: `{len(pipeline.classes_)}`")
    with st.expander("View all class labels"):
        for label in pipeline.classes_:
            st.markdown(f"- {label}")

    with st.expander("View all expected features"):
        for feat in feature_names:
            st.markdown(f"- {feat}")

# Main content area based on selected mode
st.markdown(f"### Mode: {input_mode}")

if input_mode == "Manual Entry":
    input_df = render_manual_input_form(feature_names)
    if st.button("Predict", type="primary"):
        predictions, probabilities = predict(pipeline, feature_names, input_df)
        render_results(predictions, probabilities, pipeline)

else:
    st.subheader("Upload a CSV File")
    st.caption(
        "Each row should represent one network flow record. "
        "Column names must match the training feature set exactly."
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        uploaded_df = pd.read_csv(uploaded_file)
        st.write(f"Uploaded: {len(uploaded_df)} rows, {len(uploaded_df.columns)} columns.")

        compatible, missing_cols, extra_cols = check_csv_compatibility(uploaded_df, feature_names)

        if not compatible:
            st.warning(
                f"The uploaded file is missing {len(missing_cols)} required column(s): "
                f"{sorted(missing_cols)}. Please ensure your CSV matches the training feature set."
            )
        else:
            if extra_cols:
                st.info(
                    f"{len(extra_cols)} extra column(s) found and will be ignored: "
                    f"{sorted(extra_cols)}"
                )

            st.dataframe(uploaded_df.head(5), use_container_width=True)

            if st.button("Run Predictions", type="primary"):
                predictions, probabilities = predict(pipeline, feature_names, uploaded_df)
                render_results(predictions, probabilities, pipeline)

st.markdown("---")

st.warning(
    "**Simulation Notice:** This application is intended purely for academic and demonstration "
    "purposes as part of a university project. It simulates a real-time intrusion detection "
    "interface but is not production-ready. The model was trained on the CIC-IDS-2017 dataset "
    "under controlled conditions. For meaningful results, input data must be derived from the "
    "same feature extraction process used during training (e.g. via CICFlowMeter on raw PCAP "
    "files). It cannot be applied to arbitrary network captures or other datasets without "
    "retraining, and should not be used for actual security monitoring or threat response."
)
