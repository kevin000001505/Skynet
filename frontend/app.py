import streamlit as st
import numpy as np
import pandas as pd
import spacy
import joblib
import os
import sys

# Add parent directory to path so imports work correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

# Load NLP model
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
from data_processing.cleaning import BaseDataCleaner

cleaner = BaseDataCleaner()
label_transofrm = {"1": "Positive", "0": "Negative"}


@st.cache_resource
def load_model():
    try:
        model_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "big_data.pkl"
        )
        model_path = os.path.abspath(model_path)
        ml_model = joblib.load(model_path)
        return ml_model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.error(f"Path attempted: {os.path.abspath(model_path)}")
        return None


def clean_text(text):
    doc = nlp(text)
    clean_text = cleaner._process_doc(doc)
    return clean_text


def classify_semantic(text, ml_model):
    prediction_prob = ml_model.predict_proba([text])
    high_confidence = np.max(prediction_prob, axis=1) > config.PROBABILITY_THRESHOLD

    if np.any(high_confidence):
        class_idx = np.argmax(prediction_prob[0])
        prediction_label = str(class_idx)
        st.write("Prediction:", label_transofrm[prediction_label])
        st.write("Confidence:", f"{np.max(prediction_prob[0]) * 100:.2f}%")
    else:
        st.write("Confidence:", f"{np.max(prediction_prob[0]) * 100:.2f}%")
        st.write("Prediction: Low confidence, unable to classify reliably")


def main():
    # Set page config first
    st.set_page_config(page_title="Skynet", page_icon="🤖", layout="wide")

    # Load the model
    ml_model = load_model()
    if ml_model is None:
        st.error("Failed to load the model. Please check the logs.")
        return

    st.title("🤖Skynet")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Text Input")
        text_input = st.text_area(
            "Enter your text here:",
            height=200,
            placeholder="Type or paste your text here...",
        )

    with col2:
        st.subheader("File Upload")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["txt", "csv"],
            help="Supported formats: TXT, CSV",
        )

    # Classify button
    if st.button("Classify", type="primary") and text_input:
        clean_text_result = clean_text(text_input)
        try:
            classify_semantic(clean_text_result, ml_model)
        except Exception as e:
            st.error(f"Error during prediction: {e}")

    if uploaded_file:
        st.info(f"Processing file: {uploaded_file.name}")
        if uploaded_file.type == "text/plain":
            try:
                text_data = uploaded_file.read().decode("utf-8")
                clean_text_result = clean_text(text_data)
                classify_semantic(clean_text_result, ml_model)
            except Exception as e:
                st.error(f"Error processing text file: {e}")

        elif uploaded_file.type == "text/csv":
            try:
                df = pd.read_csv(uploaded_file)

                columns = df.columns.tolist()
                text_column = st.selectbox(
                    "Select text column", options=columns, key="text_col"
                )
                label_options = [col for col in columns if col != text_column]
                label_column = st.selectbox(
                    "Select label column", options=label_options, key="label_col"
                )

                # Process the data
                if st.button("Classify Data", type="primary", key="analyze_csv"):
                    # Clean the text data
                    df["cleaned_text"] = df[text_column].astype(str).apply(clean_text)

                    predictions = ml_model.predict(df["cleaned_text"])
                    pred_proba = ml_model.predict_proba(df["cleaned_text"])
                    df["prediction"] = [
                        label_transofrm[str(pred)] for pred in predictions
                    ]
                    df["confidence"] = [
                        f"{np.max(prob) * 100:.2f}%" for prob in pred_proba
                    ]

                    # Prepare results display
                    display_cols = [text_column, "prediction", "confidence"]

                    # If label column was selected, include it
                    if label_column != "None":
                        display_cols.insert(1, label_column)

                    st.write("### Prediction Results")
                    st.dataframe(df[display_cols])

                    # Download results option
                    csv = df[display_cols].to_csv(index=False)
                    st.download_button(
                        label="Download results as CSV",
                        data=csv,
                        file_name="prediction_results.csv",
                        mime="text/csv",
                    )
            except Exception as e:
                st.error(f"Error processing CSV file: {e}")
        else:
            st.error("Unsupported file type. Please upload a TXT or CSV file.")


if __name__ == "__main__":
    main()
