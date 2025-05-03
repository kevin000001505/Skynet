import streamlit as st
import numpy as np
import pickle
import spacy
import joblib
import config

vectorizer = joblib.load("../models/tfidf/big_data_tfidf_vectorizer.pkl")

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
from data_processing.cleaning import BaseDataCleaner

cleaner = BaseDataCleaner()
label_transofrm = {"1": "Positive", "0": "Negative"}


@st.cache_resource
def load_model():
    with open("../models/big_data.pkl", "rb") as file:
        ml_model = pickle.load(file)
    return ml_model


def clean_text(text):
    doc = nlp(text)
    clean_text = cleaner._process_doc(doc)
    return clean_text


def main():
    ml_model = load_model()
    st.title("🤖Skynet")

    # Create two columns for different input methods
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Text Input")
        text_input = st.text_area(
            "Enter your text here:",
            height=200,
            placeholder="Type or paste your text here...",
        )
        clean_text_result = clean_text(text_input)
        vectorizer_input = vectorizer.transform([clean_text_result])
        prediction_prob = ml_model.predict_proba(vectorizer_input)
        high_confidence = np.max(prediction_prob, axis=1) > config.PROBABILITY_THRESHOLD

        if np.any(high_confidence):
            class_idx = np.argmax(prediction_prob[0])
            prediction_label = str(class_idx)
            st.write("Prediction:", label_transofrm[prediction_label])
            st.write("Confidence:", f"{np.max(prediction_prob) * 100:.2f}%")
        else:
            st.write("Prediction: Low confidence, unable to classify reliably")

    with col2:
        st.subheader("File Upload")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["txt", "csv"],
            help="Supported formats: TXT, CSV",
        )

    # Process button
    if st.button("Analyze", type="primary"):
        if text_input or uploaded_file:
            if text_input:
                st.info("Processing text input...")
                # Add your text processing logic here

            if uploaded_file:
                st.info(f"Processing file: {uploaded_file.name}")
                # Add your file processing logic here
        else:
            st.warning("Please provide either text input or upload a file.")


if __name__ == "__main__":
    # Set page config
    st.set_page_config(page_title="Skynet", page_icon="🤖", layout="wide")
    main()
