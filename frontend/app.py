import streamlit as st
import numpy as np
import pandas as pd
import spacy
import joblib
import os
import re
import sys

# Add parent directory to path so imports work correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from transformer.BERT import BertPrediction

# Load NLP model
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
from data_processing.cleaning import BaseDataCleaner

cleaner = BaseDataCleaner()
label_transform = {"1": "Positive", "0": "Negative"}


@st.cache_resource
def load_model():
    try:
        model_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "big_data.pkl"
        )
        model_path = os.path.abspath(model_path)
        ml_model = joblib.load(model_path)
        return ml_model, BertPrediction(version="0.2")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.error(f"Path attempted: {os.path.abspath(model_path)}")
        return None, None


def clean_text(text):
    text = re.sub(config.REGEX_URL, "", text)
    doc = nlp(text)
    clean_text = cleaner._process_doc(doc)
    return clean_text


def classify_semantic(text, ml_model, bert_model):
    prediction_prob = ml_model.predict_proba([text])
    high_confidence = np.max(prediction_prob, axis=1) > config.PROBABILITY_THRESHOLD

    class_idx = np.argmax(prediction_prob[0])
    prediction_label = str(class_idx)
    RFPrediction = label_transform[prediction_label]
    RFConfidence = f"{np.max(prediction_prob[0]) * 100:.2f}%"

    if np.any(high_confidence):
        return {
            "RFPrediction": RFPrediction,
            "RFConfidence": RFConfidence,
        }
    else:
        bert_model_prediction = bert_model.predict(text)
        return {
            "RFPrediction": RFPrediction,
            "RFConfidence": RFConfidence,
            "BERTPrediction": label_transform[str(bert_model_prediction[0]["label"])],
            "BERTConfidence": f"{bert_model_prediction[0]['score'] * 100:.2f}%",
        }


def main():
    # Set page config first
    st.set_page_config(page_title="Skynet", page_icon="🤖", layout="wide")

    # Load the model
    ml_model, bert_model = load_model()
    if ml_model is None or bert_model is None:
        st.error("Failed to load the model. Please check the logs.")
        return

    st.title("🤖Skynet")
    st.write(f"Random Forest confidence threshold: {config.PROBABILITY_THRESHOLD}")

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
            type=["csv"],
            help="Supported formats: CSV",
        )

    # Classify button
    if st.button("Classify", type="primary") and text_input:
        placeholder = st.empty()
        clean_text_result = clean_text(text_input)
        try:
            placeholder.info("Classifying... Please wait while we process your text.")
            result = classify_semantic(clean_text_result, ml_model, bert_model)
            st.write("## Prediction Result")
            if 'BERTPrediction' in result:
                df = pd.DataFrame({
                    "Model": ["Random Forest", "distilBERT"],
                    "Prediction": [result['RFPrediction'], result['BERTPrediction']],
                    "Confidence": [result['RFConfidence'], result['BERTConfidence']]
                })
            else:
                df = pd.DataFrame({
                    "Model": ["Random Forest"],
                    "Prediction": [result['RFPrediction']],
                    "Confidence": [result['RFConfidence']]
                })
            st.markdown(
                df.style.hide(axis="index").to_html(),
                unsafe_allow_html=True
            )
            placeholder.success("Classification complete!")
            placeholder.empty()
        except Exception as e:
            st.error(f"Error during prediction: {e}")

    if uploaded_file:
        placeholder = st.empty()
        placeholder.info(f"Processing file: {uploaded_file.name}")

        if uploaded_file.type == "text/csv":
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
                        label_transform[str(pred)] for pred in predictions
                    ]
                    df["confidence"] = [
                        round(np.max(prob) * 100, 2) for prob in pred_proba
                    ]

                    low_confidence_index = df[
                        df["confidence"] < config.PROBABILITY_THRESHOLD
                    ].index.to_list()

                    bert_result = bert_model.predict(
                        df.loc[low_confidence_index]["cleaned_text"].to_list()
                    )
                    for idx, item in zip(low_confidence_index, bert_result):
                        df.at[idx, "prediction"] = label_transform[item["label"]]
                        df.at[idx, "confidence"] = round(item["score"] * 100, 2)

                    display_cols = [text_column, "prediction", "confidence"]

                    # If label column was selected, include it
                    if label_column != "None":
                        display_cols.insert(1, label_column)

                    placeholder.empty()

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
