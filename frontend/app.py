import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import traceback
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
bert_transform = {
    "LABEL_0": "Negative",
    "LABEL_1": "Positive",
}


@st.cache_resource
def load_model():
    try:
        model_path = os.path.join(config.PROJECT_ROOT, "models", "big_data.pkl")
        model_path = os.path.abspath(model_path)
        ml_model = joblib.load(model_path)
        return ml_model, BertPrediction(version="1")
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
    clean_text_result = clean_text(text)
    prediction_prob = ml_model.predict_proba([clean_text_result])
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
            "BERTPrediction": bert_transform[str(bert_model_prediction[0]["label"])],
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
    placeholder = st.empty()
    # Classify button
    if st.button("Classify", type="primary") and text_input:
        placeholder = st.empty()
        try:
            placeholder.info("Classifying... Please wait while we process your text.")
            result = classify_semantic(text_input, ml_model, bert_model)
            st.write("## Prediction Result")
            if "BERTPrediction" in result:
                df = pd.DataFrame(
                    {
                        "Model": ["Random Forest", "distilBERT"],
                        "Prediction": [
                            result["RFPrediction"],
                            result["BERTPrediction"],
                        ],
                        "Confidence": [
                            result["RFConfidence"],
                            result["BERTConfidence"],
                        ],
                    }
                )
            else:
                df = pd.DataFrame(
                    {
                        "Model": ["Random Forest"],
                        "Prediction": [result["RFPrediction"]],
                        "Confidence": [result["RFConfidence"]],
                    }
                )
            st.markdown(df.style.hide(axis="index").to_html(), unsafe_allow_html=True)
            placeholder.success("Classification complete!")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

    if uploaded_file:
        placeholder.info(f"Processing file: {uploaded_file.name}")

        if uploaded_file.type == "text/csv":
            try:
                df = pd.read_csv(uploaded_file)

                # Initialize session state for selection flow if needed
                if "selection_stage" not in st.session_state:
                    st.session_state.selection_stage = 0

                # Step 1: Select text column
                columns = df.columns.tolist()
                columns.insert(0, "None")  # Add option for no text column
                text_column = st.selectbox(
                    "Select text column", options=columns, key="text_col"
                )

                # Once text column is selected, move to next stage
                if text_column and st.session_state.selection_stage < 1:
                    st.session_state.selection_stage = 1

                # Step 2: Select label column (only show if text column is selected)
                if st.session_state.selection_stage >= 1:
                    label_options = [
                        col for col in columns if col != text_column or col == "None"
                    ]
                    label_column = st.selectbox(
                        "Select label column", options=label_options, key="label_col"
                    )

                    # Once label column is selected and not "None", move to next stage
                    if label_column != "None" and st.session_state.selection_stage < 2:
                        st.session_state.selection_stage = 2

                    # Step 3: Select positive/negative labels if needed
                    if label_column != "None" and st.session_state.selection_stage >= 2:
                        label_values = df[label_column].unique().tolist()
                        if len(label_values) > 2:
                            pos_label = st.selectbox(
                                "Select positive label",
                                options=label_values,
                                key="pos_label",
                            )

                            # Once positive label is selected, show negative label options
                            if pos_label:
                                neg_label = st.selectbox(
                                    "Select negative label",
                                    options=[
                                        label
                                        for label in label_values
                                        if label != pos_label
                                    ],
                                    key="neg_label",
                                )

                                # Filter data once both labels are selected
                                if neg_label:
                                    df = df[
                                        df[label_column].isin([pos_label, neg_label])
                                    ]

                # Process the data
                if st.button("Classify Data", type="primary", key="analyze_csv"):
                    df["cleaned_text"] = (
                        df[text_column].astype(str).apply(lambda x: clean_text(x))
                    )

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
                        df.loc[low_confidence_index][text_column].to_list()
                    )
                    for idx, item in zip(low_confidence_index, bert_result):
                        df.at[idx, "prediction"] = bert_transform[item["label"]]
                        df.at[idx, "confidence"] = round(item["score"] * 100, 2)

                    display_cols = [text_column, "prediction", "confidence"]

                    # If label column was selected, include it
                    if label_column != "None":
                        display_cols.insert(1, label_column)

                    df[label_column] = df[label_column].str.lower()
                    df["prediction"] = df["prediction"].str.lower()

                    cm = confusion_matrix(df[label_column], df["prediction"])

                    labels = sorted(df[label_column].unique())

                    fig, _ = plt.subplots(figsize=(3, 2))
                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt="d",
                        cmap="Blues",
                        xticklabels=labels,
                        yticklabels=labels,
                    )
                    plt.xlabel("Predicted")
                    plt.ylabel("Actual")
                    plt.title("Confusion Matrix of Sentiment Analysis Predictions")

                    st.pyplot(fig)

                    st.write("### Classification Report")
                    report = classification_report(
                        df[label_column], df["prediction"], output_dict=True
                    )
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df)

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
                st.error(f"Error location: {str(e.__traceback__.tb_lineno)}")
                st.error(f"Detailed traceback: {traceback.format_exc()}")
        else:
            st.error("Unsupported file type. Please upload a TXT or CSV file.")
    placeholder.empty()


if __name__ == "__main__":
    main()
