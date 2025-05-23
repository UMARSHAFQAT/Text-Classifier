import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Sentiment Classifier", layout="centered")

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

classifier = load_model()

st.title("🧠 Sentiment Classifier")
st.markdown("Enter some text and get the predicted sentiment (positive or negative).")

user_input = st.text_area("Enter your text here:", height=150)

if st.button("Classify Sentiment"):
    if user_input.strip():
        result = classifier(user_input)[0]
        label = result["label"]
        score = result["score"]

        # Set color based on sentiment
        color = "#d9534f" if label == "NEGATIVE" else "#5cb85c"  # red or green

        # Display with colored background using HTML
        st.markdown(
            f"""
            <div style="
                background-color: {color};
                padding: 15px;
                border-radius: 8px;
                color: white;
                font-weight: bold;
                font-size: 18px;
                ">
                Sentiment: {label} <br>
                Confidence: {score:.2%}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning("Please enter some text to analyze.")
