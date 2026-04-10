import numpy as np
import pandas as pd
import streamlit as st


def load_text_model():
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(random_state=42, max_iter=1000)
    return {"positive": {"great", "excellent"}, "negative": {"bad", "poor"}}


def predict_text_sentiment(text, model):
    tokens = {t.strip('.,!?').lower() for t in text.split()}
    score = len(tokens & model["positive"]) - len(tokens & model["negative"])
    label = "Positive" if score >= 0 else "Negative"
    return label, score

#######################################

def load_audio_model():
    import torch.nn as nn
    model = nn.Sequential(
        nn.Linear(26, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 2)
    )
    return {"energy_threshold": 0.1}


def predict_audio_sentiment(signal, model):
    energy = np.mean(signal ** 2) if signal is not None else 0.0
    label = "Positive" if energy >= model["energy_threshold"] else "Negative"
    return label, energy

#######################################

def main():
    st.title("Business Model Dashboard")
    st.write("Audio and Text Customer Review Model Accuracy Comparison")

    tabs = st.tabs(["Text Sentiment", "Audio Sentiment"])


    with tabs[0]:
        st.header("Text Sentiment")
        text_model = load_text_model()
        text_input = st.text_area("Paste a customer review")
        if st.button("Analyze Text") and text_input:
            label, score = predict_text_sentiment(text_input, text_model)
            st.success(f"Sentiment: {label}")
            st.write(f"Score: {score}")

    with tabs[1]:
        st.header("Audio Sentiment")
        audio_model = load_audio_model()
        audio_file = st.file_uploader("Upload WAV file", type=["wav"])
        if audio_file is not None:
            # Placeholder: treat raw bytes as signal for demo
            signal = np.frombuffer(audio_file.getbuffer(), dtype=np.int16)
            label, energy = predict_audio_sentiment(signal, audio_model)
            st.success(f"Sentiment: {label}")
            st.write(f"Energy: {energy:.4f}")


if __name__ == "__main__":
    main()
