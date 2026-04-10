import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


def load_text_model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_selection import CountVectorizer
    model = LogisticRegression(random_state=42, max_iter=1000)
    sample_reviews = [
        'This food is awful',
        'I love this food',
        'This place makes amazing burgers',
        'This is not the best restaurant I have been to',
        'I hate their food',
        'The staff could be friendlier',
        'Not bad!',
        'It took so long for them to make the food I left before they could hand it to me',
        'Compliments to the chef!'
    ]
    labels = [0, 1, 1, 0, 0, 0, 1, 0, 1]

    vectorizer = CountVectorizer()  
    vectorizer.fit(sample_reviews)
    X_transformed = vectorizer.transform(sample_reviews) 
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_transformed, labels) 
    return {'model': model, 'vectorizer': vectorizer}


def predict_text_sentiment(text, model):
    tokens = {t.strip('.,!?').lower() for t in text.split()}
    score = len(tokens & model["positive"]) - len(tokens & model["negative"])
    label = "Positive" if score >= 0 else "Negative"
    return label, score


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


def load_churn_model():
    # TODO: Replace with your trained model
    return np.array([0.02, 0.03, -0.04]), -0.5


def predict_churn(features, model):
    weights, bias = model
    score = np.dot(features, weights) + bias
    prob = 1 / (1 + np.exp(-score))
    label = "Churn" if prob > 0.5 else "No Churn"
    return label, prob


def main():
    st.title("Business Model Dashboard")
    st.write("Starter template for Assignment 5. Replace placeholder models with your own.")

    tabs = st.tabs(["Churn", "Text Sentiment", "Audio Sentiment"])

    with tabs[0]:
        st.header("Customer Churn")
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
        monthly = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
        total = st.number_input("Total Charges", min_value=0.0, value=1000.0)

        churn_model = load_churn_model()
        if st.button("Predict Churn"):
            features = np.array([tenure, monthly, total])
            label, prob = predict_churn(features, churn_model)
            st.success(f"Prediction: {label}")
            st.write(f"Probability: {prob:.2f}")

            fig, ax = plt.subplots()
            ax.bar(["Churn"], [prob])
            ax.set_ylim(0, 1)
            ax.set_ylabel("Probability")
            st.pyplot(fig)

    with tabs[1]:
        st.header("Text Sentiment")
        text_model = load_text_model()
        text_input = st.text_area("Paste a customer review")
        if st.button("Analyze Text") and text_input:
            label, score = predict_text_sentiment(text_input, text_model)
            st.success(f"Sentiment: {label}")
            st.write(f"Score: {score}")

    with tabs[2]:
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
    main()    main()
