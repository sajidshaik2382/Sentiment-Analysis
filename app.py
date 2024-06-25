import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image

with open("naive_bayes_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

st.title("Sentiment Analysis App")
st.markdown("By Sajid")
image = Image.open("sentiment.png")
st.image(image, use_column_width=True)

st.subheader("Enter your text here:")
user_input = st.text_area("")

if st.button("Predict"):
    text_vectorized = vectorizer.transform([user_input])  # Preprocess the input text
    prediction = model.predict(text_vectorized)[0]  # Make predictions

    st.header("Prediction:")
    if prediction == 'negative':
        st.subheader("The sentiment of the given text is: Negative")
    elif prediction == 'neutral':
        st.subheader("The sentiment of the given text is: Neutral")
    elif prediction == 'positive':
        st.subheader("The sentiment of the given text is: Positive")