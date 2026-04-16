import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📰 Fake News Detection App")

user_input = st.text_area("Enter News Text:")

if st.button("Check News"):
    data = vectorizer.transform([user_input])
    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("❌ Fake News")
    else:
        st.success("✅ Real News")