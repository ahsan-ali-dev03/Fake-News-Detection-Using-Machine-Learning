import streamlit as st
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# UI
st.title("Fake News Detection App")

text = st.text_area("Enter News Text")

if st.button("Check News"):
    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        vec = vectorizer.transform([text])
        result = model.predict(vec)
        
        if result[0] == 0:
            st.error("Fake News ❌")
        else:
            st.success("Real News ✅")
