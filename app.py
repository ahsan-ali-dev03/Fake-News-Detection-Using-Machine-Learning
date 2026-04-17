import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("cleaned_dataset.csv")

X = data["text"]
y = data["label"]

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X, y)

# UI
st.title("📰 Fake News Detection App")

user_input = st.text_area("Enter News Text:")

if st.button("Check News"):
    input_data = vectorizer.transform([user_input])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Real News")
    else:
        st.error("❌ Fake News")
