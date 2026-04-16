import pandas as pd
import nltk
import string
from nltk.corpus import stopwords

nltk.download('stopwords')

# Load dataset (combined dataset.csv)
df = pd.read_csv("dataset.csv")

# Remove null values
df.dropna(inplace=True)

# Stopwords
stop_words = set(stopwords.words('english'))

# Cleaning function
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Apply cleaning
df['clean_text'] = df['text'].apply(clean_text)

# Save cleaned file
df.to_csv("cleaned_dataset.csv", index=False)

print("Cleaning done ✅")