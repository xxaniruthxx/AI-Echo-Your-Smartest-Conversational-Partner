import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.utils import resample

nltk.download("stopwords")
nltk.download("wordnet")

df = pd.read_excel("chatgpt_style_reviews_dataset.xlsx")

df['sentiment'] = df['rating'].apply(
    lambda x: "Positive" if x >= 4 else "Neutral" if x == 3 else "Negative"
)

stop = set(stopwords.words("english"))
for w in ["not", "no", "never"]:
    if w in stop:
        stop.remove(w)

lemm = WordNetLemmatizer()

def clean(t):
    t = str(t).lower()
    t = re.sub(r"[^a-zA-Z\s]", "", t)
    words = t.split()
    words = [lemm.lemmatize(i) for i in words if i not in stop]
    return " ".join(words)

df["clean_review"] = df["review"].apply(clean)

pos = df[df.sentiment == "Positive"]
neu = df[df.sentiment == "Neutral"]
neg = df[df.sentiment == "Negative"]

min_len = min(len(pos), len(neu), len(neg))

df_bal = pd.concat([
    resample(pos, n_samples=min_len, random_state=42),
    resample(neu, n_samples=min_len, random_state=42),
    resample(neg, n_samples=min_len, random_state=42)
]).sample(frac=1)

tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df_bal["clean_review"])
y = df_bal["sentiment"]

model = LinearSVC()
model.fit(X, y)

st.title("AI Echo - Sentiment Analysis App")

review = st.text_area("Enter a review:")

if st.button("Predict"):
    cleaned = clean(review)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)[0]
    st.success(f"Sentiment: {prediction}")
