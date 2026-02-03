import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


data = pd.read_excel("clean_reviews_dataset.xlsx")


data["sentiment"] = data["rating"].apply(
    lambda x: "Positive" if x >= 4 else "Neutral" if x == 3 else "Negative"
)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

data["clean_review"] = data["review"].apply(clean_text)


vectorizer = TfidfVectorizer(max_features=2000)
X = vectorizer.fit_transform(data["clean_review"])
y = data["sentiment"]


model = LogisticRegression()
model.fit(X, y)


positive_words = ["good", "very good", "nice", "excellent", "amazing", "great", "love", "awesome"]
negative_words = ["bad", "worst", "terrible", "awful", "hate", "poor", "useless"]
neutral_words  = ["okay", "average", "fine", "normal", "not bad", "not good"]


st.title("AI Echo - Sentiment Analysis App")


user_text = st.text_area("Enter your review:")

if st.button("Predict Sentiment"):
    if user_text.strip() == "":
        st.warning("Please type something")
    else:
        text = user_text.lower()
        predicted = None

        
        for w in neutral_words:
            if w in text:
                predicted = "Neutral"
                break

        
        if predicted is None:
            for w in positive_words:
                if w in text:
                    predicted = "Positive"
                    break

        # Negative
        if predicted is None:
            for w in negative_words:
                if w in text:
                    predicted = "Negative"
                    break

        # ML fallback
        if predicted is None:
            clean = clean_text(user_text)
            text_num = vectorizer.transform([clean])
            predicted = model.predict(text_num)[0]

        # Show result
        if predicted == "Positive":
            st.success("Sentiment: Positive ")
        elif predicted == "Neutral":
            st.info("Sentiment: Neutral ")
        else:
            st.error("Sentiment: Negative ")

