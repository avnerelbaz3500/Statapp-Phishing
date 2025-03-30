import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import unicodedata
import regex as re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Charger le modèle entraîné et le vectorizer
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

def clean_text(text):
    """ Nettoyage du texte en accord avec l'entraînement du modèle """
    try:
        text = unicodedata.normalize("NFKC", text)
    except:
        return ""
    text = text.lower()
    sequences = [
        '\\[.*?\\]', 'https?://\\S+|www\\.\\S+', '<.*?>+', '[%s]' % re.escape(string.punctuation), '\\n', '\\r', '\\w*\\d\\w*'
    ]
    for sequence in sequences:
        text = re.sub(sequence, '', text)
    return text

sw = set(stopwords.words('english') + ['hou', 'ect'])
lemmatizer = WordNetLemmatizer()

def stop_lem(text):
    text = ' '.join(word for word in text.split(' ') if word not in sw)
    return ' '.join(lemmatizer.lemmatize(word) for word in text.split(' '))

def preprocessing(text):
    return stop_lem(clean_text(text))

def predict_email(email_text):
    email_text_cleaned = preprocessing(email_text)
    email_vectorized = vectorizer.transform([email_text_cleaned])
    prediction = model.predict(email_vectorized)
    return "Phishing" if prediction[0] == 1 else "Légitime"

# Interface Streamlit
st.title("Détection de Phishing")
st.write("Entrez un email ci-dessous et nous vous dirons s'il est suspect.")

email_input = st.text_area("Collez votre email ici")

if st.button("Analyser"):
    result = predict_email(email_input)
    st.write(f"Résultat : {result}")
