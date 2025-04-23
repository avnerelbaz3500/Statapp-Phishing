import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import unicodedata
import regex as re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Configuration de la page Streamlit
st.set_page_config(page_title="Détection de Phishing Email", page_icon="🔍", layout="centered")

# Charger le modèle et le vectorizer
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# Fonction de nettoyage du texte
def clean_text(text):
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

# Liste des stopwords
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

# Application du style CSS personnalisé pour un thème bleu cybersécurité
st.markdown(
    """
    <style>
        body {
            background-color: #1c1f26;  /* Fond sombre pour l'ambiance cybersécurité */
            color: #e3f2fd;  /* Texte bleu clair */
            font-family: 'Arial', sans-serif;
            margin: 0;
            padding: 0;
        }
        .css-1d391kg { /* Sélecteur interne de Streamlit pour l'en-tête */
            background-color: #03a9f4 !important;  /* Fond bleu pour l'en-tête */
            color: #ffffff !important;  /* Texte blanc pour le titre */
            padding: 20px;
            border-radius: 10px;
        }
        .main-title {
            text-align: center;
            color: #ffffff;  /* Texte en blanc */
            font-size: 2.5em;
        }
        .stTextArea {
            background-color: #2e3b46;  /* Fond plus foncé pour la zone de texte */
            border: 2px solid #03a9f4;  /* Bordure bleu clair */
            border-radius: 10px;
            color: #e3f2fd;  /* Texte bleu clair dans la zone de texte */
        }
        .stButton>button {
            background-color: #0288d1;  /* Bouton bleu pour action */
            border: none;
            border-radius: 5px;
            color: #FFFFFF;
            font-size: 1.2em;
            padding: 10px 20px;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #0277bd;  /* Hover effet légèrement plus foncé */
        }
        .result {
            font-size: 1.5em;
            margin-top: 20px;
            text-align: center;
        }
        .phishing {
            color: #f44336;  /* Rouge pour phishing */
        }
        .legitime {
            color: #4CAF50;  /* Vert pour légitime */
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Interface Streamlit
st.markdown("<h1 class='main-title'>🔍 Détection de Phishing Email</h1>", unsafe_allow_html=True)
st.write("### Entrez un email ci-dessous et nous vous dirons s'il est suspect.")

# Champs de saisie et bouton
email_input = st.text_area("📧 Collez votre email ici", placeholder="Copiez-collez ici le texte d'un email suspect...")

# Analyse de l'email
if st.button("Analyser"):
    if not email_input.strip():
        st.warning("Veuillez entrer un email avant d'analyser.")
    else:
        result = predict_email(email_input)
        if result == "Phishing":
            st.markdown(f"<div class='result phishing'>⚠️ Cet email semble être du phishing !</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result legitime'>✅ Cet email est légitime.</div>", unsafe_allow_html=True)
