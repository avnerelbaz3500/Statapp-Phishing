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
st.set_page_config(page_title="Détection d'emails de phishing", page_icon="🔍", layout="centered")

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


st.markdown(
    """
    <style>
        /* Enlever la barre blanche en haut */
        header {visibility: hidden;}

        /* Changer la couleur de fond principale de la page */
        [data-testid="stAppViewContainer"] { 
            background-color: #0a192f;  /* Bleu foncé */
        }
        
        /* Modifier la couleur du texte */
        body {
            color: #ffffff !important;  /* Texte en blanc */
            font-family: 'Arial', sans-serif;
        }

        /* Conteneur principal pour remonter le titre */
        .main-container {
            margin-top: -80px;  /* Remonter le titre vers le haut */
            text-align: center;
        }

        /* Style du titre principal */
        .main-title {
            color: #ffffff !important;  /* Texte blanc */
            font-size: 38px;
            font-weight: bold;
            text-align: center;
            text-shadow: 0 0 8px #00C7FF, 0 0 16px #0088CC; /* Effet néon plus doux */
            margin-bottom: 5px;  /* Réduire l'espace sous le titre */
            margin-top: -60px;
        }

        /* Espacement sous le titre */
        .spacer {
            padding-top: 30px;  /* Ajoute un espace sous le titre */
        }

        /* Description sous le titre */
        .description {
            color: #ffffff !important;  /* Texte blanc */
            font-size: 18px;
            font-weight: normal;
            text-align: center;
            margin-top: 50px;  /* Ajouter un espace sous le titre */
            margin-bottom: 20px;
        }

        /* Label "📧 Collez votre email ici" */
        label {
            color: #ffffff !important;  /* Texte blanc */
            font-size: 1.3em;
            font-weight: bold;
        }

        /* Zone de texte : couleur bleue comme le bouton, avec transparence */
        textarea {
            background-color: rgba(2, 136, 209, 0.3) !important;  /* Bleu clair avec transparence */
            border: none !important;  /* Supprimer la bordure */
            border-radius: 10px !important;
            color: #ffffff !important;  /* Texte blanc */
            font-size: 16em;
            padding: 10px;
        }

        /* Bouton */
        .stButton>button {
            background-color: #0288d1;  /* Bouton bleu */
            border: none;
            border-radius: 10px;
            color: #FFFFFF;
            font-size: 1.2em;
            padding: 10px 20px;
            width: 100%;
        }

        /* Effet hover du bouton */
        .stButton>button:hover {
            background-color: #0277bd;  /* Légèrement plus foncé */
        }

        /* Résultat du phishing */
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
    unsafe_allow_html=True
)

# Interface Streamlit
st.markdown("<h1 class='main-title'>🔍 Détection d'emails de phishing</h1>", unsafe_allow_html=True)
# Ajouter un div qui force un espacement sous le titre
st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
st.markdown("<p class='description'>Vous avez reçu un email suspect ? Copiez-collez son contenu ci-dessous, et notre outil analysera s'il s'agit d'une tentative de phishing. Restez vigilant face aux cybermenaces ! 🚨</p>", unsafe_allow_html=True)

# Champs de saisie et bouton
email_input = st.text_area(" ", placeholder="Copiez-collez ici le texte d'un email suspect...")

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
