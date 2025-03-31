import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import unicodedata
import regex as re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from streamlit.web import cli as stcli
import sys
from nbformat import read
from nbconvert import PythonExporter
from types import ModuleType
import json
import openai
import pandas as pd




# Charger le modèle entraîné et le vectorizer
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
model = pickle.load(open("models/multinomial_nb_model.pkl", "rb"))

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

sw = set(stopwords.words('english') + ['hou', 'ect'])
lemmatizer = WordNetLemmatizer()

def stop_lem(text):
    text = ' '.join(word for word in text.split(' ') if word not in sw)
    return ' '.join(lemmatizer.lemmatize(word) for word in text.split(' '))

def preprocessing(text):
    return stop_lem(clean_text(text))

def predict_email(email_text_cleaned):
    email_vectorized = vectorizer.transform([email_text_cleaned])
    prediction = model.predict(email_vectorized)
    return "Phishing" if prediction[0] == 1 else "Légitime"


st.title("Détection de Phishing")
st.write("Entrez un email ci-dessous et nous vous dirons s'il est suspect.")

email_input = st.text_area("Collez votre email ici")

if st.button("Analyser"):
    result = predict_email(email_input)
    st.write(f"Résultat : {result}")


# Section to generate an email based on user characteristics


import openai


AZURE_OPENAI_ENDPOINT = ""
AZURE_OPENAI_API_KEY = ""
AZURE_DEPLOYMENT_NAME = "gpt-35-turbo-16k" 
AZURE_API_VERSION = "2024-08-01-preview"  


client = openai.AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_API_VERSION,
)

examples_df = pd.read_csv("models/filtered_phishing_emails.csv").sample(n=5)
examples=examples_df['phishing_email'].tolist()


def generate_phishing_email(entry, examples=examples):
    prompt = f""" 
You are an advanced AI trained in cybersecurity awareness. Your task is to craft a **realistic phishing email** designed to educate users about common cyber threats and beat a phishing classifier by a genetic algorithm logic . Given a dictionary containing a person’s characteristics {entry.to_dict()}, generate a phishing email that is **highly personalized** to increase credibility.

---

**Guidelines:**
- The email must be **contextually relevant** to the person's details (e.g., job role, interests, recent activities).
- Use a **convincing sender name and email** (e.g., a trusted organization, a financial service, or a known brand).
- Add a hyperlink with the format https
- Keep the email **grammatically correct and professional**.


Take inspiration from these previous mails that haven't been flagged : {" ".join(examples)}. Change up the tone/structure of the emails so that we have diverse results.
 


"""
    messages = [
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model=AZURE_DEPLOYMENT_NAME,
        messages=messages,
        temperature=0.5,
        top_p=0.9,
    )
    return response.choices[0].message.content



st.title("Génération d'Email")
st.write("Téléchargez un fichier contenant les caractéristiques de l'utilisateur pour générer un email.")



def parse_user_data(file):
    try:
        user_data = json.load(file)
        return {
            "name": user_data.get("name", "Utilisateur"),
            "age": user_data.get("age", 0),
            "location": user_data.get("location", "Inconnu"),
            "preferences": user_data.get("preferences", "")
        }
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier : {e}")
        return None

def generate_email_from_data(name, age, location, preferences):
    entry = pd.Series({"name": name, "age": age, "location": location, "preferences": preferences})
    return generate_phishing_email(entry)

def display_generated_email(email):
    st.write("Email généré :")
    st.text_area("Email", value=email, height=200)


uploaded_file = st.file_uploader("Téléchargez un fichier (format JSON)", type=["json"])

if uploaded_file is not None:
    user_data = parse_user_data(uploaded_file)
    if user_data and st.button("Générer un email"):
        generated_email = generate_email_from_data(
            user_data["name"], user_data["age"], user_data["location"], user_data["preferences"]
        )
        display_generated_email(generated_email)

st.write("Ou entrez manuellement les caractéristiques de l'utilisateur ci-dessous :")
manual_name = st.text_input("Nom de l'utilisateur", "")
manual_age = st.number_input("Âge de l'utilisateur", min_value=0, step=1)
manual_location = st.text_input("Localisation de l'utilisateur", "")
manual_preferences = st.text_area("Préférences de l'utilisateur", "")

if st.button("Utiliser les données manuelles pour générer un email"):
    generated_email = generate_email_from_data(manual_name, manual_age, manual_location, manual_preferences)
    display_generated_email(generated_email)
