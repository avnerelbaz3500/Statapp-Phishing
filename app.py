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

def predict_email(email_text_cleaned,display_percentage=False):
    email_vectorized = vectorizer.transform([email_text_cleaned])
    prediction = model.predict(email_vectorized)
    if display_percentage:
        prediction_proba = model.predict_proba(email_vectorized)
        spam_probability = prediction_proba[0][1] * 100
        if prediction[0] == 1:
            return f"Phishing ({spam_probability:.2f}%)"
        else:
            return f"Légitime ({100 - spam_probability:.2f}%)"
    
    return "Phishing" if prediction[0] == 1 else "Légitime"


st.title("Détection de Phishing")

st.write("Entrez un email ci-dessous et nous vous dirons s'il est suspect.")

email_input = st.text_area("Collez votre email ici")

if st.button("Analyser"):
    result = predict_email(email_input,display_percentage=True)
    st.write(f"Résultat : {result}")


# Section to generate an email based on user characteristics


import openai
import random
import json
from mistralai import Mistral





examples_df = pd.read_csv("models/filtered_phishing_emails.csv")
examples=examples_df['phishing_email'].tolist()








email_themes = [
    "Financial and Payment-Related Subjects",
    "Security and Account-Related Subjects",
    "Offers and Promotions",
    "Work and Business-Related Subjects",
    "Delivery and Shipping",
    "Miscellaneous"
]

with open('azure_config.json', 'r') as config_file:
    azure_config = json.load(config_file)

AZURE_OPENAI_ENDPOINT = azure_config['AZURE_OPENAI_ENDPOINT']
AZURE_OPENAI_API_KEY = azure_config['AZURE_OPENAI_API_KEY']
AZURE_DEPLOYMENT_NAME = azure_config['AZURE_DEPLOYMENT_NAME']
AZURE_API_VERSION = azure_config['AZURE_API_VERSION']


client_gpt = openai.AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_API_VERSION,
)

with open('mistral_key.txt', 'r') as file:
            mistral_api_key = file.readline().strip()

mistral_model = "open-mixtral-8x22b"

client_mistral = Mistral(api_key=mistral_api_key)

def generate_phishing_email(entry, examples=examples,type="individual",model=None,theme=None,language="french"):
    if not theme:
        theme = random.choice(email_themes)
    if not model:
        model = random.choice(["gpt","mistral"])
        
    sample_examples = random.sample(examples, 5)
    prompt = f""" 
You are an advanced AI trained in cybersecurity awareness. Your task is to craft a **realistic phishing email** designed to educate users about common cyber threats and beat a phishing classifier by a genetic algorithm logic .  Given a dictionary containing a {type}’s characteristics {entry.to_dict()}, generate a phishing email {"template" if type!="individual" else ""} that is **highly personalized** to increase credibility.

---

**Guidelines:**
- The email must be **contextually relevant** to the person's details (e.g., job role, interests, recent activities).
- Use a **convincing sender name and email** (e.g., a trusted organization, a financial service, or a known brand).
- Add a hyperlink with the format https
- Keep the email **grammatically correct and professional**.
-If there is no specific info on the target make the theme {theme}, however if there is a specific info on the target, use it to make the email more realistic by making the specific info the theme.


Take inspiration from these previous mails that haven't been flagged : {"\n".join(sample_examples)}. Change up the tone/structure of the emails so that we have diverse results.
IMPORTANT: The language of the email should be in {language}.


"""

    if model == "gpt":
        messages = [
            {"role": "user", "content": prompt}
        ]
        response = client_gpt.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,
            messages=messages,
            temperature=0.5,
            top_p=0.9,
        )
        return response.choices[0].message.content
    else:
        

        chat_response = client_mistral.chat.complete(
            model= mistral_model,
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
        )
        
        return chat_response.choices[0].message.content




st.title("Génération d'Email")

language = st.selectbox("Choisissez la langue de l'email", ["french", "english", "spanish"])

st.write("Téléchargez un fichier contenant les caractéristiques de l'utilisateur pour générer un email.")



def parse_user_data(file):
    try:
        user_data = json.load(file)
        return {
            "name": user_data.get("name", "Utilisateur"),
            "age": user_data.get("age", 0),
            "location": user_data.get("location", "Inconnu"),
            "description": user_data.get("description", "")
        }
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier : {e}")
        return None
    
    
def generate_email_from_data(name, age, location, description):
    entry = pd.Series({"name": name, "age": age, "location": location, "description": description})
    return generate_phishing_email(entry,language=language)

def display_generated_email(email):
    st.write("Email généré :")
    st.text_area("Email", value=email, height=200)


uploaded_file = st.file_uploader("Téléchargez un fichier (format JSON)", type=["json"])

if uploaded_file is not None:
    user_data = parse_user_data(uploaded_file)
    if user_data and st.button("Générer un email"):
        generated_email = generate_email_from_data(
            user_data["name"], user_data["age"], user_data["location"], user_data["description"]
        )
        display_generated_email(generated_email)

st.write("Ou entrez manuellement les caractéristiques de l'utilisateur ci-dessous :")
manual_name = st.text_input("Nom de l'utilisateur", "")
manual_age = st.number_input("Âge de l'utilisateur", min_value=0, step=1)
manual_location = st.text_input("Localisation de l'utilisateur", "")
manual_description = st.text_area("Préférences de l'utilisateur", "")


if st.button("Utiliser les données manuelles pour générer un email"):
    generated_email = generate_email_from_data(manual_name, manual_age, manual_location, manual_description)
    display_generated_email(generated_email)

st.title("Génération d'Emails pour un Groupe")

st.write("Entrez les caractéristiques générales d'un groupe pour générer un template d'email personnalisé.")

def collect_group_characteristics():
    st.subheader("Caractéristiques générales du groupe")
    group_name = st.text_input("Nom du groupe", "")
    common_age = st.number_input("Âge moyen du groupe", min_value=0, step=1)
    common_location = st.text_input("Localisation commune du groupe", "")
    common_description = st.text_area("Description du groupe", "")
    return {
        "group_name": group_name,
        "common_age": common_age,
        "common_location": common_location,
        "common_description": common_description
    }

def generate_email_template(group_characteristics):
    entry = pd.Series({
        "name": group_characteristics.get("group_name", "Groupe"),
        "age": group_characteristics.get("common_age", 0),
        "location": group_characteristics.get("common_location", "Inconnu"),
        "Description": group_characteristics.get("common_description", "")
    })
    return generate_phishing_email(entry,type="groupe",theme=group_characteristics.get("common_description", None),language=language)

def display_email_template(email_template):
    st.write("Template d'email généré pour le groupe :")
    st.text_area("Template d'email", value=email_template, height=200)

group_characteristics = collect_group_characteristics()


if st.button("Générer un template pour le groupe"):
    email_template = generate_email_template(group_characteristics)
    display_email_template(email_template)