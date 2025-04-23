import pandas as pd
import pickle
import regex as re
import string
import unicodedata
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from tqdm import tqdm

tqdm.pandas()

# Charger les données
df = pd.read_csv("data/merged_data.csv")

# Nettoyage du texte
def clean_text(text):
    try:
        text = unicodedata.normalize("NFKC", text)
    except:
        return ""
    text = str(text).lower()
    sequences = ['\[.*?\]', 'https?://\S+|www\.\S+', '<.*?>+', '[%s]' % re.escape(string.punctuation), '\n', '\r', '\w*\d\w*']
    for sequence in sequences:
        text = re.sub(sequence, '', text)
    return text

df['body'] = df['body'].apply(clean_text)

# Stopwords et lemmatisation
sw = set(stopwords.words('english') + ['hou', 'ect'])
lemmatizer = WordNetLemmatizer()

def stop_lem(text):
    text = ' '.join(word for word in text.split() if word not in sw)
    return ' '.join(lemmatizer.lemmatize(word) for word in text.split())

df['body'] = df['body'].progress_apply(stop_lem)

# Séparation en train/test
x_train, x_test, y_train, y_test = train_test_split(df["body"], df["label"], random_state=42)

# Vectorisation
vect = CountVectorizer()
vect.fit(x_train)

x_train = vect.transform(x_train)
x_test = vect.transform(x_test)

# Entraînement du modèle
nb = MultinomialNB()
nb.fit(x_train, y_train)

# Sauvegarde du modèle et du vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vect, f)

with open("model.pkl", "wb") as f:
    pickle.dump(nb, f)
