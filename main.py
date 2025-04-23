#!/usr/bin/env python
# coding: utf-8

# #  Phishing Email Generation
# 
# Ce notebook a un double objectif :
# 
# 1. **Entraîner un classifieur d’emails de phishing** à l’aide de techniques classiques de NLP (nettoyage, vectorisation, Naive Bayes).
# 2. **Utiliser le prompt engineering** sur un modèle de langage avancé (type GPT) pour générer des emails de phishing suffisamment convaincants pour **tromper ce classifieur**.
# 
# Ce projet s’inscrit dans une logique de **red teaming / adversarial NLP**.
# 

# # Construction d'un classifieur Naive Bayes

# ## Librairies

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
#import kagglehub
import os
import shutil
import regex as re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import numpy as np
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
import unicodedata
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize import word_tokenize
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,accuracy_score
import pickle
import names
from datetime import datetime, timedelta
from faker import Faker
import random
import openai
tqdm.pandas()


# ##  Chargement du Dataset
# 
# Le fichier `merged_data.csv` contient des emails labellisés :
# - `0` : email légitime
# - `1` : email de phishing
# 
# Nous allons inspecter la structure des données avant de les préparer pour le modèle.
# 
# Originie du dataset: https://zenodo.org/records/8339691

# In[35]:


from datasets import load_dataset
import glob


csv_files = glob.glob(os.path.join("data", "*.csv"))
csv_files.remove('data/adult_data.csv')



df = []
for f in csv_files:
    df_temp = pd.read_csv(f)
    df_temp['origin_dataset'] = os.path.basename(f)
    df.append(df_temp)


df = pd.concat(df, ignore_index=True)




df.head()


# In[36]:


df['body'] = df['body'].fillna('') + ' ' + df['text_combined'].fillna('')
df.drop(columns=['sender', 'receiver', 'subject', 'urls', 'text_combined'], inplace=True)
df.head()


# In[ ]:


df.to_csv('data/merged_data.csv', index=False)


# In[ ]:


original_df=df.copy()


# ##  Répartition des emails vrais vs phishing
# 
# Il est crucial de vérifier l’équilibre entre les classes afin d’éviter un biais lors de l’entraînement.
# 

# In[38]:


plt.hist(df.loc[:,'label'], bins=[-0.5,0.5,1.5] ,rwidth=0.8)
plt.xticks([0,1],['True','Fake'])
plt.ylabel("Frequency")
plt.title("Count of true/fake emails")
plt.show()


# On voit que le dataset est équilibré on aura donc pas de manipulations d'under/over sampling

# ## Data preprocessing : Nettoyage et Prétraitement
# 
# Cette étape standardise les emails :
# - suppression des balises HTML, des URLs, de la ponctuation
# - mise en minuscules
# - lemmatisation et suppression des stopwords
# 
# Cela prépare les textes pour leur vectorisation.

# Cleaning

# In[3]:


def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    try:
        text = unicodedata.normalize("NFKC", text)  # Normalize characters
    except:
        print(text)
        return
    text = str(text).lower()
    sequences=['\[.*?\]','https?://\S+|www\.\S+','<.*?>+','[%s]' % re.escape(string.punctuation),'\n','\r','\w*\d\w*']
    for sequence in sequences:
        text=re.sub(sequence,'',text)
    
    return text


# In[39]:


df['body']=df['body'].progress_apply(clean_text)


# In[40]:


df.sample(n=10)['body']


# #### Stopwords + Lemmatization

# In[41]:


sw=set(stopwords.words('english') + ['hou','ect'])
lemmatizer = WordNetLemmatizer()


def stop_lem(text):
    '''Remove stop words and lemmatize the text.'''
    text=' '.join(word for word in text.split(' ') if word not in sw)
    return ' '.join(lemmatizer.lemmatize(word) for word in text.split(' '))

df['body']=df['body'].progress_apply(stop_lem)


# In[42]:


df.sample(n=10)["body"]


# In[43]:


def preprocessing(text):
    '''Helper to prepare text for model input by combining cleaning, stopword removal, and lemmatization.'''
    return stop_lem(clean_text(text))


# In[45]:


preprocessing('This is a test email. Please ignore it. https://example.com [test] 1234') 


# To make data manipulation easier

# In[46]:


true_df,fake_df=df.loc[df['label']==0],df.loc[df['label']==1]


# # Statistiques Descriptives

# WordClouds

# In[47]:


wc=WordCloud(
    background_color='white', 
    max_words=200, 
    collocations=False
)

wc.generate(' '.join(text for text in true_df['body']))
plt.figure(figsize=(15,10))
plt.title('Top words for true emails')
plt.imshow(wc)
plt.axis("off")


# In[48]:


wc=WordCloud(
    background_color='white', 
    max_words=200, 
    collocations=False
)

wc.generate(' '.join(text for text in fake_df['body']))
plt.figure(figsize=(15,10))
plt.title('Top words for fake emails')
plt.imshow(wc)
plt.axis("off")


# In[49]:


plt.figure(figsize=(8, 5))
sns.kdeplot(true_df['body'].progress_apply(len).value_counts().sort_index()[:150], fill=True,label='True emails')  
sns.kdeplot(fake_df['body'].progress_apply(len).value_counts().sort_index()[:150], fill=True,label='Fake emails')  
plt.legend()
plt.xlabel("Text length")
plt.ylabel("Density")
plt.title("KDE plot")


# In[50]:


def count_dict(data):
    words=''.join(text for text in data['body']).split(' ')
    word_counts={}
    for word in tqdm((words)):
        word_counts[word]=word_counts.get(word,0) + 1
    word_counts.pop('')
    s=sum(word_counts.values())
    return {k:v/s for k,v in word_counts.items()}
true_word_count=count_dict(true_df)
fake_word_count=count_dict(fake_df)


# In[51]:


plt.plot(list(sorted(true_word_count.values(),reverse=True))[:100],label='word frequency')
plt.plot([max(list(true_word_count.values()))/(x) for x in np.arange(100)[1:]],label='Theoretical Zipf distribution')
plt.ylabel('frequency')
plt.xlabel('rank')
plt.title('Word frequency vs rank for the real email dataset')
plt.legend()
plt.show()
plt.plot(list(sorted(fake_word_count.values(),reverse=True))[:100],label='word frequency')
plt.plot([max(list(fake_word_count.values()))/x for x in np.arange(100)[1:]],label='Theoretical Zipf distribution')
plt.ylabel('frequency')
plt.xlabel('rank')
plt.title('Word frequency vs rank for the fake email dataset')
plt.legend()


# In[52]:


def gen_sentiment(df,col):
    def pol(x):
        try:
            return TextBlob(x).sentiment.polarity
        except:
            return x
    def sub(x):
        try:
            return TextBlob(x).sentiment.subjectivity
        except:
            return x


    df["polarity"]=df[col].progress_apply(pol)
    df["subjectivity"]=df[col].progress_apply(sub)
    
    
gen_sentiment(true_df,'body')
gen_sentiment(fake_df,'body')


# In[54]:


plt.figure(figsize=(8, 5))
sns.kdeplot(true_df['subjectivity'],fill=True,label='True emails')  
sns.kdeplot(fake_df['subjectivity'], fill=True,label='Fake emails')  
plt.legend()
plt.xlabel("Subjectivity value")
plt.ylabel("Density")
plt.title("Density estimation of subjectivity")


# In[ ]:


def count_hyperlinks(text):
        return len(re.findall(r'http+', text))


def count_attachments(text):
        return len(re.findall(r'attachment|file|pdf|image|zip|doc', text, re.IGNORECASE))


df["hyperlink_count"] = original_df["body"].apply(count_hyperlinks)


fig, ax = plt.subplots(1, 1, figsize=(8, 5))


ax.bar(["Real", "Fake"], df.groupby("label")["hyperlink_count"].mean())
ax.set_title("Average Hyperlinks per Email")
ax.set_ylabel("Average")

plt.show()


# In[59]:


dataset = load_dataset("SetFit/enron_spam")


dates_labels = list(zip(dataset['train']['date'], dataset['train']['label']))
print(dates_labels[:10]) 


# In[60]:


df_date = pd.DataFrame(dates_labels, columns=["date", "label"])
df_date["date"] = pd.to_datetime(df_date["date"], errors="coerce")
df_date["month"] = df_date["date"].dt.month_name()

plt.figure(figsize=(10, 5))
sns.histplot(data=df_date, x="month", bins=12, hue=df["label"],discrete=True)


plt.xlabel("Month")
plt.ylabel("Number of Emails")
plt.title("Email Distribution by Month")
plt.legend(labels=["Spam","Real"])
plt.xticks(rotation=45) 

plt.show()


# In[61]:


df_date["month_day"] = df_date["date"].apply(lambda x:x.day)

plt.figure(figsize=(10, 5))
sns.histplot(data=df_date, x="month_day", bins=31, hue=df["label"],discrete=True)


plt.xlabel("Day")
plt.ylabel("Number of Emails")
plt.title("Email Distribution by day of the month")
plt.legend(labels=["Spam","Real"])
plt.xticks(rotation=45) 

plt.show()


# In[62]:


df_date["weekday"] = df_date["date"].apply(lambda x:x.weekday())

plt.figure(figsize=(10, 5))
sns.histplot(data=df_date, x="weekday", bins=31, hue=df["label"],discrete=True)


plt.xlabel("Day of the week")

plt.ylabel("Number of Emails")
plt.title("Email Distribution by day of the week")
plt.xticks(rotation=45) 
plt.legend(labels=["Spam","Real"])

plt.show()


# # Naive Bayes Model
# 
# https://scikit-learn.org/stable/modules/naive_bayes.html

# ##  Split &  Vectorisation
# 
# Le jeu est divisé en `train` / `test`, puis transformé en vecteurs à l’aide de `CountVectorizer`, basé sur le bag-of-words.
# 

# In[12]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(df["body"],df["label"],random_state=42)


# In[13]:


x_train.sample(n=10), y_train.sample(n=10)


# In[14]:


vect=CountVectorizer()
vect.fit(x_train)


x_train=vect.transform(x_train)
x_test=vect.transform(x_test)


# In[15]:


pickle.dump(vect,open("vectorizer.pkl","wb"))


# ##  Entraînement du Classifieur (Naive Bayes)
# 
# Un modèle `MultinomialNB` est entraîné sur les vecteurs de mots.
# Il constitue notre **filtre anti-phishing de référence**.
# 

# In[29]:


nb = MultinomialNB()

nb.fit(x_train, y_train)


# ##  Évaluation du Classifieur
# 
# On évalue la précision du modèle avec la `confusion matrix` et le score d’accuracy.
# 

# In[17]:


y_pred=nb.predict(x_test)

c_mat=confusion_matrix(y_test,y_pred)

ConfusionMatrixDisplay(c_mat).plot()
accuracy_score(y_test,y_pred)


# In[18]:


pickle.dump(nb,open("multinomial_nb_model.pkl", "wb"))


# In[19]:


#!mkdir models
get_ipython().system('mv multinomial_nb_model.pkl models')


# ##  Fonction de Prédiction
# 
# Cette fonction permet de prédire rapidement une liste d’emails textuels avec le classifieur.
# 

# In[28]:


def predict(text_list):
    "Array of naive bayes model prediction for all texts in text_list"
    return nb.predict(vect.transform(np.array([preprocessing(text) for text in text_list])))

predict(["Ceci est un premier test",'Deuxième test'])


# ## Importation des Profils Cibles (adult.csv)
# 
# Afin de générer des emails de phishing ultra-ciblés, nous utilisons le jeu de données `adult.csv` comme base de profils types (âge, métier, sexe, etc.).
# 
# Nous enrichissons ces profils avec des **noms, entreprises et autres données simulées** (via les bibliothèques `names` et `Faker`), pour pouvoir ensuite générer des emails contextuellement pertinents.
# 
# Ces profils serviront de "prompts" au modèle de langage pour la génération d'emails de phishing personnalisés.
# 
# 
# Origine du dataset: https://www.kaggle.com/datasets/ivanhrek/uci-adult?resource=download
# 

# In[11]:


adult_df = pd.read_csv('data/adult_data.csv',header=None)



adult_df.columns = adult_df.iloc[0]
adult_df = adult_df[1:]


adult_df.head()


# In[13]:


def generate_random_name(sex):
    if sex.lower() == 'male':
        return names.get_full_name(gender='male')
    elif sex.lower() == 'female':
        return names.get_full_name(gender='female')
    else:
        return names.get_full_name()


adult_df['name'] = adult_df['sex'].progress_apply(generate_random_name)


adult_df.head()


# In[14]:


fake = Faker()

# Ajout d'une colonne entreprise
adult_df['company'] = [fake.company() for _ in range(len(adult_df))]

adult_df.head()


# In[15]:


# To keep track of mails that weren't flagged as spam
non_spam_examples=[]


# In[16]:


sample_df = adult_df.sample(n=50)


# In[24]:


email_themes = [
    "Financial and Payment-Related Subjects",
    "Security and Account-Related Subjects",
    "Offers and Promotions",
    "Work and Business-Related Subjects",
    "Delivery and Shipping",
    "Miscellaneous"
]


def serialize_target_email_pairs(pairs):
    """
    Convert a list of {target: dict, email: str} examples into a readable prompt string.
    Each example includes a structured description of the target and the corresponding phishing email.
    
    :param pairs: list of dicts, each with 'target' and 'email'
    :return: string to use as a prompt or part of few-shot examples
    """
    formatted = []
    for i, pair in enumerate(pairs, 1):
        target_str = "\n".join(f"- {k.capitalize()}: {v}" for k, v in pair["target"].items())
        example = f"""### Example {i}

**Target Profile:**
{target_str}

**Generated Phishing Email:**
{pair["email"].strip()}

"""
        formatted.append(example)
    return "\n".join(formatted)
    
    
client = openai.OpenAI(api_key="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")

def generate_phishing_email(entry, examples=non_spam_examples,type="individual",model=None,theme=None,language="french"):
    """ 
    Generate a phishing email based on the target(individual of group)'s profile, a set of examples, a theme, a model, and a language.
    - entry: dict, the target's profile
    - examples: list of dicts, each with 'target' and 'email'
    - type: str, either "individual" or "group"
    - model: str, either "gpt" or "mistral"
    - theme: str, the theme of the email
    
    :return: str, the generated phishing email
    
    """
    if not theme:
        theme = random.choice(email_themes)
    if not model:
        model = random.choice(["gpt","mistral"])
    examples = random.sample(examples, min(5, len(examples))) if examples else []
    examples=serialize_target_email_pairs(examples) if examples else ""        

    prompt = f""" 
You are an advanced AI trained in cybersecurity awareness. Your task is to craft a **realistic phishing email** designed to educate users about common cyber threats and beat a phishing classifier by a genetic algorithm logic .  Given a dictionary containing a {type}’s characteristics {entry.to_dict()}, generate a phishing email {"template" if type!="individual" else ""} that is **highly personalized** to increase credibility.

---

**Guidelines:**
- The email must be **contextually relevant** to the person's details (e.g., job role, interests, recent activities).
- Use a **convincing sender name and email** (e.g., a trusted organization, a financial service, or a known brand).
- Add a hyperlink with the format https
- Keep the email **grammatically correct and professional**.
-The theme of the email should be **{theme}**.
-No emojis


Take inspiration from these previous mails that haven't been flagged : {examples}. Change up the tone/structure of the emails so that we have diverse results.
IMPORTANT: The language of the email should be in {language}.


"""
    
        
    
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Use your deployed model name (Azure OpenAI users replace this with deployment name)
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    
        
    
    return response.choices[0].message.content.strip()
    

    







sample_df.head()


# In[19]:


def compute_spam_percentage(df):
    predictions = predict(df['phishing_email'].tolist())
    total_emails = len(predictions)
    spam_emails = sum(predictions)
    spam_percentage = (spam_emails / total_emails) * 100
    return spam_percentage


# In[25]:


from tqdm import tqdm


sample_df = adult_df.sample(n=50)


sample_df['phishing_email'] = sample_df.progress_apply(generate_phishing_email, axis=1)


sample_df['is_spam'] = sample_df['phishing_email'].apply(lambda email: predict(email) == 1)  # Change 1 to 0 if needed


non_spam_df = sample_df[~sample_df['is_spam']] 


non_spam_examples += [
    {"target": row.to_dict(), "email": row['phishing_email']} 
    for _, row in non_spam_df.iterrows()
]


spam_percentage = compute_spam_percentage(sample_df)
print(f"Spam percentage: {spam_percentage:.2f}%")


# In[1]:


get_ipython().system('pip freeze > requirements.txt')


# In[69]:


#main.ipynb to python file
get_ipython().system('jupyter nbconvert --to script main.ipynb')


# In[ ]:




