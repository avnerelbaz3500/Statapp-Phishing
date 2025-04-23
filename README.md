#  Phishing Email Generation & Detection

This project combines classical NLP and modern generative AI to simulate adversarial email scenarios for cybersecurity research. It allows you to:

1.  Train a **Naive Bayes classifier** on labeled phishing email datasets.  
2.  Generate **contextual phishing emails** using LLMs like GPT or Mistral.  
3.  Analyze emails through visualization (wordclouds, Zipf's law, subjectivity, etc.).  
4.  Evaluate model robustness through adversarial examples.

---

##  Project Structure

\`\`\`
Phishing/
│
├── apps/
│   └── app.py                 # Streamlit interface to test email classification
│
├── data/                      # CSV datasets
│   ├── adult_data.csv
│   ├── Enron.csv
│   ├── CEAS_08.csv
│   ├── Nigerian_Fraud.csv
│   ├── phishing_email.csv
│   └── merged_data.csv
│
├── models/
│   ├── vectorizer.pkl         # CountVectorizer saved
│   ├── multinomial_nb_model.pkl
│   └── filtered_phishing_emails.json
│
├── venv/                      # Virtual environment
│
├── main.ipynb                 # Jupyter Notebook for full pipeline
├── main.py                    # Converted version of main.ipynb
├── azure_config.json          # Azure OpenAI config (not versioned)
├── mistral_key.txt            # Mistral API key (not versioned)
├── README.md
├── LICENSE
├── .gitignore
└── requirements.txt           # All dependencies
\`\`\`

---

##  Installation

1. **Clone the repo:**
\`\`\`bash
git clone https://github.com/avnerelbaz3500/Statapp-Phishing/
cd phishing
\`\`\`

2. **Create a virtual environment (optional but recommended):**
\`\`\`bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\\Scripts\\activate
\`\`\`

3. **Install dependencies:**
\`\`\`bash
pip install -r requirements.txt
\`\`\`

4. **Set up API keys:**
- Create a file \`azure_config.json\`:
\`\`\`json
{
  \"AZURE_OPENAI_ENDPOINT\": \"https://<your-endpoint>.openai.azure.com/\",
  \"AZURE_OPENAI_API_KEY\": \"<your-key>\",
  \"AZURE_DEPLOYMENT_NAME\": \"<your-deployment-name>\",
  \"AZURE_API_VERSION\": \"2023-12-01-preview\"
}
\`\`\`

- Create a file \`mistral_key.txt\` with your Mistral API key inside.

---

## ▶ Run the Streamlit App

From the root of the project, launch the app using:

\`\`\`bash
streamlit run apps/app.py
\`\`\`

This will open a local web interface for testing your emails against the trained classifier.

---

##  Notebooks

Use \`main.ipynb\` for:
- Data loading & merging  
- Preprocessing (cleaning, lemmatization, etc.)  
- Training & evaluating the Naive Bayes classifier  
- Visual analytics (wordclouds, polarity, etc.)  
- Generating adversarial phishing emails using LLMs  
- Filtering successful evasion examples  

---

##  Visualizations

The project includes:
- Wordclouds for legitimate vs phishing emails  
- Zipf law plots (frequency vs rank)  
- KDE plots for subjectivity and polarity  
- Temporal distribution of phishing attempts  

---

##  Datasets

The project combines real datasets:
- [Enron Spam](https://www.cs.cmu.edu/~enron/)  
- [SpamAssassin](https://spamassassin.apache.org/)  
- [Nigerian Scam Corpus](https://www.unodc.org/)  
- [UCI Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)  

---


##  Contributing

Pull requests welcome! For major changes, please open an issue first to discuss what you would like to change.

---

##  Disclaimer

This project is for **academic and cybersecurity research purposes only**. Generated phishing emails must never be used for malicious purposes."
