#-----------------------------------------------------
# Bibliotheken importieren
#-----------------------------------------------------
import pandas as pd
import numpy as np
import nltk
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import TruncatedSVD

#-----------------------------------------------------
# NLTK-Ressourcen herunterladen
#-----------------------------------------------------
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

#-----------------------------------------------------
# Datensatz laden
#-----------------------------------------------------
df = pd.read_csv("amazon_alexa.tsv", sep="\t")

#-----------------------------------------------------
# Texte vorverarbeiten
#-----------------------------------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):

    # Kleinbuchstaben
    text = text.lower()

    # Sonderzeichen entfernen
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenisierung
    tokens = word_tokenize(text)

    # Nur Wörter behalten
    tokens = [word for word in tokens if word.isalpha()]

    # Stopwörter entfernen
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatisierung
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)

df["cleaned_reviews"] = df["verified_reviews"].astype(str).apply(preprocess_text)


negative_reviews = df[df["rating"] <= 2]

df[["verified_reviews", "cleaned_reviews"]].head()

#-----------------------------------------------------
# BoW erstellen
#-----------------------------------------------------
bow_vectorizer = CountVectorizer()
bow_matrix = bow_vectorizer.fit_transform(negative_reviews["cleaned_reviews"])

#-----------------------------------------------------
# TF-IDF erstellen
#-----------------------------------------------------
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(negative_reviews["cleaned_reviews"])

#-----------------------------------------------------
# LDA mit BoW durchführen
#-----------------------------------------------------
lda_model = LatentDirichletAllocation(
    n_components=5,
    random_state=42
)

lda_model.fit(bow_matrix)

wordsLDA = bow_vectorizer.get_feature_names_out()

print("LDA:")
for index, topic in enumerate(lda_model.components_):

    print(f"\nThema {index + 1}")

    top_wordsLDA = [wordsLDA[i] for i in topic.argsort()[-10:]]

    print(top_wordsLDA)

#-----------------------------------------------------
# LSA mit TF-IDF durchführen
#-----------------------------------------------------
lsa_model = TruncatedSVD(
    n_components=5,
    random_state=42
)

lsa_model.fit(tfidf_matrix)

wordsLSA = tfidf_vectorizer.get_feature_names_out()

print("LSA:")
for index, topic in enumerate(lsa_model.components_):

    print(f"\nThema {index + 1}")

    top_wordsLSA = [wordsLSA[i] for i in topic.argsort()[-10:]]

    print(top_wordsLSA)
