# preprocess.py
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords', quiet=True)

factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Stopwords bahasa Indonesia, jangan hapus negasi
stopwords_id = set(stopwords.words('indonesian'))
negation_words = {'tidak', 'tak', 'gak', 'ga', 'nggak', 'kurang'}
stopwords_id = stopwords_id - negation_words

# ---------- Fungsi preprocessing ----------
def cleaning(text):
    text = re.sub(r'http\S+', '', str(text))
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower().strip()

def handle_negation(text):
    return re.sub(r'\b(tidak|tak|gak|ga|nggak|kurang)\s+(\w+)', r'\1_\2', text)

def handle_intensifier(text):
    return re.sub(r'\b(sangat|banget|sekali)\s+(\w+)', r'\1_\2', text)

def preprocess(text):
    text = cleaning(text)
    text = handle_negation(text)
    text = handle_intensifier(text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stopwords_id]
    tokens = [stemmer.stem(w) for w in tokens]
    return ' '.join(tokens)