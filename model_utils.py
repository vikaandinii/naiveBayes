import pickle
import os
from preprocess import preprocess

base_path = r"C:\ml"  # ganti sesuai folder

with open(os.path.join(base_path, "model_nb_pln.pkl"), 'rb') as f:
    nb = pickle.load(f)

with open(os.path.join(base_path, "vectorizer_pln.pkl"), 'rb') as f:
    vectorizer = pickle.load(f)

def predict_text(text):
    text_proc = preprocess(text)
    vec = vectorizer.transform([text_proc])
    return nb.predict(vec)[0]