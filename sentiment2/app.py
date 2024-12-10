from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, ArrayDictionary

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Load model dan vectorizer
MODEL_PATH = "model_sentiment.pkl"  # Sesuaikan dengan lokasi model
VECTORIZER_PATH = "vectorizer.pkl"  # Sesuaikan dengan lokasi vectorizer

with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

with open(VECTORIZER_PATH, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Stopword remover
stop_words = StopWordRemoverFactory().get_stop_words()
new_array = ArrayDictionary(stop_words)
stop_words_remover = StopWordRemoverFactory().create_stop_word_remover()

# Stemming
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

# Normalisasi
norm = {" dgn ": " dengan ", " gue ": " saya ", " tdk ": " tidak ", " blum ": " belum ", " josss ": " bagus "}

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Normalisasi
    for key, value in norm.items():
        text = text.replace(key, value)
    # Stopword removal
    text = stop_words_remover.remove(text)
    # Stemming
    text = stemmer.stem(text)
    return text

# Halaman utama
@app.route('/')
def home():
    return render_template('home.html')

# API untuk analisis sentimen
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil input teks
        user_input = request.form.get('text')

        if not user_input:
            return jsonify({'error': 'Teks tidak boleh kosong!'}), 400

        # Preprocessing
        processed_text = preprocess_text(user_input)

        # Vectorisasi teks
        user_input_vectorized = vectorizer.transform([processed_text])

        # Konversi ke array (karena GaussianNB membutuhkan array)
        user_input_array = user_input_vectorized.toarray()

        # Prediksi
        prediction = model.predict(user_input_array)

        # Mapping hasil prediksi
        sentiment = "Positif" if prediction[0] == 1 else "Negatif"

        return render_template("result.html", user_input=user_input, sentiment=sentiment)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
