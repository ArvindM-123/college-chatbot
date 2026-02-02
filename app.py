import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import json
import random
import pickle
import numpy as np

from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model

import nltk
from nltk.stem import WordNetLemmatizer

# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------
# NLTK setup (IMPORTANT FOR DEPLOYMENT)
# -----------------------------
NLTK_DATA_DIR = os.path.join(BASE_DIR, "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)

nltk.data.path.append(NLTK_DATA_DIR)

# Download only once at startup
nltk.download("punkt", download_dir=NLTK_DATA_DIR)
nltk.download("wordnet", download_dir=NLTK_DATA_DIR)

lemmatizer = WordNetLemmatizer()

# -----------------------------
# Load chatbot files
# -----------------------------
with open(os.path.join(BASE_DIR, "intent.json"), encoding="utf-8") as f:
    intents = json.load(f)

model = load_model(os.path.join(BASE_DIR, "model.h5"))
words = pickle.load(open(os.path.join(BASE_DIR, "words.pkl"), "rb"))
classes = pickle.load(open(os.path.join(BASE_DIR, "classes.pkl"), "rb"))

# -----------------------------
# NLP functions
# -----------------------------
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)

    for sw in sentence_words:
        for i, word in enumerate(words):
            if word == sw:
                bag[i] = 1

    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0]

    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []
    for r in results:
        return_list.append({
            "intent": classes[r[0]],
            "probability": str(r[1])
        })

    return return_list


def get_response(intents_list, intents_json):
    if not intents_list:
        return "Sorry, I didn't understand that"

    tag = intents_list[0]["intent"]

    for intent in intents_json["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "Sorry, something went wrong."

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()

        if not data or "message" not in data:
            return jsonify({"reply": "Invalid request"}), 400

        user_message = data["message"]

        intents_list = predict_class(user_message)
        response = get_response(intents_list, intents)

        return jsonify({"reply": response})

    except Exception as e:
        print(f"Error: {e}", flush=True)
        return jsonify({"reply": f"Internal Error: {str(e)}"}), 200

# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
