# ================== DISABLE GPU ==================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ================== IMPORTS ==================
import nltk
import json
import pickle
import random
import traceback
import numpy as np
from flask import Flask, render_template, request, jsonify
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# ================== NLTK SETUP ==================
NLTK_DATA_DIR = "/tmp/nltk_data"
nltk.data.path.append(NLTK_DATA_DIR)

nltk.download("punkt", download_dir=NLTK_DATA_DIR)
nltk.download("wordnet", download_dir=NLTK_DATA_DIR)

# ================== PATH SETUP ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ================== APP SETUP ==================
app = Flask(__name__, template_folder="templates")
lemmatizer = WordNetLemmatizer()

# ================== LOAD FILES ==================
model = load_model(os.path.join(BASE_DIR, "model.h5"))
words = pickle.load(open(os.path.join(BASE_DIR, "words.pkl"), "rb"))
classes = pickle.load(open(os.path.join(BASE_DIR, "classes.pkl"), "rb"))

with open(os.path.join(BASE_DIR, "intent.json"), encoding="utf-8") as f:
    intents = json.load(f)

# ================== NLP FUNCTIONS ==================
def clean_up_sentence(sentence):
    tokens = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(w.lower()) for w in tokens]

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    result = model.predict(np.array([bow]), verbose=0)[0]

    max_prob = float(np.max(result))
    max_index = int(np.argmax(result))

    if max_prob < 0.75:
        return []

    return [{"intent": classes[max_index], "probability": max_prob}]

def get_response(intents_list):
    if not intents_list:
        return "Sorry 😅 I don’t understand that. Try asking about the college."

    tag = intents_list[0]["intent"]
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "Sorry 😕 something went wrong."

# ================== ROUTES ==================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True, silent=True)

        if not data or "message" not in data:
            return jsonify({"reply": "Please type a message 🙂"})

        msg = data["message"].strip()

        if msg == "":
            return jsonify({"reply": "Please type a message 🙂"})

        ints = predict_class(msg)
        reply = get_response(ints)

        return jsonify({"reply": reply})

    except Exception:
        print("❌ CHAT ERROR")
        traceback.print_exc()
        return jsonify({"reply": "Server error. Please try again."}), 500

# ================== RUN ==================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
