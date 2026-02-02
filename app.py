import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import json
import random
import pickle
import numpy as np

from flask import Flask, request, jsonify, render_template

# REMOVED: tensorflow
# from tensorflow.keras.models import load_model

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
# NLTK setup
# -----------------------------
NLTK_DATA_DIR = os.path.join(BASE_DIR, "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)

nltk.download("punkt", download_dir=NLTK_DATA_DIR)
nltk.download("punkt_tab", download_dir=NLTK_DATA_DIR)
nltk.download("wordnet", download_dir=NLTK_DATA_DIR)

lemmatizer = WordNetLemmatizer()

# -----------------------------
# 🚀 NUMPY MODEL IMPLEMENTATION
# -----------------------------
class NumpyModel:
    def __init__(self, weights_path):
        print(f"Loading weights from {weights_path}...")
        data = np.load(weights_path)
        self.w1, self.b1 = data['w1'], data['b1']
        self.w2, self.b2 = data['w2'], data['b2']
        self.w3, self.b3 = data['w3'], data['b3']
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x)) # Stability fix
        return e_x / e_x.sum(axis=0)

    def predict(self, x):
        # Forward pass (Dense -> Relu -> Dropout ignored -> Dense -> Relu -> Dense -> Softmax)
        # Layer 1
        h1 = np.dot(x, self.w1) + self.b1
        h1 = self.relu(h1)
        
        # Layer 2
        h2 = np.dot(h1, self.w2) + self.b2
        h2 = self.relu(h2)
        
        # Layer 3 (Output)
        out = np.dot(h2, self.w3) + self.b3
        return self.softmax(out)

# -----------------------------
# Load chatbot files
# -----------------------------
with open(os.path.join(BASE_DIR, "intent.json"), encoding="utf-8") as f:
    intents = json.load(f)

# REPLACED: model = load_model(...)
model = NumpyModel(os.path.join(BASE_DIR, "model_weights.npz"))

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
    
    # REPLACED: res = model.predict(np.array([bow]), verbose=0)[0]
    res = model.predict(bow) # Direct numpy call

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
