from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import json
import random
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# -------- NLTK SETUP --------
nltk.download('punkt')
nltk.download('wordnet')

# -------- APP SETUP --------
app = Flask(__name__, template_folder="templates")

lemmatizer = WordNetLemmatizer()

# -------- LOAD FILES --------
model = load_model("model.h5")
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

with open("intent.json", encoding="utf-8") as f:
    intents = json.load(f)

# -------- HELPER FUNCTIONS --------
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]


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
    res = model.predict(np.array([bow]), verbose=0)[0]

    max_prob = float(np.max(res))
    max_index = int(np.argmax(res))

    CONFIDENCE_THRESHOLD = 0.75  # 🔥 this fixes random replies

    if max_prob < CONFIDENCE_THRESHOLD:
        return []   # unknown input → no intent

    return [{
        "intent": classes[max_index],
        "probability": max_prob
    }]

def get_response(intents_list):
    if not intents_list:
        return "Sorry 😅 I don’t understand that. Try asking about college."

    tag = intents_list[0]["intent"]

    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "Sorry 😕 something went wrong."


# -------- ROUTES --------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get", methods=["POST"])
def chatbot_response():
    msg = request.form["msg"]
    ints = predict_class(msg)
    response = get_response(ints)
    return jsonify(response)   # ✅ ONLY ONE RETURN

# -------- RUN --------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
