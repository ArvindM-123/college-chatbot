print("Python is running correctly!")

import json
import random

with open("intent.json") as file:
    data = json.load(file)


def get_response(user_input):
    user_input = user_input.lower()
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            if pattern.lower() in user_input:
                return random.choice(intent["responses"])
    return "Sorry, I didn't understand that"

print("Chatbot is running! (type 'quit' to exit)")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("Bye! Have a nice day")
        break
    response = get_response(user_input)
    print("Bot:", response)
