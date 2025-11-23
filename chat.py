
import random
import json
import torch
from model import NeuralNet
from nltk_utils import tokenize, bag_of_words
from weather_utils import get_real_weather
import warnings
from urllib3.exceptions import NotOpenSSLWarning
warnings.filterwarnings('ignore', category=NotOpenSSLWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents and model
with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Théoden"
print("Harken, rider of distant lands! Let us hold counsel together in speech. When thou wouldst depart, say only 'quit'.")

while True:
    sentence = input("You: ").strip()
    
    if sentence.lower() in ["quit", "farewell", "goodbye", "i ride hence"]:
        print(f"{bot_name}: Fare thee well, rider of the Mark! Ride swift, ride true — may the wind ever fill thy sails. Westu hál!")
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                if tag == "weather":
                    print(f"{bot_name}: Of what land dost thou seek tidings? Speak the name of the city!")
                    city = input("You: ").strip()
                    if not city:
                        city = "Edoras"
                    weather_report = get_real_weather(city)
                    print(f"{bot_name}: {weather_report}")
                else:
                    response = random.choice(intent['responses'])
                    print(f"{bot_name}: {response}")
                break
    else:
        print(f"{bot_name}: Thy words are strange to me, wayfarer. Speak plainer, or teach me thy tongue.")