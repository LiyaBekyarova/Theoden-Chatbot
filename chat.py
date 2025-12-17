
import random
import json
import re
import torch
from model import NeuralNet
from nltk_utils import tokenize, bag_of_words
from weather_utils import get_real_weather

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
CONFIDENCE_THRESHOLD = 0.65

def extract_city(msg):
    """Extract city name from user message."""
    # Common patterns: "weather in Paris", "weather for London", "Paris weather", etc.
    patterns = [
        r'(?:weather|temperature|forecast)\s+(?:in|for|at)\s+([A-Za-z\s]+)',
        r'([A-Za-z\s]+)\s+weather',
        r"what'?s?\s+(?:the\s+)?weather\s+(?:like\s+)?(?:in|at|for)\s+([A-Za-z\s]+)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, msg, re.IGNORECASE)
        if match:
            city = match.group(1).strip()
            # Remove common words that aren't cities
            stopwords = ['the', 'a', 'an', 'like', 'today', 'tomorrow']
            city_words = [w for w in city.split() if w.lower() not in stopwords]
            if city_words:
                return ' '.join(city_words)
    
    return None

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > CONFIDENCE_THRESHOLD:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                if tag == "weather":
                    city = extract_city(msg)
                    if city:
                        weather_report = get_real_weather(city)
                        return weather_report
                    else:
                        weather_report = get_real_weather("Varna")
                        return f"Of what land dost thou seek tidings? I shall speak of Varna for now:\n{weather_report}"
                else:
                    response = random.choice(intent['responses'])
                    return response
                
    else:
        return "Thy words are strange to me, wayfarer. Speak plainer, or teach me thy tongue."

if __name__ == "__main__":
    print("Harken, rider of distant lands! Let us hold counsel together in speech. When thou wouldst depart, say only 'quit'.")
    
    while True:
        sentence = input("You: ").strip()
        
        if sentence.lower() in ["quit", "farewell", "goodbye", "i ride hence"]:
            print(f"{bot_name}: Fare thee well, rider of the Mark! Ride swift, ride true — may the wind ever fill thy sails. Westu hál!")
            break

        response = get_response(sentence)
        print(f"{bot_name}: {response}")