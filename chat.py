import random
import json

import torch

from model import JadejaTeam
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = JadejaTeam(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "JADEJA"
print("Let's ask the medicine_info & symptom_check !!! (type 'quit' to exit)")
while True:
    sentence = input("You: ")
    if sentence == "our_team":
        for intent in intents['intents']: 
                for pattern in intent["patterns"]:
                    if "team" in intent["patterns"]:
                        print(f"{bot_name}: {random.choice(intent['responses'])}")
                        break
                    continue
    elif sentence == "symptom_check":
        print(f"{bot_name}: What kind of health issue do you have")
        print(f"{bot_name}: I have the list of datasets for the initial process \n\t\ntype anyone...\n\thigh_bp\n\twound_infection\n\tnerve_pain\n\tdry_cough\n\tcorona\n\tasthma")
        symp=input("Symptom_type: ")
        if symp == "high_bp":
            print('This might be cure your issue....')
            for intent in intents['intents']: 
                for pattern in intent["patterns"]:
                    if "eritel-am" in intent["patterns"]:
                        print(f"{bot_name}: {random.choice(intent['responses'])}")
                        break
                    continue
        elif symp == "corona":
            print('This might be cure your issue....')
            for intent in intents['intents']: 
                for pattern in intent["patterns"]:
                    if "remdesivir" in intent["patterns"]:
                        print(f"{bot_name}: {random.choice(intent['responses'])}")
                        break
                    continue
        elif symp == "wound_infection":
            print('This might be cure your issue....')
            for intent in intents['intents']: 
                for pattern in intent["patterns"]:
                    if "cipladine" in intent["patterns"]:
                        print(f"{bot_name}: {random.choice(intent['responses'])}")
                        break
                    continue
        elif symp == "nerve_pain":
            print('This might be cure your issue........')
            for intent in intents['intents']: 
                for pattern in intent["patterns"]:
                    if "mazetol-200" in intent["patterns"]:
                        print(f"{bot_name}: {random.choice(intent['responses'])}")
                        break
                    continue
        elif symp == "asthma":
            print('This might be cure your issue........')
            for intent in intents['intents']: 
                for pattern in intent["patterns"]:
                    if "VentorlinInhaler" in intent["patterns"]:
                        print(f"{bot_name}: {random.choice(intent['responses'])}")
                        break
                    continue
        elif symp == "dry_cough":
            print('This might be cure your issue........')
            for intent in intents['intents']: 
                for pattern in intent["patterns"]:
                    if "Mucolite-tab" in intent["patterns"]:
                        print(f"{bot_name}: {random.choice(intent['responses'])}")
                        break
                    continue                                        
    elif sentence == "medicine_info":
        print("We have contained the medicines informations are \n\tcipladine \n\teritel-am \n\tmazetol-200 \n\tmucolite\n\tVentorlinInhaler\n\tremdesivir")
        med_data=input("med_name: ")
        for intent in intents['intents']: 
            for pattern in intent["patterns"]:
                if med_data in intent["patterns"]:
                    print(f"{bot_name}: {random.choice(intent['responses'])}")
                    break
                continue
    elif sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted =   torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print('')
