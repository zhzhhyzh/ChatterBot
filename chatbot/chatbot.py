# https://chatbotsmagazine.com/contextual-chat-bots-with-tensorflow-4391749d0077
import nltk, os
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import pandas as pd
import spacy
# Load spaCy's English NLP model
nlp = spacy.load("en_core_web_sm")


import numpy as np 
import tflearn
import tensorflow as tf
import random
import json
import pickle
import uuid
try: 
    with open("intents.json") as file:
        data = json.load(file)
except:
    
    df = pd.read_parquet("hf://datasets/goendalf666/sales-conversations/data/train-00000-of-00001-facf96cb8c12ba2c.parquet")
        
    def is_computer_related(sentence):
        keywords = ["computer", "laptop", "desktop", "CPU", "RAM", "monitor", 
                    "mouse", "keyboard", "graphics card", "GPU", "SSD", "hard drive"]
        
        sentence_lower = sentence.lower()
        
        for keyword in keywords:
            if keyword.lower() in sentence_lower.split(" "):
                # print (keyword.lower())
                return True
        return False

    # Function to extract verbs and noun phrases from a sentence
    def extract_keywords(text):
        doc = nlp(text)
        verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]
        return verbs, noun_phrases

       

    intents = []

    for idx, row in df.iterrows():
        currentContext = str(uuid.uuid4()) # fix conversation flow, 1 row = 1 conversation

        if row[str("0")] and is_computer_related(row[str("0")]):
            for i in range(0, len(df.columns)-1, 2):
                if row[str(i)] :
                    verbs, noun_phrases = extract_keywords(row[str(i)].replace("Customer: ",""))
                    # Combine verbs and noun phrases to form a simple intent
                    if verbs and noun_phrases:
                        intent = f"{verbs[0]}_{'_'.join(noun_phrases).replace(' ', '_')}"
                    elif verbs:
                        intent = f"{verbs[0]}"
                    elif noun_phrases:
                        intent = f"{'_'.join(noun_phrases).replace(' ', '_')}"
                    else:
                        intent = "unknown_intent"
                    isExist = False
                    for currentIntent in intents:
                        if currentIntent["tag"]== intent:
                            currentIntent["patterns"].append(row[str(i)].replace("Customer: ",""))
                            currentIntent["responses"].append(row[str(i+1)].replace("Salesman: ",""))
                            if i == 0: 
                                currentIntent["context_set"].append(currentContext)
                            else:
                                currentIntent["context_filter"].append(currentContext)

                            isExist = True
                            break
                    if not isExist:
                        if i == 0: 
                            intents.append({"tag":intent, "patterns":[row[str(i)].replace("Customer: ","")],"responses":[row[str(i+1)].replace("Salesman: ","")],"context_set":[currentContext]})
                        else:
                            intents.append({"tag":intent, "patterns":[row[str(i)].replace("Customer: ","")],"responses":[row[str(i+1)].replace("Salesman: ","")],"context_filter":[currentContext]})

    intents = { "intents": intents}                    
    # Optionally, save the intents to a JSON file
    import json
    with open('intents.json', 'w') as f:
        json.dump(intents, f, indent=4)
    with open("intents.json") as file:
        data = json.load(file)


try:
    with open("data.pickle","rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
            for pattern in intent["patterns"]:
                wrds = nltk.word_tokenize(pattern)
                words.extend(wrds)
                docs_x.append(wrds)
                docs_y.append(intent["tag"])
                
            if intent["tag"] not in labels:
                labels.append(intent["tag"])
                
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    # preprocessing
    training = []
    output = []
    out_empty = [0 for _ in range(len(labels))]

    # one hot encoded
    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        
        training.append(bag)
        output.append(output_row)


    training = np.array(training)
    output = np.array(output)
    
    with open("data.pickle","wb") as f:
        pickle.dump((words, labels, training, output), f)


# modeling
tf.compat.v1.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net)

if [os.path.isfile(i) for i in ["model.tflearn.meta", "model.tflearn.index"]] == [True, True]:
    model.load("model.tflearn")
else:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    
    return np.array(bag)

# def chat():
#     print("Start talking with the bot! (type quit to stop)")
#     while True:
#         inp = input("You: ")
#         if inp.lower() == "quit":
#             break

#         result = model.predict([bag_of_words(inp, words)])[0]
#         result_index = np.argmax(result)
#         tag = labels[result_index]
#         if result[result_index] > 0.7:
#             for tg in data["intents"]:
#                 if tg['tag'] == tag:
#                     responses = tg['responses']
#             print("Bot:",random.choice(responses))

#         else:
#             print("Bot: I didnt get that. Can you explain or try again.")
def generate_response(user_input, context):
        result = model.predict([bag_of_words(user_input, words)])[0]
        result_index = np.argmax(result)
        tag = labels[result_index]
        if result[result_index] > 0.7:
            responseFound = False
            for tg in data["intents"]:
                # if "context_filter" in tg:
                #     print(context)
                #     print(context in tg['context_filter'])
                # print(context)
                # print(tag)
                if tg['tag'] == tag and (context=="" or(  "context_filter" in tg  and context in tg['context_filter'])):

                    index = -1
                    responses = tg['responses']
                    # print(responses)
                    if context =="":
                        index = random.randrange(0, len(responses))
                        if 'context_set' in tg:
                            context = tg['context_set'][index]
                    else:
                        index = tg['context_filter'].index(context)
                    

                    responseFound = True
                
            if responseFound:
                return  json.dumps({"res":responses[index], "context":context})
            else:
                return  json.dumps({"res":"I didnt get that. Can you explain or try again.","context":context})
        else:
            return  json.dumps({"res":"I didnt get that. Can you explain or try again.","context":context})
            
            


