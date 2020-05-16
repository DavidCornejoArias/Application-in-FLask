import nltk
nltk.download('punkt')
import numpy as np
#import tensorflow as tf
#tensorflow version:
#https://pip


import random, json, string, pickle
from nltk.stem.lancaster import LancasterStemmer
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Activation, Dropout

# Trying to make this thing work well
class modelsProcess:
    def __init__(self, json_fileAdress, weightsAdress):
        self.json_file = open(json_fileAdress, 'r')
        self.weightsAdress = weightsAdress

    def modelPreprocessing(self):
        loaded_model_json = self.json_file.read()
        self.json_file.close()

        loaded_model = model_from_json(loaded_model_json)
        print("Loaded model architecture")

        loaded_model.load_weights(self.weightsAdress)
        print("Loaded model from disk")

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        loaded_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        return loaded_model

class testingPreprocessing:
    def __init__(self, testJsonAdress):
        with open(testJsonAdress) as file:
            self.data = json.load(file)

    def processing(self):
        stemmer = LancasterStemmer()
        words, labels, doc_x, doc_y = [], [], [], []
        # from patterns in intent, tokenize that shit, add to words and doc_X and add intent[tag]
        for intent in self.data["intents"]:
            for pattern in intent["patterns"]:
                
                word_token = nltk.word_tokenize(pattern)
                words.extend(word_token)
                doc_x.append(word_token)
                doc_y.append(intent["tag"])

                if intent["tag"] not in labels:
                    labels.append(intent["tag"])
        punctuation = list(string.punctuation)
        words = [stemmer.stem(w.lower()) for w in words if w not in punctuation]
        words = sorted(list(set(words)))
        labels = sorted(labels)

        return words, labels, self.data["intents"], doc_x, doc_y
    
    def trainingData(self,doc_x, doc_y, words, labels):
        out_empty = [0] * len(labels)
        stemmer = LancasterStemmer()
        training, output = [], []
        for x, doc in enumerate(doc_x):
            bag = []
            word_stems = [stemmer.stem(w.lower()) for w in doc]
            for w in words:
                bag.append(1) if w in word_stems else bag.append(0)
            
            output_row = out_empty[:]
            output_row[labels.index(doc_y[x])] = 1 # check if it's the correct tag

            training.append(bag)
            output.append(output_row)

        training = np.array(training) # shape (26, 46)
        output = np.array(output) # shape (26, 6)

        with open("data_sp.pickle", "wb") as f:
            pickle.dump((words, labels, training, output), f)
        print("picke done")

        return training, output
    
    def training(self, trainingData2, outputData, modelName, weightName):
        model = Sequential()
        model.add(Dense(128, input_shape=(len(trainingData2[0]),), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(outputData[0]), activation='softmax'))
        # Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        hist = model.fit(trainingData2, outputData, epochs=100, batch_size=5, verbose=1)
        # serialize model to json, cb_model_sp2.json
        model_json = hist.model.to_json()
        with open(modelName, "w") as json_file:
            json_file.write(model_json)
            print('Saved model architecture.')

        # serialize the weights, cb_weights_sp2
        model.save_weights(weightName, hist)
        print("Saved model weights.")

# turn sentences of the user into bag of words
class chatting:
    def __init__(self, text, words):
        self.text = text
        self.words = words
    """docstring for chatting"""

    def bag_of_words(self):
        bag = [0] * len(self.words)
        stemmer = LancasterStemmer()
        s_words = nltk.word_tokenize(self.text)
        s_words = [stemmer.stem(word.lower()) for word in s_words]

        for word in s_words:
            for i, w in enumerate(self.words):
                if w == word:
                    bag[i] = 1

        return np.array(bag)

    # user interaction
    def chat(self, labels, model, intents):
        #print("Start talking with the bot!")
        #print("Type quit to stop.")
        #while True:
        user_text = self.text
        if user_text.lower() == "quit":
            return "Bye, bye." 

        input_data = self.bag_of_words()
        input_data = np.reshape(input_data, (1, input_data.shape[0]))
        results = model.predict(input_data)[0] # this gives a probability
        result_index = np.argmax(results)
        tag = labels[result_index]

        if results[result_index] > 0.7:
            for t in intents:
                if t["tag"] == tag:
                    responses = t["responses"]

            return str(random.choice(responses))
        else:
            return "Sorry, I didn't get that, try gain."
    #chat()