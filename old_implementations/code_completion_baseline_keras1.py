import tflearn
import numpy
from keras.models import Sequential, Model
from keras.layers import *

class Code_Completion_Baseline:
    
    batch_size = 128
    embedding_dims = 100
    layer_dim = 128
    epochs = 20
    dropout = 0.4

    def token_to_string(self, token):
        return token["type"] + "-@@-" + token["value"]
    
    def string_to_token(self, string):
        splitted = string.split("-@@-")
        return {"type": splitted[0], "value": splitted[1]}
    
    def one_hot(self, string):
        vector = [0] * len(self.string_to_number)
        vector[self.string_to_number[string]] = 1
        return vector
    
    def prepare_data(self, token_lists):
        # encode tokens into one-hot vectors
        all_token_strings = set()
        for token_list in token_lists:
            for token in token_list:
                all_token_strings.add(self.token_to_string(token))
        all_token_strings = list(all_token_strings)
        all_token_strings.sort()
        print("Unique tokens: " + str(len(all_token_strings)))
        self.string_to_number = dict()
        self.number_to_string = dict() 
        max_number = 0
        for token_string in all_token_strings:
            self.string_to_number[token_string] = max_number
            self.number_to_string[max_number] = token_string
            max_number += 1
        
        # prepare x,y pairs
        xs = []
        ys = []
        for token_list in token_lists:
            for idx, token in enumerate(token_list):
                if idx > 0:
                    token_string = self.token_to_string(token)
                    previous_token_string = self.token_to_string(token_list[idx - 1])
                    xs.append(self.one_hot(previous_token_string))
                    ys.append(self.one_hot(token_string))

        print("x,y pairs: " + str(len(xs)))        
        return (xs, ys)

    def create_network(self):
        self.model = Sequential()
        self.model.add(Dense(len(self.string_to_number), activation='relu', input_shape=(len(self.string_to_number),)))
        self.model.add(Dense(self.layer_dim, activation='relu'))
        self.model.add(Dense(len(self.string_to_number), activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()
    
    def load(self, token_lists, model_file):
        self.prepare_data(token_lists)
        self.create_network()
        self.model.load(model_file)
    
    def train(self, token_lists, model_file):
        (xs, ys) = self.prepare_data(token_lists)
        self.create_network()
        self.model.fit(xs, ys, batch_size=self.batch_size, epochs=self.epochs, verbose=2)
        #self.model.save(model_file)
        
    def query(self, prefix, suffix):
        previous_token_string = self.token_to_string(prefix[-1])
        x = self.one_hot(previous_token_string)
        y = self.model.predict(numpy.array([x]))
        predicted_seq = y[0]
        if type(predicted_seq) is numpy.ndarray:
            predicted_seq = predicted_seq.tolist() 
        best_number = predicted_seq.index(max(predicted_seq))
        best_string = self.number_to_string[best_number]
        best_token = self.string_to_token(best_string)
        return [best_token]
    
