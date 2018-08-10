import tflearn
import numpy
from keras.models import Sequential, Model
from keras.layers import *
from keras.callbacks import EarlyStopping
from keras.models import load_model

class Code_Completion_Baseline:
    
    batch_size = 64
    layer_dim = 128
    epochs = 40
    dropout = 0.3
    nr_considered_words_prefix = 7
    verbose = 2

    def token_to_string(self, token):
        return token["type"] + "-@@-" + token["value"]
    
    def string_to_token(self, string):
        splitted = string.split("-@@-")
        return {"type": splitted[0], "value": splitted[1]}
    
    def one_hot(self, string):
        vector = [0] * len(self.string_to_number)
        vector[self.string_to_number[string]] = 1
        return vector
    
    def create_token_dicts(self, token_lists):
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
        
    
    def prepare_data(self, token_lists):

        # prepare x,y pairs
        xs = []
        ys = []
        samples = 0
        for token_list in token_lists:
            
            # Create a matrix of the five last words, every word with the dimension of the len(all_token_list)
            for idx, token in enumerate(token_list):
                if idx > self.nr_considered_words_prefix:
                    
                    x = numpy.zeros((self.nr_considered_words_prefix , len(self.string_to_number)))
                    row_idx = 0
                    for i in range(idx - self.nr_considered_words_prefix, idx):
                        i_token_string = self.token_to_string(token_list[i])
                        x [row_idx,:] = self.one_hot(i_token_string)
                        row_idx += 1
                    
                    xs.append(x)
                    current_token_string = self.token_to_string(token)
                    ys.append(numpy.array(self.one_hot(current_token_string)))
                    samples +=1

        #transform into a numpy array
        final_xs = numpy.zeros((samples, self.nr_considered_words_prefix , len(self.string_to_number)))
        final_ys = numpy.zeros((samples , len(self.string_to_number)))
        for i, entry in enumerate(xs):
            final_xs[i,:,:] = entry
        for i, entry in enumerate(ys):
            final_ys[i,:] = entry
            
        print("x,y pairs: " + str(len(xs)))     
        #print("x Shape: {}".format(numpy.shape(xs)))   
        return (final_xs, final_ys)

    def create_network(self):
        self.model = Sequential()
        self.model.add(LSTM(128, activation='relu', input_shape=(self.nr_considered_words_prefix, len(self.string_to_number),)))
        self.model.add(Dropout(self.dropout))
        self.model.add(Dense(len(self.string_to_number), activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()
        
    
    def load(self, token_lists, model_file):
        self.create_token_dicts(token_lists)
        #self.prepare_data(token_lists)
        #self.create_network()
        self.model = load_model("C:/Users/alexa/dl_software/model/lstm.model")
    
    def train(self, token_lists,test_token_list, model_file):
        self.create_token_dicts(numpy.append(token_lists, test_token_list))
        (xs, ys) = self.prepare_data(token_lists)
        (xs_test, ys_test) = self.prepare_data(test_token_list)
        self.create_network()
        early_stopping = EarlyStopping(monitor='val_acc', patience=2)
        self.model.fit(xs, ys, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose, validation_data=(xs_test, ys_test), callbacks=[early_stopping])
        self.model.save("C:/Users/alexa/dl_software/model/lstm.model")
        
    def query(self, prefix, suffix):
        # take last n positions of prefix as input for the model
        query_array = numpy.zeros((1, self.nr_considered_words_prefix,len(self.string_to_number)))
        # If prefix is smaller than the numbers that we need to consider, we just use zeros as padding
        len_prefix = len(prefix)
        len_query_array = self.nr_considered_words_prefix
        
        max_iterations = self.nr_considered_words_prefix
        if len(prefix) < self.nr_considered_words_prefix:
            max_iterations = len(prefix)

        for i in range(0, max_iterations):
            i_token_string = self.token_to_string(prefix[len_prefix - i -1])
            query_array[0, len_query_array - i -1 , ] = self.one_hot(i_token_string)
        
        y = self.model.predict(query_array)
        predicted_seq = y[0]
        if type(predicted_seq) is numpy.ndarray:
            predicted_seq = predicted_seq.tolist() 
        best_number = predicted_seq.index(max(predicted_seq))
        best_string = self.number_to_string[best_number]
        best_token = self.string_to_token(best_string)
        return [best_token]
    
