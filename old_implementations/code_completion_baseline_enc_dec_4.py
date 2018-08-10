import tflearn
import numpy
from keras.models import Sequential, Model
from keras.layers import *
from keras.callbacks import EarlyStopping
from keras.models import load_model
import seq2seq
from seq2seq.models import *
from keras.callbacks import ModelCheckpoint



class Code_Completion_Baseline:
    
    batch_size = 64
    layer_dim = 128
    epochs = 5
    dropout = 0.3
    nr_considered_words_prefix = 5
    nr_considered_words_suffix = 5
    nr_considered_words_total = nr_considered_words_prefix + nr_considered_words_suffix
    verbose = 1
    hole_size = 1

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
        '''
        Method to prepare that data so that it can be used in the neural network.
        It will create x and y (=labels) pairs from full Javascript source code.
        It will do it like the following:
        
        x consists of n prefix, and m suffix words.
        prefix =  words before a possible missing code part
        suffix = words after a possible missing code part
        
        y = missing code part
        '''
        # prepare x,y pairs
        xs = []
        ys = []
        samples = 0
        for token_list in token_lists:
            
            # Create a matrix of the five last words, every word with the dimension of the len(all_token_list)
            for idx, token in enumerate(token_list):
                has_token_list_enough_prefixes = idx > self.nr_considered_words_prefix
                has_token_list_enough_suffixes = (idx + self.nr_considered_words_suffix) < len(token_list)
                if has_token_list_enough_prefixes and has_token_list_enough_suffixes:
                    # Create emtpy array with the length of considered words for prefix and suffix and length of one-hot vectors
                    x = numpy.zeros((self.nr_considered_words_total , len(self.string_to_number)))
                    row_idx = 0
                    # Fill array with values for prefix
                    for i in range(idx - self.nr_considered_words_prefix, idx):
                        i_token_string = self.token_to_string(token_list[i])
                        x [row_idx,:] = self.one_hot(i_token_string)
                        row_idx += 1
                        
                    # Fill array with values for suffix
                    for i in range(idx + self.hole_size, idx + self.nr_considered_words_suffix + self.hole_size):
                        i_token_string = self.token_to_string(token_list[i])
                        x [row_idx,:] = self.one_hot(i_token_string)
                        row_idx += 1
                    
                    xs.append(x)
                    current_token_string = self.token_to_string(token)
                    ys.append(numpy.array(self.one_hot(current_token_string)))
                    samples +=1


        #transform into a numpy array
        final_xs = numpy.zeros((samples, self.nr_considered_words_total , len(self.string_to_number)))
        final_ys = numpy.zeros((samples , 1, len(self.string_to_number)))
        for i, entry in enumerate(xs):
            final_xs[i,:,:] = entry
        for i, entry in enumerate(ys):
            final_ys[i,:,:] = entry
            
        print("x,y pairs: " + str(len(xs)))     
        #print("x Shape: {}".format(numpy.shape(xs)))   
        return (final_xs, final_ys)

    def create_network(self):
        #self.model = Sequential()
        
        # One-direcitonal LSTM
        #self.model.add(LSTM(128, activation='relu', input_shape=(self.nr_considered_words_total, len(self.string_to_number),)))
        # Bi-directional LSTM
        #self.model.add(Bidirectional(LSTM(128, activation='relu'), input_shape=(self.nr_considered_words_total, len(self.string_to_number),)))
        #self.model.add(Dropout(self.dropout))
        #self.model.add(Dense(len(self.string_to_number), activation='softmax'))
        
        
        self.model = AttentionSeq2Seq(input_shape=(self.nr_considered_words_total, len(self.string_to_number),) , hidden_dim=128, output_length=1, output_dim=len(self.string_to_number))
        self.model.add(Dense(len(self.string_to_number), activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        #self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        #model = SimpleSeq2Seq(input_dim=5, hidden_dim=10, output_length=8, output_dim=8)
        self.model.summary()
        
    
    def load(self, token_lists, model_file):
        self.create_token_dicts(token_lists)
        self.model = load_model(model_file)
    
    def train(self, token_lists,test_token_list, model_file):
        self.create_token_dicts(numpy.append(token_lists, test_token_list))
        (xs, ys) = self.prepare_data(token_lists)
        
        print(numpy.shape(xs))
        print(numpy.shape(ys))
        print(ys[0])
        (xs_test, ys_test) = self.prepare_data(test_token_list)
        self.create_network()
        
        checkpoint = ModelCheckpoint(model_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        
        #early_stopping = EarlyStopping(monitor='val_acc', patience=3)
        self.model.fit(xs, ys, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose, validation_data=(xs_test, ys_test),
                        callbacks=[checkpoint
                            #early_stopping
                            ])
        
        self.model.save(model_file)
        
    def query(self, prefix, suffix):
        # take last n positions of prefix and m items of suffix as input for the model
        query_array = numpy.zeros((1, self.nr_considered_words_total,len(self.string_to_number)))
        # If prefix is smaller than the numbers that we need to consider, we just use zeros as padding
        len_prefix = len(prefix)
        len_suffix = len(suffix)
        
        # Add the prefix to the array
        max_iterations_prefix = self.nr_considered_words_prefix
        if len(prefix) < self.nr_considered_words_prefix:
            max_iterations_prefix = len_prefix
        
        for i in range(0, max_iterations_prefix):
            i_token_string = self.token_to_string(prefix[len_prefix - i - 1])
            query_array[0, self.nr_considered_words_prefix - i - 1 , ] = self.one_hot(i_token_string)
            
       
        # Add the suffix to the array
        max_iterations_suffix = self.nr_considered_words_suffix
        if len(suffix) < self.nr_considered_words_suffix:
            max_iterations_suffix = len_suffix
        for i in range(self.nr_considered_words_prefix, self.nr_considered_words_prefix + max_iterations_suffix):
            i_token_string = self.token_to_string(suffix[i - self.nr_considered_words_prefix])
            query_array[0, i , ] = self.one_hot(i_token_string)
        
        
        y = self.model.predict(query_array)
        predicted_seq = y[0]
        if type(predicted_seq) is numpy.ndarray:
            predicted_seq = predicted_seq.tolist() 
        best_number = predicted_seq.index(max(predicted_seq))
        best_string = self.number_to_string[best_number]
        best_token = self.string_to_token(best_string)
        print("Best token: {}".format(best_token))
        print(predicted_seq)
        print("----------")
        return [best_token]
    
