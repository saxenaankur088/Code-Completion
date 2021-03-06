import tflearn
import numpy
from keras.models import Sequential, Model
from keras.layers import *
from keras.callbacks import EarlyStopping
from keras.models import load_model
import seq2seq
import matplotlib.pyplot as plt
from seq2seq.models import SimpleSeq2Seq
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from random import randint


class Code_Completion_Baseline:
    
    batch_size = 256
    layer_dim = 128
    epochs = 100
    dropout = 0.3
    nr_considered_words_prefix = 5
    nr_considered_words_suffix = 5
    nr_considered_words_total = nr_considered_words_prefix + nr_considered_words_suffix
    verbose = 2
    max_hole_size = 3
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
        
    def create_hole(self, tokens):
        hole_size = randint(1, self.max_hole_size)
        hole_start_idx = randint(1, len(tokens) - hole_size)
        prefix = tokens[0:hole_start_idx]
        expected = tokens[hole_start_idx:hole_start_idx + hole_size]
        suffix = tokens[hole_start_idx + hole_size:]
        return(prefix, expected, suffix)
    
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
        self.y_dim = self.max_hole_size * len(self.number_to_string)
        for token_list in token_lists:
            # create 50 random samples from every file
            range_end = 400 # 50 len(token_list)
            for i in range(0,range_end):
                prefix, expected, suffix = self.create_hole(token_list)

                x = numpy.zeros(self.nr_considered_words_total)
                
                # Values for prefix
                lower_prefix_bound = self.nr_considered_words_prefix
                if len(prefix) < self.nr_considered_words_prefix:
                    lower_prefix_bound = len(prefix)
                
                row_idx = 0             
                for i in range(-1, -1-lower_prefix_bound , -1):
                    i_token_string = self.token_to_string(token_list[i])
                    x [self.nr_considered_words_prefix - 1 - row_idx] = self.string_to_number.get(i_token_string)
                    row_idx += 1
                
                #Values for suffix
                row_idx = self.nr_considered_words_prefix
                lower_suffix_bound = self.nr_considered_words_suffix
                if len(suffix) < self.nr_considered_words_suffix:
                    lower_suffix_bound = len(suffix)
                for i in range(0, lower_suffix_bound):
                    i_token_string = self.token_to_string(token_list[i])
                    x [row_idx] = self.string_to_number.get(i_token_string)
                    row_idx += 1
                xs.append(x)
                #y values
                y = numpy.zeros(self.y_dim)
                
                for i, token in enumerate(expected):
                    type_value_pair = self.token_to_string(token)
                    one_position = self.string_to_number.get(type_value_pair)
                    y[i * len(self.number_to_string) + one_position] = 1
                ys.append(y)
                samples +=1

        final_xs = numpy.zeros((samples, self.nr_considered_words_total))
        final_ys = numpy.zeros((samples, self.max_hole_size, len(self.string_to_number)))
        for i, entry in enumerate(xs):
            final_xs[i,:] = entry
        for i, entry in enumerate(ys):
            final_ys[i,:,:] = numpy.reshape(entry, (self.max_hole_size, len(self.string_to_number)))

          
        print("x,y pairs: " + str(len(final_xs)))     
        print("x Shape: {}".format(numpy.shape(final_xs)))
        print("y Shape: {}".format(numpy.shape(final_ys)))   
        return (final_xs, final_ys)

    def create_network(self, load = False, model_file = None):
        self.model = Sequential()
        vocab_size = len(self.string_to_number) 
        
        self.model.add(Embedding(input_dim = vocab_size, input_length=self.nr_considered_words_total, output_dim = 16, mask_zero = True))
        # One-direcitonal LSTM
        self.model.add(LSTM(64, activation='tanh', return_sequences = False))
        #self.model.add(Dropout(self.dropout))
        self.model.add(Dense(vocab_size * self.max_hole_size, activation='sigmoid'))
        #self.model.add(Dropout(self.dropout))
        self.model.add(Reshape((self.max_hole_size, vocab_size)))
        self.model.add(Dense(vocab_size, activation="softmax"))
        
        if load == True:
            self.model.load_weights(model_file)
        # Bi-directional LSTM
        #self.model.add(Bidirectional(LSTM(128, activation='relu'), input_shape=(self.nr_considered_words_total, len(self.string_to_number),)))
        #self.model.add(Dropout(self.dropout))
        #model = SimpleSeq2Seq(input_dim=5, hidden_dim=10, output_length=8, output_dim=8)
        adam = optimizers.Adam(lr=0.01, decay=1e-6, clipnorm=1., clipvalue=0.5)
        self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        self.model.summary()
        
    
    def load(self, token_lists, model_file):
        self.create_token_dicts(token_lists)
        self.create_network(True, model_file)
    
    def train(self, token_lists,test_token_list, model_file):

          
        self.create_token_dicts(numpy.append(token_lists, test_token_list))
        (xs, ys) = self.prepare_data(token_lists)
        print(ys[0])
        print(ys[1])
        print(ys[2])
        print(ys[3])
        (xs_test, ys_test) = self.prepare_data(test_token_list)
        
        self.create_network()
        early_stopping = EarlyStopping(monitor='val_acc', patience=3)

        checkpoint = ModelCheckpoint(model_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        
        #early_stopping = EarlyStopping(monitor='val_acc', patience=3)
        history = self.model.fit(xs, ys, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose, validation_data=(xs_test, ys_test),
                        callbacks=[checkpoint
                            #early_stopping
                            ])

        self.plot_acc(history)
        
        
    def plot_acc(self, keras_history):
        plt.plot(keras_history.history['acc'])
        plt.plot(keras_history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        
    def query(self, prefix, suffix):
        # take last n positions of prefix and m items of suffix as input for the model
        query_array = numpy.zeros((1, self.nr_considered_words_total))
        # If prefix is smaller than the numbers that we need to consider, we just use zeros as padding
        len_prefix = len(prefix)
        len_suffix = len(suffix)
        
        # Add the prefix to the array
        max_iterations_prefix = self.nr_considered_words_prefix
        if len(prefix) < self.nr_considered_words_prefix:
            max_iterations_prefix = len_prefix
        
        for i in range(0, max_iterations_prefix):
            i_token_string = self.token_to_string(prefix[len_prefix - i - 1])
            query_array[0, self.nr_considered_words_prefix - i - 1] = self.string_to_number.get(i_token_string)
            
       
        # Add the suffix to the array
        max_iterations_suffix = self.nr_considered_words_suffix
        if len(suffix) < self.nr_considered_words_suffix:
            max_iterations_suffix = len_suffix
        for i in range(self.nr_considered_words_prefix, self.nr_considered_words_prefix + max_iterations_suffix):
            i_token_string = self.token_to_string(suffix[i - self.nr_considered_words_prefix])
            query_array[0, i] = self.string_to_number.get(i_token_string)
        
        
        y = self.model.predict(query_array)
        predicted_seq = y[0]
        if type(predicted_seq) is numpy.ndarray:
            predicted_seq = predicted_seq.tolist() 
        best_number = predicted_seq.index(max(predicted_seq))
        best_string = self.number_to_string[best_number]
        best_token = self.string_to_token(best_string)
        return [best_token]
    
