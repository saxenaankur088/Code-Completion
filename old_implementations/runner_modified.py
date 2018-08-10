from os import listdir
from os.path import join, isfile
import json
from random import randint

#########################################
## START of part that students may change
from code_completion_baseline_lstm_prefix_suffix_embed_var_hole import Code_Completion_Baseline
#from utils import string_token

training_dir = "training_data/programs_800_full/"
query_dir = "training_data/programs_200_full/"

"""
training_dir = "training_data/programs_800_middle/"
query_dir = "training_data/programs_200_middle/"

training_dir = "training_data/programs_800/"
query_dir = "training_data/programs_200/"
"""
model_file = "C:/Users/alexa/dl_software/model/var_hole.hdf5"
use_stored_model = False

max_hole_size = 1
simplify_tokens = True
## END of part that students may change
#########################################

def simplify_token(token):
    if token["type"] == "Identifier":
        token["value"] = "ID"
    elif token["type"] == "String":
        token["value"] = "\"STR\""
    elif token["type"] == "RegularExpression":
        token["value"] = "/REGEXP/"
    elif token["type"] == "Numeric":
        token["value"] = "5"

# load sequences of tokens from files
def load_tokens(token_dir):
    token_files = [join(token_dir, f) for f in listdir(token_dir) if isfile(join(token_dir, f)) and f.endswith("_tokens.json")]
    token_lists = [json.load(open(f, encoding='utf8')) for f in token_files]
    if simplify_tokens:
        for token_list in token_lists:
            for token in token_list:
                simplify_token(token)
    return token_lists

# removes up to max_hole_size tokens
def create_hole(tokens):
    hole_size = randint(1, max_hole_size)
    hole_start_idx = randint(1, len(tokens) - hole_size)
    prefix = tokens[0:hole_start_idx]
    expected = tokens[hole_start_idx:hole_start_idx + hole_size]
    suffix = tokens[hole_start_idx + hole_size:]
    return(prefix, expected, suffix)

# checks if two sequences of tokens are identical
def same_tokens(tokens1, tokens2):
    if len(tokens1) != len(tokens2):
        return False
    for idx, t1 in enumerate(tokens1):
        t2 = tokens2[idx]
        if t1["type"] != t2["type"] or t1["value"] != t2["value"]:
            return False  
    return True

#########################################
## START of part that students may change
code_completion = Code_Completion_Baseline()
## END of part that students may change
#########################################

# train the network
training_token_lists = load_tokens(training_dir)
test_token_list =  load_tokens(query_dir)
if use_stored_model:
    code_completion.load(training_token_lists, model_file)
else:
    code_completion.train(training_token_lists, test_token_list, model_file)
    #code_completion.train(training_token_lists, model_file)

# query the network and measure its accuracy
query_token_lists = load_tokens(query_dir)
correct = incorrect = 0
for tokens in query_token_lists:
    
    (prefix, expected, suffix) = create_hole(tokens)
    #print("Expected token: {}".format(expected))
    string_token_expected = code_completion.token_to_string(expected[0])
    #print(code_completion.one_hot(string_token_expected))
    completion = code_completion.query(prefix, suffix)
    if same_tokens(completion, expected):
        correct += 1
    else:
        incorrect += 1
accuracy = correct / (correct + incorrect)
print("Accuracy: " + str(correct) + " correct vs. " + str(incorrect) + " incorrect = "  + str(accuracy))
