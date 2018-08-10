from code_completion_baseline_origin import Code_Completion_Baseline
from runner import load_tokens
import operator


training_dir = "training_data/programs_800_full/"

tokens_list = load_tokens(training_dir)

counting_dict = {}
ccb = Code_Completion_Baseline

for program_tokens in tokens_list:
    for token in program_tokens:
        string_token = ccb.token_to_string(ccb, token)
        if string_token not in counting_dict:
            counting_dict.update({string_token:1})
        else:
            current_counter = counting_dict.get(string_token)
            counting_dict.update({string_token:current_counter+1})

overall_counter = sum(counting_dict.values())
            
sorted_occurance_list = sorted(counting_dict.items(), key=operator.itemgetter(1))
sorted_occurance_list.reverse()


for entry in sorted_occurance_list:
    type_value_concatenated = entry[0]
    occurances = entry[1]
    value_type_dict = ccb.string_to_token(ccb, type_value_concatenated)
    print(value_type_dict)
    value = value_type_dict.get("value")
    type = value_type_dict.get("type")
    percentage = occurances / overall_counter
    #print("Type: {:20} Value: {:10} Occurance: {:8}    Percentage: {:.2f}".format(type, value, occurances, percentage))
    print("{:20}\t{:10}\t{:8}\t{:.2f}".format(type, value, occurances, percentage))

print("There are {} classes".format(len(sorted_occurance_list)))

