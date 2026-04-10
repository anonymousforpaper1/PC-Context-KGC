from c_clause import RankingHandler, Loader
from clause import Options
import os
import pandas as pd
import ast
import json
import re



DATASETS = ['Family', 'Nations', 'UML', 'Kinship', 'WN18', 'WN18RR', 'FB15-237', 'CODEX-S']

dataset = DATASETS[0]
k_start = 100
k_end = 2600
k_intervals = 100
confidence_percentage = 0
em_iterations = 100

cond_prob_threshold = 0.0


confidence_threshold = confidence_percentage/100
train_path = f'../Data-AnyBURL/1-OriginalDataset/{dataset}/train.txt'
test_path = f'../Data-AnyBURL/1-OriginalDataset/{dataset}/test.txt'

original_rules_path = f'../Data-AnyBURL/1-OriginalDataset/{dataset}/{dataset.lower()}_10_{confidence_threshold}_10.txt'
rules_path = f'../Data-AnyBURL/1-OriginalDataset/{dataset}/{dataset.lower()}_10_{confidence_threshold}_10_confidence_1.txt'


sorted_rule_files_k_dir = f'../Tests-AnyBURL/{dataset}/sorted_rule_files_k_{confidence_percentage}_{em_iterations}'
os.makedirs(sorted_rule_files_k_dir, exist_ok=True)
ranking_dir = f'../Tests-AnyBURL/{dataset}/rankings_files_{confidence_percentage}_{em_iterations}'
head_prediction_mappings_dir = f'../Tests-AnyBURL/{dataset}/head_preds_files_{confidence_percentage}_{em_iterations}'
tail_prediction_mappings_dir = f'../Tests-AnyBURL/{dataset}/tail_pred_files_{confidence_percentage}_{em_iterations}'
os.makedirs(ranking_dir, exist_ok=True)
os.makedirs(head_prediction_mappings_dir, exist_ok=True)
os.makedirs(tail_prediction_mappings_dir, exist_ok=True)
mapping_triple_rules_dir = f'../Tests-AnyBURL/{dataset}/mappings_triples_rules_k_{confidence_percentage}_{em_iterations}'
os.makedirs(mapping_triple_rules_dir, exist_ok=True)

rule_name_mappings = f'../Data-AnyBURL/1-OriginalDataset/{dataset}/mapped_{dataset.lower()}_10_{confidence_threshold}_10.csv'



conditional_prob_table = f'../Data-AnyBURL/4-Rules/4-2-3-Rules-EVI/NoGreedy/condprob_{dataset}_train_missing_{em_iterations}_{confidence_percentage}_{cond_prob_threshold}.csv'

sorted_rule_file_conf_1 = f'../Data-AnyBURL/1-OriginalDataset/{dataset}/sorted_conditional_prob_{dataset.lower()}_10_{confidence_threshold}_10_confidence_1.txt'

# if os.path.exists(rule_name_mappings):
#     print(f"Skipping mapping: '{rule_name_mappings}' already exists.")
# else:
with open(original_rules_path, 'r') as file:
    lines = file.readlines()

# Create lists for the DataFrame
rule_names = [f'R{i+1}' for i in range(len(lines))]
rules = [line.strip() for line in lines]
rule_texts = [line.strip().split('\t', 3)[-1] for line in lines]  # Extract rule text after third tab

# Create DataFrame
df = pd.DataFrame({
    'rule_name': rule_names,
    'rule': rules,
    'rule_text': rule_texts
})

# Write to CSV file
df.to_csv(rule_name_mappings, index=False)

print(f"DataFrame has been created and saved to {rule_name_mappings}")
print("\nFirst few rows of the DataFrame:")
print(df.head())


def generate_modified_rules(condprob_path, mapped_rule_path, output_path):
    # Read the CSV files
    condprob_df = pd.read_csv(condprob_path)
    mapped_rule_df = pd.read_csv(mapped_rule_path)

    # Create a dictionary for quick rule lookup
    rule_dict = dict(zip(mapped_rule_df['rule_name'], mapped_rule_df['rule_text']))

    # Initialize set to track all seen rules
    all_seen_rules = set()

    # Initialize output file
    with open(output_path, 'w') as f:
        # Iterate through each row in condprob_df
        for i in range(len(condprob_df)):
            # Get current RuleSet
            current_rules = set(ast.literal_eval(condprob_df.iloc[i]['RuleSets']))

            # Find new rule(s) by comparing with all previously seen rules
            new_rules = current_rules - all_seen_rules
            # If there's a new rule, write it to the file
            for new_rule in new_rules:
                if new_rule in rule_dict:
                    rule_text = rule_dict[new_rule]
                    # Write to file with modified prefix
                    f.write(f"1000\t1000\t1.0\t{rule_text}\n")

            # Update all seen rules
            all_seen_rules.update(current_rules)


generate_modified_rules(condprob_path=conditional_prob_table, mapped_rule_path=rule_name_mappings, output_path=sorted_rule_file_conf_1)

import csv

def create_rule_dictionary(input_path_file):
    rule_dict = {}
    with open(input_path_file, mode='r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            rule_dict[row['rule_text']] = row['rule_name']
    return rule_dict


input_file = rule_name_mappings
ruleText_ruleName_dict = create_rule_dictionary(input_file)
# print(ruleText_ruleName_dict)


def generate_head_tail_prediction_mappings(train_path, test_path, rules_path, ranking_txt_path,
                                           head_prediction_mappings, tail_prediction_mappings):
    opts = Options()
    opts.set("qa_handler.aggregation_function", "maxplus")
    opts.set("ranking_handler.collect_rules", "True")

    loader = Loader(opts.get("loader"))
    # set filter to "" if not required
    loader.load_data(data=train_path, target=test_path)
    loader.load_rules(rules=rules_path)
    ranker = RankingHandler(opts.get("ranking_handler"))
    ranker.calculate_ranking(loader=loader)

    head_ranking = ranker.get_ranking(direction="head", as_string=True)
    tail_ranking = ranker.get_ranking(direction="tail", as_string=True)

    ranker.write_ranking(path=ranking_txt_path, loader=loader)

    # obtain rule features for every query
    head_rules = ranker.get_rules(direction="head", as_string=True)
    tail_rules = ranker.get_rules(direction="tail", as_string=True)

    # write rule features
    ranker.write_rules(path=head_prediction_mappings, loader=loader, direction="head", as_string=True)
    ranker.write_rules(path=tail_prediction_mappings, loader=loader, direction="tail", as_string=True)
    # from clause.util.utils import read_jsonl
    # # list of dicts
    # read_jsonl(tail_prediction_mappings)


for k in range(k_start, k_end+k_intervals, k_intervals):
    top_k_sorted_rule_file = f'{sorted_rule_files_k_dir}/rules_top_k_{k}.txt'
    with open(sorted_rule_file_conf_1, 'r') as f:
        all_rules = f.readlines()

    # Select top k rules, ensuring k doesn't exceed available rules
    top_k_rules = all_rules[:min(k, len(all_rules))]

    # Write to new file
    with open(top_k_sorted_rule_file, 'w') as f:
        f.writelines(top_k_rules)

for k in range(k_start, k_end+k_intervals, k_intervals):
    top_k_sorted_rule_file = f'{sorted_rule_files_k_dir}/rules_top_k_{k}.txt'
    ranking_txt_path = f'{ranking_dir}/ranking_{k}.txt'
    head_prediction_mappings = f'{head_prediction_mappings_dir}/head_prediction_mappings_{k}.json'
    tail_prediction_mappings = f'{tail_prediction_mappings_dir}/tail_prediction_mappings_{k}.json'

    print(k)
    generate_head_tail_prediction_mappings(train_path=train_path, test_path=test_path,
                                           rules_path=top_k_sorted_rule_file, ranking_txt_path=ranking_txt_path,
                                           head_prediction_mappings=head_prediction_mappings,
                                           tail_prediction_mappings=tail_prediction_mappings)

for k in range(k_start, k_end+k_intervals, k_intervals):
    top_k_sorted_rule_file = f'{sorted_rule_files_k_dir}/rules_top_k_{k}.txt'
    ranking_txt_path = f'{ranking_dir}/ranking_{k}.txt'
    head_prediction_mappings = f'{head_prediction_mappings_dir}/head_prediction_mappings_{k}.json'
    tail_prediction_mappings = f'{tail_prediction_mappings_dir}/tail_prediction_mappings_{k}.json'
    mappings_triples_rules = f'{mapping_triple_rules_dir}/predictedTriples_rules_{confidence_percentage}_k_{k}.csv'
    dict_triples_rules = dict()

    try:
        # Read the file content
        with open(head_prediction_mappings, 'r') as file:
            content = file.read()

        # Try to load the JSON directly
        try:
            data = json.loads(content)
            if not isinstance(data, list):
                print("Error: JSON root should be a list of dictionaries")
                data = [data]  # Wrap single dict in a list
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")
            print("Attempting to parse multiple JSON objects...")

            # Split content into potential JSON objects (assuming they are separated by }{ or whitespace)
            # This is a basic approach; adjust regex based on your file's structure
            json_objects = re.split(r'}\s*{', content.strip('[]{}'))
            json_objects = [obj.strip() for obj in json_objects if obj.strip()]

            data = []
            for i, obj in enumerate(json_objects):
                try:
                    # Reconstruct valid JSON by adding braces
                    if not obj.startswith('{'):
                        obj = '{' + obj
                    if not obj.endswith('}'):
                        obj = obj + '}'
                    parsed = json.loads(obj)
                    if isinstance(parsed, dict):
                        data.append(parsed)
                except json.JSONDecodeError as e:
                    print(f"Failed to parse object {i + 1}: {e}")
                    continue

            if not data:
                print("Error: No valid JSON objects found")
                exit(1)

        # Iterate through each dictionary
        for index, item in enumerate(data, 1):
            # print(f"\nDictionary {index}:")
            # print("-" * 50)

            # Print query
            query = item.get('query', 'Not found')
            tail = query[0]
            rel = query[1]
            # print("Query:", item.get('query', 'Not found'))

            # Print answers
            answers = item.get('answers', 'Not found')
            # print("Answers:", item.get('answers', 'Not found'))
            #
            # # Print rules
            # print("Rules:")

            rules = item.get('rules', [])
            # for rule_set in rules:
            #     print("  -", rule_set)
            #
            # print("-" * 50)

            for answer_number in range(len(answers)):
                for rule in rules[answer_number]:
                    predicted_triple = (answers[answer_number], rel, tail)
                    if predicted_triple not in dict_triples_rules:
                        dict_triples_rules[predicted_triple] = [ruleText_ruleName_dict[rule]]
                    else:
                        dict_triples_rules[predicted_triple].append(ruleText_ruleName_dict[rule])


    except FileNotFoundError:
        print(f"Error: File '{head_prediction_mappings}' not found")
    except Exception as e:
        print(f"Error: {str(e)}")

    try:
        # Read the file content
        with open(tail_prediction_mappings, 'r') as file:
            content = file.read()

        # Try to load the JSON directly
        try:
            data = json.loads(content)
            if not isinstance(data, list):
                print("Error: JSON root should be a list of dictionaries")
                data = [data]  # Wrap single dict in a list
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")
            print("Attempting to parse multiple JSON objects...")

            # Split content into potential JSON objects (assuming they are separated by }{ or whitespace)
            # This is a basic approach; adjust regex based on your file's structure
            json_objects = re.split(r'}\s*{', content.strip('[]{}'))
            json_objects = [obj.strip() for obj in json_objects if obj.strip()]

            data = []
            for i, obj in enumerate(json_objects):
                try:
                    # Reconstruct valid JSON by adding braces
                    if not obj.startswith('{'):
                        obj = '{' + obj
                    if not obj.endswith('}'):
                        obj = obj + '}'
                    parsed = json.loads(obj)
                    if isinstance(parsed, dict):
                        data.append(parsed)
                except json.JSONDecodeError as e:
                    print(f"Failed to parse object {i + 1}: {e}")
                    continue

            if not data:
                print("Error: No valid JSON objects found")
                exit(1)

        # Iterate through each dictionary
        for index, item in enumerate(data, 1):
            # print(f"\nDictionary {index}:")
            # print("-" * 50)

            # Print query
            query = item.get('query', 'Not found')
            head = query[0]
            rel = query[1]
            # print("Query:", item.get('query', 'Not found'))

            # Print answers
            answers = item.get('answers', 'Not found')
            # print("Answers:", item.get('answers', 'Not found'))
            #
            # # Print rules
            # print("Rules:")

            rules = item.get('rules', [])
            # for rule_set in rules:
            #     print("  -", rule_set)
            #
            # print("-" * 50)

            for answer_number in range(len(answers)):
                for rule in rules[answer_number]:
                    predicted_triple = (head, rel, answers[answer_number])
                    if predicted_triple not in dict_triples_rules:
                        dict_triples_rules[predicted_triple] = [ruleText_ruleName_dict[rule]]
                    else:
                        dict_triples_rules[predicted_triple].append(ruleText_ruleName_dict[rule])


    except FileNotFoundError:
        print(f"Error: File '{tail_prediction_mappings}' not found")
    except Exception as e:
        print(f"Error: {str(e)}")

    for key, value in dict_triples_rules.items():
        dict_triples_rules[key] = list(set(value))

    with open(mappings_triples_rules, 'w', newline='') as csvfile:
        # Define CSV writer and headers
        writer = csv.writer(csvfile)
        writer.writerow(['predicted_triples', 'rules_fired'])

        # Write each key-value pair
        for key, value in dict_triples_rules.items():
            key_str = str(key)
            value_str = str(value)
            writer.writerow([key_str, value_str])



