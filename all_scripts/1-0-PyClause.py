from c_clause import PredictionHandler, Loader
from clause import Options
import os

DATASETS = ['Family', 'Nations', 'UML', 'Kinship', 'WN18', 'WN18RR', 'FB15-237', 'CODEX-S']
dataset = DATASETS[0]

confidence_percentage = 0
confidence_threshold = confidence_percentage/100

train_path = f'../Data-AnyBURL/1-OriginalDataset/{dataset}/train.txt'
test_path = f'../Data-AnyBURL/1-OriginalDataset/{dataset}/train.txt'

rules_path = f'../Data-AnyBURL/1-OriginalDataset/{dataset}/{dataset.lower()}_10_{confidence_threshold}_10_confidence_1.txt'

opts = Options()
# opts.set("qa_handler.aggregation_function", "maxplus")

loader = Loader(opts.get("loader"))
# set filter to "" if not required
loader.load_data(data=train_path)
loader.load_rules(rules=rules_path)

opts.set("prediction_handler.collect_explanations", True)
opts.set("prediction_handler.num_top_rules", -1)
scorer = PredictionHandler(options=opts.get("prediction_handler"))

scorer.calculate_scores(triples=test_path, loader=loader)

targets, pred_rules, groundingsss = scorer.get_explanations(as_string=True)

# print(targets[52], pred_rules[52], groundings[52])
# Function to flatten nested lists
target_file_path = f'../Data-AnyBURL/1-OriginalDataset/{dataset}/target_triples.txt'
skip_target_file = os.path.exists(target_file_path)

# Open the other files in append mode
with open(f'../Data-AnyBURL/1-OriginalDataset/{dataset}/pred_rules_{confidence_percentage}.txt', "a") as pred_rules_file, \
     open(f'../Data-AnyBURL/1-OriginalDataset/{dataset}/triple_pred_rules_grounding_{confidence_percentage}.txt', "a") as all_file, \
     (open(target_file_path, "a") if not skip_target_file else open(os.devnull, "w")) as targets_file:  # Skip if exists

    count = 0
    for target, pred_rule in zip(targets, pred_rules):
        # Write to respective files
        if not skip_target_file:
            targets_file.write(str(target) + "\n")

        pred_rules_file.write(str(pred_rule) + "\n")
        all_file.write(str(target) + "\n")
        all_file.write(str(pred_rule) + "\n")

        # Print to console
        # print(target, pred_rule, grounding)

        # count+=1
        # if count >2:
        #     break

if skip_target_file:
    print(f"Skipped writing to {target_file_path} because it already exists.")
