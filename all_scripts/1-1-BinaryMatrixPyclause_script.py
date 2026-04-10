import pandas as pd
import json
import ast
import os
import argparse


def create_triplet_mapping(target_triples_file, triplet_mapping_file):
    mapping = {}

    with open(target_triples_file, "r") as file:
        for index, line in enumerate(file, start=1):
            triplet = eval(line.strip())
            mapping[f"T{index}"] = triplet

    with open(triplet_mapping_file, "w") as json_file:
        json.dump(mapping, json_file, indent=4)


def create_rule_mapping(rules_path, rule_mapping_file):
    rule_mapping = {}

    with open(rules_path, "r") as file:
        for index, line in enumerate(file, start=1):
            rule_parts = line.strip().split("\t")
            if len(rule_parts) == 4:
                rule_mapping[f"R{index}"] = rule_parts[3]

    with open(rule_mapping_file, "w") as json_file:
        json.dump(rule_mapping, json_file, indent=4)


def create_rule_csv(csv_path, rules_path, rules_mapping_path):
    with open(rules_mapping_path, 'r') as f:
        rule_map = json.load(f)

    rules_conf = {}
    with open(rules_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                rule_str = parts[3]
                confidence = float(parts[2])
                rules_conf[rule_str] = confidence

    # Build the output list
    data = []
    for rule_id, rule_str in rule_map.items():
        confidence = rules_conf.get(rule_str, None)  # None if rule_str not found
        if confidence is not None:
            data.append((rule_id, confidence))
        else:
            print(f"Warning: Rule string not found in rules file: {rule_id} → {rule_str}")

    # Create and save DataFrame
    df = pd.DataFrame(data, columns=["Rule", "Confidence"])
    df.to_csv(csv_path, index=False)
    print(f"Saved rule-confidence CSV to {csv_path}")


def create_binary_matrix(rule_mapping_file, triplet_mapping_file, pred_rule_grounding_path,
                         output_csv="binary_matrix.csv", rule_cap=None):
    # Load the rule and triplet mappings
    with open(rule_mapping_file, "r") as f:
        rule_mapping = json.load(f)
    with open(triplet_mapping_file, "r") as f:
        triplet_mapping = json.load(f)

    # Reverse mappings for easy lookup
    rule_to_id = {v: k for k, v in rule_mapping.items()}
    triplet_to_id = {tuple(v): k for k, v in triplet_mapping.items()}  # Convert lists to tuples for matching

    all_rule_ids = sorted(rule_mapping.keys(), key=lambda x: int(x[1:]))
    all_triplet_ids = sorted(triplet_mapping.keys(), key=lambda x: int(x[1:]))

    # Apply rule cap
    if rule_cap is not None:
        all_rule_ids = all_rule_ids[:rule_cap]  # Limit rules to cap

    binary_matrix = {rule_id: {triplet_id: "" for triplet_id in all_triplet_ids} for rule_id in all_rule_ids}

    with open(pred_rule_grounding_path, "r") as file:
        lines = file.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:  # Skip empty lines
            i += 1
            continue

        try:
            triplet = ast.literal_eval(line)
        except (SyntaxError, ValueError):
            print(f"Skipping malformed line at {i}: {line}")
            i += 1
            continue

        triplet_key = triplet_to_id.get(tuple(triplet), None)  # Get mapped triples id
        i += 1

        if triplet_key is None or triplet_key not in all_triplet_ids:
            continue

        try:
            rules = ast.literal_eval(lines[i].strip())
        except (SyntaxError, ValueError):
            print(f"Skipping malformed rule set at {i}: {lines[i]}")
            i += 1
            continue

        i += 1
        i += 1

        for rule in rules:
            rule_key = rule_to_id.get(rule, None)  # Get mapped rule id

            if rule_key is not None and rule_key in all_rule_ids:
                binary_matrix[rule_key][triplet_key] = 1  # If rule was fired

    df = pd.DataFrame.from_dict(binary_matrix, orient="index").fillna("")

    df = df.replace("", pd.NA).dropna(how="all").fillna("")

    df.to_csv(output_csv)

    return df


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process dataset and create binary matrix')
    parser.add_argument('--dataset', type=str,
                        choices=['UML', 'WN18RR', 'FB15-237', 'Kinship', 'Family', 'JF17K', 'Nations', 'WN18', 'CODEX-S'],
                        default='Family', help='Dataset name')
    parser.add_argument('--confidence', type=int, default=0,
                        help='Confidence percentage (0-100)')
    parser.add_argument('--rule_cap', type=int, default=None,
                        help='Maximum number of rules to include (default: all)')

    args = parser.parse_args()

    dataset = args.dataset
    confidence_percentage = args.confidence
    confidence_threshold = confidence_percentage / 100
    rule_cap = args.rule_cap

    print(f"Processing dataset: {dataset}")
    print(f"Confidence percentage: {confidence_percentage}% (threshold: {confidence_threshold})")
    if rule_cap:
        print(f"Rule cap: {rule_cap}")

    # Define paths
    target_triples_path = f'../Data-AnyBURL/1-OriginalDataset/{dataset}/target_triples.txt'
    triples_mapping_path = f'../Data-AnyBURL/1-OriginalDataset/{dataset}/triples_mapping.json'
    rules_path = f'../Data-AnyBURL/1-OriginalDataset/{dataset}/{dataset.lower()}_10_{confidence_threshold}_10.txt'
    rules_mapping_path = f'../Data-AnyBURL/1-OriginalDataset/{dataset}/rules_mapping_{confidence_percentage}.json'
    csv_path = f'../Data-AnyBURL/1-OriginalDataset/{dataset}/{dataset.lower()}_10_{confidence_threshold}_10.csv'
    pred_rule_grounding_path = f'../Data-AnyBURL/1-OriginalDataset/{dataset}/triple_pred_rules_grounding_{confidence_percentage}.txt'
    binary_matrix_dir = f'../Data-AnyBURL/3-POST-Data-PC/3-3-2-2-BM_PyClause/'
    os.makedirs(binary_matrix_dir, exist_ok=True)
    binary_matrix_path = f'{binary_matrix_dir}/{dataset}_bm_int_train_{confidence_percentage}.csv'

    # Process data
    print(f"Creating triplet mapping...")
    create_triplet_mapping(target_triples_path, triples_mapping_path)

    print(f"Creating rule mapping...")
    create_rule_mapping(rules_path, rules_mapping_path)

    print(f"Creating rule CSV...")
    create_rule_csv(csv_path, rules_path, rules_mapping_path)

    print(f"Creating binary matrix...")
    matrix_df = create_binary_matrix(rules_mapping_path, triples_mapping_path,
                                     pred_rule_grounding_path, binary_matrix_path, rule_cap)

    print(f"Binary matrix created at: {binary_matrix_path}")
    print(f"Matrix shape: {matrix_df.shape}")


if __name__ == "__main__":
    main()