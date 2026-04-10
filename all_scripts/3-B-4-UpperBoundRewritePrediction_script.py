import pandas as pd
import os
import ast

DATASETS = ['Family', 'Nations', 'UML', 'Kinship', 'WN18', 'WN18RR', 'FB15-237', 'CODEX-S']

dataset = DATASETS[0]
k_start = 100
k_end = 2600
k_intervals = 100
confidence_percentage = 0
em_iterations = 100

cutoff_prob_threshold = 0.0

confidence_threshold = confidence_percentage/100
filtered_prediction_PC_dir = f'../Tests-AnyBURL/k_filtered_prediction_PC/{dataset}/upper_bound_{confidence_percentage}_{em_iterations}'
mapping_triple_rules_dir = f'../Tests-AnyBURL/{dataset}/mappings_triples_rules_k_{confidence_percentage}_{em_iterations}'
#filtered_prediction_PC_dir = f'../Tests-AnyBURL/k_filtered_prediction_PC/{dataset}/upper_bound'
#mapping_triple_rules_dir = f'../Tests-AnyBURL/{dataset}/mappings_triples_rules_k'


rewritten_prediction_PC_dir = f'../Tests-AnyBURL/k_filtered_prediction_PC/{dataset}/upper_bound_rewritten_{confidence_percentage}_{em_iterations}'
os.makedirs(rewritten_prediction_PC_dir, exist_ok=True)

def parse_prediction_file_as_df(filepath):
    rows = []
    with open(filepath, "r") as f:
        lines = f.read().splitlines()

    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue

        test_case = lines[i].strip()
        i += 1

        heads, tails = [], []

        if i < len(lines) and lines[i].startswith("Heads:"):
            parts = lines[i][6:].strip().split("\t")
            heads = [(parts[j], float(parts[j + 1])) for j in range(0, len(parts) - 1, 2)]
            i += 1

        if i < len(lines) and lines[i].startswith("Tails:"):
            parts = lines[i][6:].strip().split("\t")
            tails = [(parts[j], float(parts[j + 1])) for j in range(0, len(parts) - 1, 2)]
            i += 1

        rows.append({
            "test_case": test_case,
            "head_predictions": str(heads),
            "tail_predictions": str(tails),
        })

    return pd.DataFrame(rows)


def rewrite_prediction_scores(original_txt_path, mapping_csv_path, output_path):
    parsed_df = parse_prediction_file_as_df(original_txt_path)

    mapping_df = pd.read_csv(mapping_csv_path)
    triplet_score_map = {
        triplet.strip(): float(prob)
        for triplet, prob in zip(mapping_df["predicted_triples"], mapping_df["ProbabilityFromNegation"])
    }

    rewritten_lines = []

    for _, row in parsed_df.iterrows():
        test_triplet = row["test_case"]
        head_preds = eval(row["head_predictions"])
        tail_preds = eval(row["tail_predictions"])

        rewritten_lines.append(test_triplet)

        # Heads rewrite
        line = ["Heads: "]
        if head_preds:
            scored_heads = []
            for h in head_preds:
                triple = f"('{h[0]}', '{test_triplet.split()[1]}', '{test_triplet.split()[2]}')"
                score = triplet_score_map.get(triple, 1.0)
                scored_heads.append((h[0], score))
            scored_heads.sort(key=lambda x: -x[1])  # Sort descending by score
            line.extend([f"{h}\t{score:.6f}\t" for h, score in scored_heads])
        rewritten_lines.append("".join(line))

        # Tails rewrite
        line = ["Tails: "]
        if tail_preds:
            scored_tails = []
            for t in tail_preds:
                triple = f"('{test_triplet.split()[0]}', '{test_triplet.split()[1]}', '{t[0]}')"
                score = triplet_score_map.get(triple, 1.0)
                scored_tails.append((t[0], score))
            scored_tails.sort(key=lambda x: -x[1])  # Sort descending by score
            line.extend([f"{t}\t{score:.6f}\t" for t, score in scored_tails])
        rewritten_lines.append("".join(line))

    with open(original_txt_path, "r") as f:
        original_lines = [line.strip() for line in f if line.strip() and not line.startswith("Heads:") and not line.startswith("Tails:")]

    rewritten_test_cases = [line for line in rewritten_lines if not line.startswith("Heads:") and not line.startswith("Tails:")]

    if len(original_lines) != len(rewritten_test_cases):
        print(f"⚠️ Sanity Check Failed: # original test cases = {len(original_lines)}, # rewritten = {len(rewritten_test_cases)}")
    else:
        print("✅ Sanity Check Passed: Test case count matches.")

    with open(output_path, "w") as f:
        f.write("\n".join(rewritten_lines))

    print(f"✅ Rewritten predictions saved to: {output_path}")


for k in range(k_start, k_end + k_intervals, k_intervals):
    prediction_triples_csv_file = f'{mapping_triple_rules_dir}/predictedTriples_rules_{confidence_percentage}_k_{k}_with_marginals.csv'
    filtered_prediction_PC_path = f'{filtered_prediction_PC_dir}/predictions_{dataset}_train_missing_{em_iterations}_{confidence_percentage}_{cutoff_prob_threshold}_k_{k}.txt'
    rewritten_prediction_path = f'{rewritten_prediction_PC_dir}/ub_predictions_{dataset}_train_missing_{em_iterations}_{confidence_percentage}_{cutoff_prob_threshold}_k_{k}.txt'
    rewrite_prediction_scores(original_txt_path = filtered_prediction_PC_path, mapping_csv_path= prediction_triples_csv_file, output_path = rewritten_prediction_path)
