import pandas as pd
import ast
import os
import subprocess
import decimal
import math
from collections import defaultdict

DATASETS = ['Family', 'Nations', 'UML', 'Kinship', 'WN18', 'WN18RR', 'FB15-237', 'CODEX-S']
k_rules_max = 2600
k_rules_max_extra = 2600
k_interval_step = 100
PARAMS = {
    'dataset': DATASETS[0],
    'confidence_percentage': 0,
    'em_iteration': 100,
    'pc_type': 'missing',
    'cutoff_prob_threshold': 0.0
}

sorted_cond_prob_table_dir = "../Data-AnyBURL/4-Rules/4-2-3-Rules-EVI/NoGreedy/"
original_ruleset_dir = "../Data-AnyBURL/1-OriginalDataset/"

k_filtered_ruleset_AB_dir = f'../Tests-AnyBURL/k_filtered_rulesets_AB/{PARAMS["dataset"]}/'
os.makedirs(k_filtered_ruleset_AB_dir, exist_ok=True)

k_filtered_prediction_AB_dir = f'../Tests-AnyBURL/k_filtered_prediction_AB/{PARAMS["dataset"]}/'
os.makedirs(k_filtered_prediction_AB_dir, exist_ok=True)

k_filtered_ruleset_PC_dir = f'../Tests-AnyBURL/k_filtered_rulesets_PC/{PARAMS["dataset"]}/'
os.makedirs(k_filtered_ruleset_PC_dir, exist_ok=True)

k_filtered_prediction_PC_dir = f'../Tests-AnyBURL/k_filtered_prediction_PC/{PARAMS["dataset"]}/unmerged/'
os.makedirs(k_filtered_prediction_PC_dir, exist_ok=True)

merged_k_filtered_prediction_PC_dir = f'../Tests-AnyBURL/k_filtered_prediction_PC/{PARAMS["dataset"]}/lower_bound/'
os.makedirs(merged_k_filtered_prediction_PC_dir, exist_ok=True)

DIRECTORIES = {
    'sorted_cond_prob_table': sorted_cond_prob_table_dir,
    'original_ruleset': original_ruleset_dir,
    'k_filtered_ruleset_AB': k_filtered_ruleset_AB_dir,
    'k_filtered_prediction_AB': k_filtered_prediction_AB_dir,
    'k_filtered_ruleset_PC': k_filtered_ruleset_PC_dir,
    'k_filtered_prediction_PC': k_filtered_prediction_PC_dir,
    'merged_k_filtered_prediction_PC': merged_k_filtered_prediction_PC_dir
}

def run_anyburl_apply(dataset, files, directory_type='AB'):

    ruleset_dir = f"k_filtered_ruleset_{directory_type}"
    prediction_dir = f"k_filtered_prediction_{directory_type}"

    # Resolve absolute paths
    input_abs = os.path.abspath(f"{DIRECTORIES[ruleset_dir]}/{files['rules']}")
    output_abs = os.path.abspath(f"{DIRECTORIES[prediction_dir]}/{files['predictions']}")
    train_abs = os.path.abspath(f"../Data-AnyBURL/1-OriginalDataset/{dataset}/train.txt")
    test_abs = os.path.abspath(f"../Data-AnyBURL/1-OriginalDataset/{dataset}/test.txt")

    # if os.path.exists(output_abs):
    #     print(f"Skipping prediction — already exists: {output_abs}")
    #     return

    # Define Java working directory
    java_cwd = os.path.abspath("..")

    # Fix formatting
    input_rel = os.path.relpath(input_abs, start=java_cwd).replace("\\", "/")
    output_rel = os.path.relpath(output_abs, start=java_cwd).replace("\\", "/")
    train_rel = os.path.relpath(train_abs, start=java_cwd).replace("\\", "/")
    test_rel = os.path.relpath(test_abs, start=java_cwd).replace("\\", "/")

    print("Resolved input:", input_rel)
    print("Resolved output:", output_rel)

    # Build the Java .properties config
    template = f"""\
    PATH_TRAINING={train_rel}
    PATH_TEST={test_rel}

    PATH_RULES={input_rel}
    PATH_OUTPUT={output_rel}

    WORKER_THREADS=7
    TOP_K_OUTPUT=100
    UNSEEN_NEGATIVE_EXAMPLES=0
    """

    properties_template = "\n".join([line.strip() for line in template.strip().splitlines()])

    config_path = os.path.abspath(f"../config-apply-{PARAMS['dataset']}.generated.properties")
    with open(config_path, "w") as f:
        f.write(properties_template)

    print(f"Wrote config to {config_path}")

    # Build and run command from the parent directory
    command = [
        "java",
        "-Xmx12G",
        "-cp", "AnyBURL-23-1.jar",
        "de.unima.ki.anyburl.Apply",
        config_path
    ]

    print("Running AnyBURL Apply...\n")

    try:
        result = subprocess.run(command, cwd="..", capture_output=True, text=True, check=True)
        print("=== STDOUT ===")
        print(result.stdout)
        print("=== STDERR ===")
        print(result.stderr)
    except subprocess.CalledProcessError as e:
        print("❌ Java process failed:")
        print("STDOUT:\n", e.stdout)
        print("STDERR:\n", e.stderr)


# Helper function to created mapped ruleset
def create_mapped_ruleset(original_ruleset_loc, mapped_ruleset_loc):
    with open(original_ruleset_loc, 'r') as file:
        lines = file.readlines()

    # Create lists for the DataFrame
    rule_names = [f'R{i+1}' for i in range(len(lines))]
    rules = [line.strip() for line in lines]

    # Create DataFrame
    df = pd.DataFrame({
        'rule_name': rule_names,
        'rule': rules
    })

    # Write to CSV file
    df.to_csv(mapped_ruleset_loc, index=False)

    print(f"DataFrame has been created and saved to {mapped_ruleset_loc}")
    # print("\nFirst few rows of the DataFrame:")
    print(df.head())



# Generates re-written rule files from the k-rows selected
def process_mapped_rule_files(mapped_ruleset_loc, sorted_cond_prob_table_loc, k_filtered_ruleset_PC_loc_base, k, step_count=1):
    # Read the mapped rules and sorted conditional probability tables
    mapped_rule_df = pd.read_csv(mapped_ruleset_loc)
    sorted_cond_prob_df = pd.read_csv(sorted_cond_prob_table_loc)

    # Lookup dictionary: rule name -> full rule text
    rule_dict = dict(zip(mapped_rule_df['rule_name'], mapped_rule_df['rule']))

    # Only use top-k rows
    sorted_cond_prob_df = sorted_cond_prob_df[:k]

    # Range of steps: e.g., 10, 20, ..., 1000 (if step_count=10 and k=1000)
    for end_row in range(step_count, k + 1, step_count):
        written_rules = set()
        rows_subset = sorted_cond_prob_df[:end_row]

        for _, row in rows_subset.iterrows():
            try:
                rule_list = ast.literal_eval(row['RuleSets'])
            except Exception as e:
                print(f"Skipping malformed RuleSets in row: {row['RuleSets']} ({e})")
                continue

            for rule_name in rule_list:
                if rule_name not in rule_dict:
                    print(f"Rule {rule_name} not found in mapped rules.")
                    continue

                original_rule = rule_dict[rule_name]
                rule_parts = original_rule.strip().split('\t')

                if len(rule_parts) < 4:
                    print(f"Skipping malformed rule: {original_rule}")
                    continue

                rule_parts[0] = str(1000)  # support
                rule_parts[1] = str(1000)  # support_head
                rule_parts[2] = str(1.0)   # confidence

                modified_rule = '\t'.join(rule_parts)
                written_rules.add(modified_rule)

        output_file_path = f"{k_filtered_ruleset_PC_loc_base}k_{end_row}.txt"

        with open(output_file_path, 'w') as output_file:
            for rule in sorted(written_rules):  # sort for consistency
                output_file.write(rule + '\n')

        print(f"Wrote {len(written_rules)} rules to {output_file_path}")

# Helper function to parse AnyBURL style rule files
def parse_rule_line(line):
    parts = line.strip().split("\t")
    if len(parts) < 3:
        return None, 0.0  # skip malformed
    confidence = float(parts[2])
    return line.strip(), confidence


# Just a helper function like in the previous merge script
def parse_prediction_file(filepath):
    parsed = []

    def parse_predictions(line, label):
        line = line.replace(label, "").strip()
        tokens = line.split('\t')
        preds = []
        for i in range(0, len(tokens) - 1, 2):
            try:
                entity = tokens[i]
                score = float(tokens[i + 1])
                preds.append((entity, score))
            except (IndexError, ValueError):
                print(f"⚠️  Skipping malformed prediction pair: {tokens[i:i+2]} in line: {line}")
                continue
        return preds

    with open(filepath, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        test_case = line
        i += 1

        heads, tails = [], []
        if i < len(lines) and lines[i].startswith("Heads:"):
            heads = parse_predictions(lines[i], "Heads:")
            i += 1

        if i < len(lines) and lines[i].startswith("Tails:"):
            tails = parse_predictions(lines[i], "Tails:")
            i += 1

        parsed.append((test_case, heads, tails))

    return parsed



def run_prediction(k, k_step):
    ###################################
    # Setting up DIRECTORIES
    ###################################s
    sorted_cond_prob_table_dir = DIRECTORIES['sorted_cond_prob_table']
    original_ruleset_dir = DIRECTORIES['original_ruleset']
    k_filtered_ruleset_AB_dir = DIRECTORIES['k_filtered_ruleset_AB']
    k_filtered_prediction_AB_dir = DIRECTORIES['k_filtered_prediction_AB']
    k_filtered_ruleset_PC_dir = DIRECTORIES['k_filtered_ruleset_PC']
    k_filtered_prediction_PC_dir = DIRECTORIES['k_filtered_prediction_PC']

    ###################################
    # Get Sorted Conditional Probability Table
    ###################################
    sorted_cond_prob_table_loc = (
        f"{sorted_cond_prob_table_dir}/condprob_"
        f"{PARAMS['dataset']}_train_"
        f"{PARAMS['pc_type']}_"
        f"{PARAMS['em_iteration']}_"
        f"{PARAMS['confidence_percentage']}_"
        f"{PARAMS['cutoff_prob_threshold']}.csv"
    )

    print(f'\nReading Sorted Conditional Probability Table from: {sorted_cond_prob_table_loc}...')
    df = pd.read_csv(sorted_cond_prob_table_loc)

    ###################################
    # Extract top-k rows and collect unique rules
    ###################################
    top_k_rows = df.head(k)
    unique_rules = set()
    for rule_list_str in top_k_rows["RuleSets"]:
        rule_list = ast.literal_eval(rule_list_str)
        unique_rules.update(rule_list)

    n = len(unique_rules)
    print(f"Top-{k} rows use {n} unique rules: {sorted(unique_rules)}")

    ###################################
    # Get the Original Full Ruleset
    ###################################
    original_ruleset_loc = (
        f"{original_ruleset_dir}/"
        f"{PARAMS['dataset']}/"
        f"{PARAMS['dataset'].lower()}_10_{PARAMS['confidence_percentage']/100}_10.txt"
    )

    print(f'\nReading Original Full Ruleset from: {original_ruleset_loc}...')
    with open(original_ruleset_loc, "r") as f:
        rules = f.readlines()

    ###################################
    # Create the Mapped Full Ruleset
    ###################################
    mapped_ruleset_loc = (
        f"{original_ruleset_dir}/"
        f"{PARAMS['dataset']}/"
        f"mapped_{PARAMS['dataset'].lower()}_10_{PARAMS['confidence_percentage']/100}_10.csv"
    )

    print(f'Creating Mapped Full Ruleset from: {mapped_ruleset_loc}...')
    create_mapped_ruleset(original_ruleset_loc, mapped_ruleset_loc)

    ###################################
    # Created the Rewritten k-filtered rulesets
    ###################################
    k_filtered_ruleset_PC_file = (
        f"filtered_ruleset_{PARAMS['dataset']}_train_"
        f"{PARAMS['pc_type']}_"
        f"{PARAMS['em_iteration']}_"
        f"{PARAMS['confidence_percentage']}_"
        f"kmax_{k}_"
    )

    k_filtered_ruleset_PC_loc = (
        f"{k_filtered_ruleset_PC_dir}"
        f"{k_filtered_ruleset_PC_file}"
    )

    process_mapped_rule_files(mapped_ruleset_loc, sorted_cond_prob_table_loc, k_filtered_ruleset_PC_loc, k, k_step)

    # ###################################
    # # Sort full ruleset by confidence and keep top-n
    # ###################################
    # parsed_rules = [parse_rule_line(line) for line in rules]
    # parsed_rules = [r for r in parsed_rules if r[0] is not None]
    # sorted_rules = sorted(parsed_rules, key=lambda x: x[1], reverse=True)
    #
    # top_n_rules = sorted_rules[:n]
    #
    # ###################################
    # # Save to filtered ruleset file
    # ###################################
    # k_filtered_ruleset_AB_file = f"{PARAMS['dataset'].lower()}_10_{PARAMS['confidence_percentage']/100}_10_k_{k}.txt"
    #
    # k_filtered_ruleset_AB_loc = (
    #     f"{k_filtered_ruleset_AB_dir}"
    #     f"{k_filtered_ruleset_AB_file}"
    # )
    #
    # with open(k_filtered_ruleset_AB_loc, "w") as f:
    #     for line, _ in top_n_rules:
    #         f.write(line + "\n")
    #
    # print(f'\nCreated k-filtered Ruleset with top-{n} rules at: {k_filtered_ruleset_AB_loc}')
    #
    # ###################################
    # # Created baseline predictions on filtered ruleset files
    # ###################################
    # k_filtered_prediction_AB_file = (
    #     f"baseline_predictions_{PARAMS['dataset']}_train_"
    #     f"{PARAMS['pc_type']}_"
    #     f"{PARAMS['em_iteration']}_"
    #     f"{PARAMS['confidence_percentage']}_"
    #     f"k_{k}.txt"
    # )
    #
    # k_filtered_prediction_AB_loc = (
    #     f"{k_filtered_prediction_AB_dir}"
    #     f"{k_filtered_prediction_AB_file}"
    # )
    #
    # print(f'\nWriting predictions to: {k_filtered_prediction_AB_loc}')
    #
    # ###################################
    # # Created baseline predictions on filtered ruleset files
    # ###################################
    # files_AB = {
    #     'rules': k_filtered_ruleset_AB_file,
    #     'predictions': k_filtered_prediction_AB_file
    # }
    #
    # directory_type = 'AB'
    #
    # run_anyburl_apply(PARAMS['dataset'], files_AB, directory_type)

    ###################################
    # Created PC-based predictions on filtered ruleset files
    ###################################
    for step_k in range(k_step, k + 1, k_step):
        rule_file = f"{k_filtered_ruleset_PC_file}k_{step_k}.txt"
        prediction_file = f"PC_predictions_{PARAMS['dataset']}_train_missing_{PARAMS['em_iteration']}_{PARAMS['confidence_percentage']}_k_{step_k}.txt"

        files_PC = {
            'rules': rule_file,
            'predictions': prediction_file
        }

        directory_type = 'PC'

        run_anyburl_apply(PARAMS['dataset'], files_PC, directory_type)

def merge_pc_predictions(k, max_rules):
    sorted_cond_prob_table_dir = DIRECTORIES['sorted_cond_prob_table']
    unmerged_dir = DIRECTORIES['k_filtered_prediction_PC']
    merged_dir = DIRECTORIES['merged_k_filtered_prediction_PC']

    os.makedirs(merged_dir, exist_ok=True)

    ###################################
    # Get Sorted Conditional Probability Table
    ###################################
    condprob_file = (
        f"{sorted_cond_prob_table_dir}/condprob_"
        f"{PARAMS['dataset']}_train_"
        f"{PARAMS['pc_type']}_"
        f"{PARAMS['em_iteration']}_"
        f"{PARAMS['confidence_percentage']}_"
        f"{PARAMS['cutoff_prob_threshold']}.csv"
    )

    print(f"Reading conditional probabilities from: {condprob_file}")
    cond_df = pd.read_csv(condprob_file)

    ###################################
    # Store predictions per test case
    ###################################
    merged = defaultdict(lambda: {"Heads": {}, "Tails": {}})
    test_cases = []

    for step in range(k):
        if step>max_rules:
            break
        prob = float(cond_df.iloc[step]["Probability"])

        pred_file = (
            f"{unmerged_dir}/PC_predictions_"
            f"{PARAMS['dataset']}_train_"
            f"{PARAMS['pc_type']}_"
            f"{PARAMS['em_iteration']}_"
            f"{PARAMS['confidence_percentage']}_k_{step+1}.txt"
        )

        if not os.path.exists(pred_file):
            print(f"Warning: Missing prediction file for k={step+1}: {pred_file}")
            continue

        parsed = parse_prediction_file(pred_file)

        for test_case, heads, tails in parsed:
            if step == 0:
                test_cases.append(test_case)

            # Add new head predictions
            for ent, _ in heads:
                if ent not in merged[test_case]["Heads"]:
                    merged[test_case]["Heads"][ent] = prob

            # Add new tail predictions
            for ent, _ in tails:
                if ent not in merged[test_case]["Tails"]:
                    merged[test_case]["Tails"][ent] = prob

    ###################################
    # Generate merged predictions
    ###################################
    out_file = (
        f"{merged_dir}/merged_predictions_"
        f"{PARAMS['dataset']}_train_"
        f"{PARAMS['pc_type']}_"
        f"{PARAMS['em_iteration']}_"
        f"{PARAMS['confidence_percentage']}_k1to{k}_pc_"
        f"{PARAMS['cutoff_prob_threshold']}.txt"
    )

    with open(out_file, "w") as f:
        for test_case in test_cases:
            f.write(test_case + "\n")

            heads = merged[test_case]["Heads"]
            tails = merged[test_case]["Tails"]

            head_line = ["Heads: "] + [f"{e}\t{s}" for e, s in sorted(heads.items(), key=lambda x: -x[1])] if sorted(heads.items(), key=lambda x: -x[1]) else ["Heads: "]
            tail_line = ["Tails: "] + [f"{e}\t{s}" for e, s in sorted(tails.items(), key=lambda x: -x[1])] if sorted(heads.items(), key=lambda x: -x[1]) else ["Tails: "]

            f.write(head_line[0] + '\t'.join(head_line[1:]) + '\n')
            f.write(tail_line[0] + '\t'.join(tail_line[1:]) + '\n')

    print(f"✅ Merged prediction file saved to: {out_file}")


run_prediction(k=k_rules_max, k_step=1)


for i in range(k_interval_step, k_rules_max_extra, k_interval_step):
    merge_pc_predictions(k=i, max_rules = k_rules_max)


