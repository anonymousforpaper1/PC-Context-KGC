import pandas as pd
import ast
import os
import subprocess
import decimal
import math
from collections import defaultdict

DATASETS = ['Family', 'Nations', 'UML', 'Kinship', 'WN18', 'WN18RR', 'FB15-237', 'CODEX-S']

PARAMS = {
    'dataset': DATASETS[3],
    'confidence_percentage': 0
}
k_start = 37000
k_end = 37001
k_interval = 37000
MULTI = True

if not MULTI:
    original_ruleset_dir = "../Data-AnyBURL/1-OriginalDataset"
else:
    original_ruleset_dir = "../Data-AnyBURL/1-MultiDataset"

k_filtered_ruleset_AB_dir = f'../Tests-AnyBURL/k_filtered_rulesets_AB_all/{PARAMS["dataset"]}/'
os.makedirs(k_filtered_ruleset_AB_dir, exist_ok=True)

k_filtered_prediction_AB_dir = f'../Tests-AnyBURL/k_filtered_prediction_AB_all/{PARAMS["dataset"]}/'
os.makedirs(k_filtered_prediction_AB_dir, exist_ok=True)

k_filtered_ruleset_PC_dir = f'../Tests-AnyBURL/k_filtered_rulesets_PC/{PARAMS["dataset"]}/'
os.makedirs(k_filtered_ruleset_PC_dir, exist_ok=True)

DIRECTORIES = {
    'original_ruleset': original_ruleset_dir,
    'k_filtered_ruleset_AB': k_filtered_ruleset_AB_dir,
    'k_filtered_prediction_AB': k_filtered_prediction_AB_dir
}

# Helper function to parse AnyBURL style rule files
def parse_rule_line(line):
    parts = line.strip().split("\t")
    if len(parts) < 3:
        return None, 0.0  # skip malformed
    confidence = float(parts[2])
    return line.strip(), confidence


def run_anyburl_apply(dataset, files, directory_type='AB'):

    ruleset_dir = f"k_filtered_ruleset_{directory_type}"
    prediction_dir = f"k_filtered_prediction_{directory_type}"

    # Resolve absolute paths
    input_abs = os.path.abspath(f"{DIRECTORIES[ruleset_dir]}/{files['rules']}")
    output_abs = os.path.abspath(f"{DIRECTORIES[prediction_dir]}/{files['predictions']}")
    train_abs = os.path.abspath(f"{original_ruleset_dir}/{dataset}/train.txt")
    test_abs = os.path.abspath(f"{original_ruleset_dir}/{dataset}/test.txt")

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

    config_path = os.path.abspath(f"../config-apply-baseline-{dataset}.generated.properties")
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


def run_prediction(n_start, n_interval, n_end):
    ###################################
    # Setting up DIRECTORIES
    ###################################s

    original_ruleset_dir = DIRECTORIES['original_ruleset']
    k_filtered_ruleset_AB_dir = DIRECTORIES['k_filtered_ruleset_AB']
    k_filtered_prediction_AB_dir = DIRECTORIES['k_filtered_prediction_AB']

    ###################################
    # Get the Original Full Ruleset
    ###################################
    original_ruleset_loc = (
        f"{original_ruleset_dir}/"
        f"{PARAMS['dataset']}/"
        f"{PARAMS['dataset'].lower()}_10_{PARAMS['confidence_percentage'] / 100}_10.txt"
    )

    print(f'\nReading Original Full Ruleset from: {original_ruleset_loc}...')
    with open(original_ruleset_loc, "r") as f:
        rules = f.readlines()

    ###################################
    # Sort full ruleset by confidence and keep top-n
    ###################################
    parsed_rules = [parse_rule_line(line) for line in rules]
    parsed_rules = [r for r in parsed_rules if r[0] is not None]
    sorted_rules = sorted(parsed_rules, key=lambda x: x[1], reverse=True)

    # for i in range(n_start, n_interval, len(rules) + n_interval):
    for i in range(n_start, n_end + n_interval, n_interval):

        top_n_rules = sorted_rules[:i]

        ###################################
        # Save to filtered ruleset file
        ###################################
        k_filtered_ruleset_AB_file = f"{PARAMS['dataset'].lower()}_10_{PARAMS['confidence_percentage'] / 100}_10_k_temp.txt"

        k_filtered_ruleset_AB_loc = (
            f"{k_filtered_ruleset_AB_dir}"
            f"{k_filtered_ruleset_AB_file}"
        )

        with open(k_filtered_ruleset_AB_loc, "w") as f:
            for line, _ in top_n_rules:
                f.write(line + "\n")

        print(f'\nCreated k-filtered Ruleset with top-{i} rules at: {k_filtered_ruleset_AB_loc}')

        ###################################
        # Created baseline predictions on filtered ruleset files
        ###################################
        k_filtered_prediction_AB_file = (
            f"baseline_predictions_{PARAMS['dataset']}_train_"
            f"{PARAMS['confidence_percentage']}_"
            f"k_{i}.txt"
        )

        k_filtered_prediction_AB_loc = (
            f"{k_filtered_prediction_AB_dir}"
            f"{k_filtered_prediction_AB_file}"
        )

        print(f'\nWriting predictions to: {k_filtered_prediction_AB_loc}')

        ###################################
        # Created baseline predictions on filtered ruleset files
        ###################################
        files_AB = {
            'rules': k_filtered_ruleset_AB_file,
            'predictions': k_filtered_prediction_AB_file
        }

        directory_type = 'AB'

        run_anyburl_apply(PARAMS['dataset'], files_AB, directory_type)


run_prediction(n_start=k_start, n_interval=k_interval, n_end=k_end)
