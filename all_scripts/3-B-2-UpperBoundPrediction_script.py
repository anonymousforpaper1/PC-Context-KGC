import pandas as pd
import os
import subprocess

DATASETS = ['Family', 'Nations', 'UML', 'Kinship', 'WN18', 'WN18RR', 'FB15-237', 'CODEX-S']

dataset = DATASETS[0]
k_start = 100
k_end = 2600
k_intervals = 100
confidence_percentage = 0
em_iterations = 100

cutoff_prob_threshold = 0.0

confidence_threshold = confidence_percentage/100

train_path = f'../Data-AnyBURL/1-OriginalDataset/{dataset}/train.txt'
test_path = f'../Data-AnyBURL/1-OriginalDataset/{dataset}/test.txt'

sorted_rule_files_k_dir = f'../Tests-AnyBURL/{dataset}/sorted_rule_files_k_{confidence_percentage}_{em_iterations}'
filtered_prediction_PC_dir = f'../Tests-AnyBURL/k_filtered_prediction_PC/{dataset}/upper_bound_{confidence_percentage}_{em_iterations}'
os.makedirs(filtered_prediction_PC_dir, exist_ok=True)


def run_anyburl_apply(train_path, test_path, rules_path, output_prediction):

    # Resolve absolute paths
    input_abs = os.path.abspath(f"{rules_path}")
    output_abs = os.path.abspath(f"{output_prediction}")
    train_abs = os.path.abspath(f"{train_path}")
    test_abs = os.path.abspath(f"{test_path}")

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

    config_path = os.path.abspath(f"../config-apply-UB-{dataset}.generated.properties")
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


for k in range(k_start, k_end+k_intervals, k_intervals):
    top_k_sorted_rule_file = f'{sorted_rule_files_k_dir}/rules_top_k_{k}.txt'
    prediction_file_path = f'{filtered_prediction_PC_dir}/predictions_{dataset}_train_missing_{em_iterations}_{confidence_percentage}_{cutoff_prob_threshold}_k_{k}.txt'
    run_anyburl_apply(train_path=train_path, test_path=test_path, rules_path=top_k_sorted_rule_file,
                      output_prediction=prediction_file_path)
