import subprocess
import csv
import os

# To set variables for the ANYBURL eval script more easily
base_vars = {
    "dataset": "Kinship",
    "input_confidence":0,
    "MULTI": True,
    "threshold": 0.0,
    "type": 'all',
    "em_iterations": 25
}
# all
# Range of k values to iterate over
k_range = range(37000, 37100, 37000)  # Example: 37000 to 37000 with step of 37000

if base_vars['MULTI']:
    original_ruleset_dir = "Data-AnyBURL/1-MultiDataset"
else:
    original_ruleset_dir = "Data-AnyBURL/1-OriginalDataset"

os.makedirs("all_scripts/metrics_baseline_temp", exist_ok=True)
csv_file_baseline = f"metrics_baseline_temp/metrics_{base_vars['dataset']}_{base_vars['input_confidence']}_{base_vars['threshold']}_baseline.csv"
if not os.path.exists(csv_file_baseline):
    with open(csv_file_baseline, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["k_value", "hits@1", "hits@3", "hits@10", "mrr"])

if base_vars['type'] == 'all':

    for k in k_range:
        # Update vars with current k
        vars = base_vars.copy()
        vars["k"] = k

        path_pred = f"Tests-AnyBURL/k_filtered_prediction_AB_all/{vars['dataset']}/baseline_predictions_{vars['dataset']}_train_{base_vars['input_confidence']}_k_{k}.txt"
        # Template
        lines = f"""\
        PATH_TRAINING={original_ruleset_dir}/{vars['dataset']}/train.txt
        PATH_TEST={original_ruleset_dir}/{vars['dataset']}/test.txt
        PATH_PREDICTIONS={path_pred}
        TOP_K=1000
        """

        # Clean up any whitespace
        properties_template = "\n".join([line.strip() for line in lines.strip().splitlines()])

        # Write config file
        config_path = f"config-eval-baseline-{base_vars['dataset']}.generated.properties"
        with open(config_path, "w") as f:
            f.write(properties_template)

        print(f"Wrote config to {config_path}")

        # Run eval
        command = [
            "java",
            "-Xmx150G",
            "-cp", "AnyBURL-23-1.jar",
            "de.unima.ki.anyburl.Eval",
            config_path
        ]

        print(f"Running AnyBURL Eval for k={k}...")
        # Capture output
        result = subprocess.run(command, capture_output=True, text=True)

        # Store stdout and stderr
        output = result.stdout  # Standard output
        errors = result.stderr  # Standard error

        # Print output for debugging
        print("Output:")
        print(output)
        if errors:
            print("Errors:")
            print(errors)

        # Save to anyburl_output.txt
        with open("all_scripts/anyburl_output.txt", "w") as f:
            f.write("Output:\n")
            f.write(output)
            if errors:
                f.write("\nErrors:\n")
                f.write(errors)

        print(f"Command execution completed for k={k}. Output saved to 'anyburl_output.txt'.")

        # Extract metrics from output (assuming line 10 contains "hits@1 hits@3 hits@10 mrr")
        output_lines = output.splitlines()
        metrics_line = output_lines[-1]  # Line 10 (index 9 since 0-based)
        metrics = metrics_line.split()  # Split by whitespace
        if len(metrics) == 4:  # Ensure we have 4 values
            hits_1, hits_3, hits_10, mrr = map(float, metrics)

            # Append to CSV
            with open(csv_file_baseline, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([k, hits_1, hits_3, hits_10, mrr])
            print(f"Metrics for k={k} appended to {csv_file_baseline}")
        else:
            print(f"Warning: Metrics line for k={k} does not contain 4 values: {metrics_line}")

