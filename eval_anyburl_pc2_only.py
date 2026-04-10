import subprocess
import csv
import os

# To set variables for the ANYBURL eval script more easily
base_vars = {
    "dataset": "Family",
    "input_confidence": 0,
    "threshold": 0.0,
    "type": 'all',
    "em_iterations": 100
}
# all
# Range of k values to iterate over
k_range = range(100, 2600, 100)


os.makedirs("metrics_ub_temp", exist_ok=True)
csv_file_ub = f"metrics_ub_temp/metrics_{base_vars['dataset']}_{base_vars['input_confidence']}_{base_vars['threshold']}_ub.csv"

if not os.path.exists(csv_file_ub):
    with open(csv_file_ub, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["k_value", "hits@1", "hits@3", "hits@10", "mrr"])
# PATH_PREDICTIONS=Tests-AnyBURL/k_filtered_prediction_AB/{vars['dataset']}/AB_predictions_{vars['dataset']}_train_missing_1000_{base_vars['input_confidence']}_k_{k}.txt
# PATH_PREDICTIONS=Tests-AnyBURL/k_filtered_prediction_PC/{vars['dataset']}/lower_bound/merged_predictions_{vars['dataset']}_train_missing_1000_{base_vars['input_confidence']}_k1to{k}_pc_{vars['threshold']}.txt
# PATH_PREDICTIONS=Tests-AnyBURL/k_filtered_prediction_PC/{vars['dataset']}/upper_bound_rewritten/ub_predictions_{vars['dataset']}_train_missing_1000_{base_vars['input_confidence']}_{vars['threshold']}_k_{k}.txt

if base_vars['type'] == 'all':

    for k in k_range:
        # Update vars with current k
        vars = base_vars.copy()
        vars["k"] = k

        path_pred = f"Tests-AnyBURL/k_filtered_prediction_PC/{vars['dataset']}/upper_bound_rewritten_{base_vars['input_confidence']}_{base_vars['em_iterations']}/ub_predictions_{vars['dataset']}_train_missing_{vars['em_iterations']}_{base_vars['input_confidence']}_{vars['threshold']}_k_{k}.txt"
        # Template
        lines = f"""\
        PATH_TRAINING=Data-AnyBURL/1-OriginalDataset/{vars['dataset']}/train.txt
        PATH_TEST=Data-AnyBURL/1-OriginalDataset/{vars['dataset']}/test.txt
        PATH_PREDICTIONS={path_pred}
        TOP_K=1000
        """

        # Clean up any whitespace
        properties_template = "\n".join([line.strip() for line in lines.strip().splitlines()])

        # Write config file
        config_path = f"config-eval-ub-{base_vars['dataset']}.generated.properties"
        with open(config_path, "w") as f:
            f.write(properties_template)

        print(f"Wrote config to {config_path}")

        # Run eval
        command = [
            "java",
            "-Xmx12G",
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
        with open("anyburl_output.txt", "w") as f:
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
            with open(csv_file_ub, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([k, hits_1, hits_3, hits_10, mrr])
            print(f"Metrics for k={k} appended to {csv_file_ub}")
        else:
            print(f"Warning: Metrics line for k={k} does not contain 4 values: {metrics_line}")

