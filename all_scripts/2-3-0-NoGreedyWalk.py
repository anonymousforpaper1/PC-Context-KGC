import pandas as pd
import os

# =============================================================================
# CONFIGURATION VARIABLES
# =============================================================================
DATASETS = ['Family', 'Nations', 'UML', 'Kinship', 'WN18', 'WN18RR', 'FB15-237', 'CODEX-S']
source = DATASETS[0]  # "Family" - change index to select different dataset
confidence_percentage = 0

filetype = "train"  # Options: "valid", "test", "train"
pctype = "missing"  # Options: "imputed", "missing"
em_iteration = 100
confidence_threshold = confidence_percentage / 100
cutoff_prob_threshold = 0.0  # May need to adjust this. 0.7 created a HUGE subdirectory.

# Directory paths
BASE_DIR = "../Data-AnyBURL/4-Rules/4-2-3-Rules-EVI"
INPUT_DIR = f"{BASE_DIR}/Marginals/kmap_pyclause"
OUTPUT_DIR = f"{BASE_DIR}/NoGreedy"


# =============================================================================
# FUNCTIONS
# =============================================================================

def generate_input_filename():
    """Generate input filename based on configuration variables"""
    return f"ruleset_em_mar_{source}_{filetype}_{pctype}_{em_iteration}_{confidence_percentage}.csv"


def generate_output_filename():
    """Generate output filename in the specified format"""
    return f"condprob_{source}_{filetype}_{pctype}_{em_iteration}_{confidence_percentage}_{cutoff_prob_threshold}.csv"


def process_ruleset_csv(input_file, output_file):
    """
    Process the ruleset CSV file with the following modifications:
    1. Sort rows by Probability in descending order
    2. Rename 'Rules' column to 'RuleSets'
    3. Convert rule values (e.g., 'R332') to list format (e.g., '["R332"]')
    4. Save to specified output file

    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
    """

    # Read the CSV file
    print(f"Reading file: {input_file}")
    df = pd.read_csv(input_file)

    print(f"Original data shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")

    # 1. Sort by Probability in descending order
    print("Sorting by Probability in descending order...")
    df_sorted = df.sort_values('Probability', ascending=False).reset_index(drop=True)

    # 2. Rename 'Rules' column to 'RuleSets'
    print("Renaming 'Rules' column to 'RuleSets'...")
    df_sorted = df_sorted.rename(columns={'Rules': 'RuleSets'})

    # 3. Convert rule values to list format ["R332"]
    print("Converting rule values to list format...")
    df_sorted['RuleSets'] = df_sorted['RuleSets'].apply(lambda x: f'["{x}"]')

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    # Save the processed file
    print(f"Saving processed file to: {output_file}")
    df_sorted.to_csv(output_file, index=False)

    print(f"Processing complete!")
    print(f"Output file shape: {df_sorted.shape}")
    print(f"Updated columns: {list(df_sorted.columns)}")

    # Display first few rows for verification
    print("\nFirst 5 rows of processed data:")
    print(df_sorted.head())

    # Display statistics
    print(f"\nProbability range: {df_sorted['Probability'].min():.6f} to {df_sorted['Probability'].max():.6f}")
    print(f"Sample RuleSets values: {df_sorted['RuleSets'].head(3).tolist()}")

    return df_sorted


def print_configuration():
    """Print current configuration for verification"""
    print("=" * 60)
    print("CURRENT CONFIGURATION:")
    print("=" * 60)
    print(f"Dataset: {source}")
    print(f"File type: {filetype}")
    print(f"PC type: {pctype}")
    print(f"EM iteration: {em_iteration}")
    print(f"Confidence percentage: {confidence_percentage}")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"Cutoff probability threshold: {cutoff_prob_threshold}")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Print current configuration
    print_configuration()

    # Generate file paths
    input_filename = generate_input_filename()
    output_filename = generate_output_filename()

    input_file_path = os.path.join(INPUT_DIR, input_filename)
    output_file_path = os.path.join(OUTPUT_DIR, output_filename)

    print(f"\nInput file: {input_file_path}")
    print(f"Output file: {output_file_path}")

    # Process the file
    try:
        processed_df = process_ruleset_csv(input_file_path, output_file_path)
        print(f"\n✅ File processing completed successfully!")
        print(f"✅ Output saved as: {output_filename}")

    except FileNotFoundError:
        print(f"❌ Error: File '{input_file_path}' not found.")
        print("Please make sure the file exists and check your configuration variables.")
        print("\nAvailable datasets:", DATASETS)

    except Exception as e:
        print(f"❌ Error occurred during processing: {str(e)}")

