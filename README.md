# PC-Context-KGC


A framework for knowledge graph completion using Probabilistic Circuit (PC) guided inference with rule-based reasoning. This project implements PC1, PC2, and PC3 inference methods for improved knowledge graph completion compared to traditional confidence-based baseline.

## Table of Contents
- [Setup Requirements](#setup-requirements)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage Pipeline](#usage-pipeline)
- [Evaluation](#evaluation)
- [Output Files](#output-files)

## Setup Requirements

### Prerequisites
- **Python** >= 3.9
- **Julia** 1.6.7
- **Java** (for AnyBURL inference and evaluation engines)
- **AnyBURL** jar file

### Dependencies
- Python: Install via `pip install -r requirements.txt`
- Julia: ProbabilisticCircuits.jl, CSV.jl, DataFrames.jl, CUDA.jl, Random.jl.
- External: AnyBURL-23-1.jar

## Project Structure

```
PC-Context-Learning/
├── all_scripts/
│   ├── scripts_julia/           # Julia implementations
│   └── *.py                     # Python scripts
├── Data-AnyBURL/
│   ├── 1-OriginalDataset/       # Raw benchmark datasets
│   ├── 3-POST-Data-PC/          # Processed data for PC learning
│   ├── 4-Rules/                 # Learned rules and probabilities
│   └── 5-PC/                    # Trained probabilistic circuits
└── Tests-AnyBURL/               # Evaluation results
```

## Installation

### 1. Julia Setup
```bash
# Download Julia 1.6.7
wget https://julialang-s3.julialang.org/bin/linux/x64/1.6/julia-1.6.7-linux-x86_64.tar.gz

# Extract
tar -xvzf julia-1.6.7-linux-x86_64.tar.gz

# Add to PATH
echo 'export PATH="/path/to/julia-1.6.7/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Verify installation
julia --version
```

### 2. AnyBURL Setup
```bash
# Download AnyBURL jar file
wget https://web.informatik.uni-mannheim.de/AnyBURL/AnyBURL-23-1.jar

# Place in project root directory
```

### 3. Python Environment
```bash
pip install -r requirements.txt
```

## Usage Pipeline

The framework follows a sequential pipeline from data preprocessing to evaluation. All examples use **Family** dataset with **confidence_threshold=0.0** as default.
Please update the parameters accordingly for other results.

### Phase 1: Data Preprocessing

#### 1-0-PyClause.py
**Purpose**: Generate PyClause groundings for rule-context matrix creation.

```bash
cd all_scripts
python 1-0-PyClause.py
```

**Input**: 
- `../Data-AnyBURL/1-OriginalDataset/Family/train.txt`
- `../Data-AnyBURL/1-OriginalDataset/Family/family_10_0.0_10_confidence_1.txt`

**Output**: 
- `../Data-AnyBURL/1-OriginalDataset/Family/target_triples.txt`
- `../Data-AnyBURL/1-OriginalDataset/Family/triple_pred_rules_grounding_0.txt`

#### 1-1-BinaryMatrixPyclause_script.py
**Purpose**: Create binary rule-context matrix with 0/1 entries.

```bash
python 1-1-BinaryMatrixPyclause_script.py --dataset Family --confidence 0
```

**Input**: 
- `../Data-AnyBURL/1-OriginalDataset/Family/target_triples.txt`
- `../Data-AnyBURL/1-OriginalDataset/Family/family_10_0.0_10.txt`
- `../Data-AnyBURL/1-OriginalDataset/Family/triple_pred_rules_grounding_0.txt`

**Output**: 
- `../Data-AnyBURL/3-POST-Data-PC/3-3-2-2-BM_PyClause/Family_bm_int_train_0.csv`

### Phase 2: PC Learning

#### 2-1-PCbyEM_PyClause.jl
**Purpose**: Learn probabilistic circuit structure and parameters using EM algorithm.

```bash
cd all_scripts/scripts_julia
julia 2-1-PCbyEM_PyClause.jl
```

**Input**: 
- `../Data-AnyBURL/3-POST-Data-PC/3-3-2-2-BM_PyClause/Family_bm_int_train_0.csv`

**Output**: 
- `../Data-AnyBURL/5-PC/PC_EM_Family_train_missing_100_0_pyclause.jls`

#### 2-2-EM-EVI_AnyBURL_KMAP.jl
**Purpose**: Compute marginal probabilities and rule subsets using kMAP algorithm.

```bash
julia 2-2-EM-EVI_AnyBURL_KMAP.jl
```

**Input**: 
- PC file from previous step
- Binary matrix
- `../Data-AnyBURL/1-OriginalDataset/Family/family_10_0.0_10.csv`

**Output**: 
- `../Data-AnyBURL/4-Rules/4-2-3-Rules-EVI/RuleSubsetsWithProb/kmap_pyclause/ruleset_em_evi_Family_train_missing_100_0.csv`
- `../Data-AnyBURL/4-Rules/4-2-3-Rules-EVI/Marginals/kmap_pyclause/ruleset_em_mar_Family_train_missing_100_0.csv`

### Phase 3: Rule Selection

#### Option A: 2-3-RuleSelection_V2.jl (Greedy Walks)
**Purpose**: Generate rule subsets using greedy conditional probability walks.

```bash
julia 2-3-RuleSelection_V2.jl
```

**Input**: 
- PC file and marginal probability table from previous step

**Output**: 
- `../Data-AnyBURL/4-Rules/4-2-3-Rules-EVI/AllRules/condprob_Family_train_missing_100_0_0.0.csv`

#### Option B: 2-3-0-NoGreedyWalk.py (Singleton Rules)
**Purpose**: Generate singleton rulesets without greedy walks.

```bash
cd ../
python 2-3-0-NoGreedyWalk.py
```

**Input**: 
- Marginal probability CSV from Phase 2

**Output**: 
- `../Data-AnyBURL/4-Rules/4-2-3-Rules-EVI/NoGreedy/condprob_Family_train_missing_100_0_0.0.csv`

### Phase 4: Prediction Generation

#### 3-A-1-GenerateRulesAndPredictions_script.py (PC1 & PC3)
**Purpose**: Generate PC1 or PC3 predictions in AnyBURL format based on input comes from AllRules or NoGreedy directory.
Generates AnyBURL format prediction file to provide to AnyBURL evaluation engine.
Additionally, note that the k_max in this script represents the maximum size of ruleset you want.
As provided in our paper, we found near peak results for very small susbet with k = 100.
```bash
python 3-A-1-GenerateRulesAndPredictions_script.py
```

**Input**: 
- Conditional probability table from Phase 3
- Original rule files

**Output**: 
- `../Tests-AnyBURL/k_filtered_prediction_PC/Family/lower_bound/merged_predictions_Family_train_missing_100_0_k1to{k}_pc_0.0.txt`

#### 3-B-1-PyClauseMapping_script.py (PC2 Setup)
**Purpose**: Create rule mappings and rankings for PC2 predictions.

```bash
python 3-B-1-PyClauseMapping_script.py
```

**Input**: 
- Conditional probability table
- Original dataset files

**Output**: 
- `../Tests-AnyBURL/Family/mappings_triples_rules_k_0_100/predictedTriples_rules_0_k_{k}.csv`

#### 3-B-2-UpperBoundPrediction_script.py (PC2 Predictions)
**Purpose**: Generate upper bound predictions for PC2.

```bash
python 3-B-2-UpperBoundPrediction_script.py
```

**Input**: 
- Top-k rule files from mapping step

**Output**: 
- `../Tests-AnyBURL/k_filtered_prediction_PC/Family/upper_bound_0_100/predictions_Family_train_missing_100_0_0.0_k_{k}.txt`


#### 3-B-3-CalculateNegationProbability_script.jl (PC2 Probability Calculation)
**Purpose**: Calculate probability scores for PC2 predictions using marginal negation queries on the probabilistic circuit.

```bash
julia 3-B-3-CalculateNegationProbability_script.jl
```

#### 3-B-4-UpperBoundRewritePrediction_script.py (PC2 Rewrite)
**Purpose**: Rewrite PC2 predictions in AnyBURL prediction file format.

```bash
python 3-B-4-UpperBoundRewritePrediction_script.py
```

**Input**: 
- Upper bound predictions and triple-rule mappings

**Output**: 
- `../Tests-AnyBURL/k_filtered_prediction_PC/Family/upper_bound_rewritten_0_100/ub_predictions_Family_train_missing_100_0_0.0_k_{k}.txt`

#### 3-C-AnyBurlOnly.py (Baseline)
**Purpose**: Generate baseline AnyBURL predictions for comparison.

```bash
python 3-C-AnyBurlOnly.py
```

**Input**: 
- Original rule files sorted by confidence

**Output**: 
- `../Tests-AnyBURL/k_filtered_prediction_AB_all/Family/baseline_predictions_Family_train_0_k_{k}.txt`

## Evaluation

### Baseline Evaluation
```bash
python eval_anyburl_baseline_only.py
```

**Output**: `metrics_baseline_temp/metrics_Family_0_0.0_baseline.csv`

### PC1 Evaluation (PC1/PC3)
```bash
python eval_anyburl_pc1_only.py
```

**Output**: `metrics_lb_temp/metrics_Family_0_0.5_lb_gw.csv`

### PC2 Evaluation (PC2)
```bash
python eval_anyburl_pc2_only.py
```

**Output**: `metrics_ub_temp/metrics_Family_0_0.0_ub.csv`

## Output Files

### Key Output Directories

- **Learned PCs**: `Data-AnyBURL/5-PC/`
- **Rule Probabilities**: `Data-AnyBURL/4-Rules/4-2-3-Rules-EVI/`
- **Predictions**: `Tests-AnyBURL/k_filtered_prediction_*/`
- **Evaluation Metrics**: `metrics_*/`

### Metrics Files Format
Each evaluation generates CSV files with columns:
- `k_value`: Number of rules used
- `hits@1`: Hits@1 metric
- `hits@3`: Hits@3 metric  
- `hits@10`: Hits@10 metric
- `mrr`: Mean Reciprocal Rank

### Example Command Sequence for Family Dataset

```bash
# Complete pipeline for Family dataset
cd all_scripts

# Phase 1: Preprocessing
python 1-0-PyClause.py
python 1-1-BinaryMatrixPyclause_script.py --dataset Family --confidence 0

# Phase 2: PC Learning
cd scripts_julia
julia 2-1-PCbyEM_PyClause.jl
julia 2-2-EM-EVI_AnyBURL_KMAP.jl

# Phase 3: Rule Selection (choose one)
#PC3
julia 2-3-RuleSelection_V2.jl  # OR
cd ../
#PC1
python 2-3-0-NoGreedyWalk.py

# Phase 4: Predictions
python 3-A-1-GenerateRulesAndPredictions_script.py
python 3-B-1-PyClauseMapping_script.py
python 3-B-2-UpperBoundPrediction_script.py
python 3-B-4-UpperBoundRewritePrediction_script.py
python 3-C-AnyBurlOnly.py

# Phase 5: Evaluation
python eval_anyburl_baseline_only.py
python eval_anyburl_pc1_only.py
python eval_anyburl_pc2_only.py
```

## Configuration

Key parameters can be modified in each script:
- `DATASETS`: Choose from ['Family', 'Nations', 'UML', 'Kinship', 'WN18', 'WN18RR', 'FB15-237', 'CODEX-S']
- `confidence_percentage`: Rule confidence threshold (0-100)
- `em_iteration`: EM algorithm iterations for PC learning
- `k_rules_max`: Maximum number of rules to evaluate

## Notes

- Ensure all prerequisite files exist before running each script
- Scripts automatically create necessary output directories
- Family dataset added in the repo with rules. For other datasets, please download from the sources.
