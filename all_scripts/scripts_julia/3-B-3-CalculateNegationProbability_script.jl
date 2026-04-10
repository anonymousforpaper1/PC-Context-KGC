using Pkg                # Use the Pkg module
Pkg.activate("Project.toml")       # For defining environment, . = current directory, .. = immediate previous directory, "path" = some other directory
#Pkg.instantiate()        # Installs and Precompiles all dependencies (We have installed Random already)

# import Pkg;
# Pkg.precompile()
#import Pkg; Pkg.add("YAML")

using Serialization
using CSV
using DataFrames
using ProbabilisticCircuits
using OrderedCollections
# using JSON3
# using YAML

DATASETS = ["Family", "Nations", "UML", "Kinship", "WN18", "WN18RR", "FB15-237", "CODEX-S"]

source = [DATASETS[1]]
dataset = DATASETS[1]
k_start = 100
k_end = 2600
k_interval = 100
confidence_percentage = 0
em_iteration = [100];

cutoff_prob_threshold = 0.0

bm_path = "../../Data-AnyBURL/3-POST-Data-PC/3-3-2-2-BM_PyClause/$(dataset)_bm_int_train_$(confidence_percentage).csv"
THRESHOLD = 0.5
confidence_threshold = confidence_percentage/100

filetype = ["train"];
pctype = ["missing"];


bm_df = CSV.read(bm_path, DataFrame)

# Extract row names (rules like R1, R2, ...)
active_rules = string.(bm_df[:, 1])

println("Active Rules: ", active_rules)


function get_source_data(source, filetype, pctype, confidence_percentage)
    if pctype === "missing"
        filepath = "../../Data-AnyBURL/3-POST-Data-PC/3-3-2-2-BM_PyClause/"
        # filepath = filepath * string(source) * "_bm_int_" * string(filetype) * ".csv"
        filepath = filepath * string(source) * "_bm_int_" * string(filetype) * "_" * string(confidence_percentage) * ".csv"
    elseif pctype === "imputed" # we dont even need this
        filepath = "../../Data-AnyBURL/3-POST-Data-PC/3-3-3-2-BM_PyClause/"
        filepath = filepath * string(source) * "_bm_imputed_int_" * string(filetype) * "_" * string(confidence_percentage) * ".csv"
    end

    println("Working on: " * string(filepath))

    # Read the data as CSV and Pipe to a Dataframe (This automatically takes the first row as column title)
    df = CSV.File(filepath) |> DataFrame

    # Julia Dataframe do not support strings as row index. So we need to remove our first column (index column).
    # Also to keep track of the numeric row index with the actual row title, we will create a mapping dictionary

    # Get all actual row indices (first column of the data) in a dictionary
    row_headers = df[:, 1]  # Extract the first column
    row_names_array = collect(row_headers)
    row_names = Dict(row_names_array[i] => i for i in 1:length(row_names_array))

    # Delete the index column
    df = df[:, 2:end]  # Remove the first column from the DataFrame

    return df
end;

function transpose_and_convert_to_boolean(data)
    # Transpose the DataFrame
    data_transposed = DataFrame(permutedims(Matrix(data)), :auto)

    # Convert Float64 to Bool (using a threshold of 0.5)
    for col in names(data_transposed)
        data_transposed[!, col] = data_transposed[!, col] .> THRESHOLD
    end

    # CSV.write("t_bm.csv", data_transposed)

    return data_transposed
end;


function prepare_data(source, filetype, pctype, confidence_percentage)
    if pctype === "missing"
        filepath = "../../Data-AnyBURL/3-POST-Data-PC/3-3-2-2-BM_PyClause/"
        # filepath = filepath * string(source) * "_bm_int_" * string(filetype) * ".csv"
        filepath = filepath * string(source) * "_bm_int_" * string(filetype) * "_" * string(confidence_percentage) * ".csv"
    elseif pctype === "imputed" # we dont even need this
        filepath = "../../Data-AnyBURL/3-POST-Data-PC/3-3-3-2-BM_PyClause/"
        filepath = filepath * string(source) * "_bm_imputed_int_" * string(filetype) * "_" * string(confidence_percentage) * ".csv"
    end

    df = CSV.File(filepath) |> DataFrame
    df = df[:, 2:end]  # Remove the first column from the DataFrame (the index column)
    bool_df = mapcols(col -> col .== 1, df)  # Convert 1 → true, 0 → false for all columns
    return bool_df
end;

function prepare_active_rules(source, filetype, pctype, em_iteration, confidence_percentage, cutoff_prob_threshold)
    if pctype === "missing"
        filepath = "../../Data-AnyBURL/3-POST-Data-PC/3-3-2-2-BM_PyClause/"
        # filepath = filepath * string(source) * "_bm_int_" * string(filetype) * ".csv"
        filepath = filepath * string(source) * "_bm_int_" * string(filetype) * "_" * string(confidence_percentage) * ".csv"
    elseif pctype === "imputed" # we dont even need this
        filepath = "../../Data-AnyBURL/3-POST-Data-PC/3-3-3-2-BM_PyClause/"
        filepath = filepath * string(source) * "_bm_imputed_int_" * string(filetype) * "_" * string(confidence_percentage) * ".csv"
    end

    bm_df = CSV.read(filepath, DataFrame)

    # Extract row names (rules like R1, R2, ...)
    active_rules = string.(bm_df[:, 1])

    return bm_df
end;


function get_pc_learned_using_em(source, filetype, pctype, em_iteration, confidence_percentage)
    filename_pc = "PC_EM_" * string(source) * "_" * string(filetype) * "_" * string(pctype)
    # filename_pc = filename_pc * "_" * string(em_iteration) * "_" * string(confidence_percentage) * ".jls"
    filename_pc = filename_pc * "_" * string(em_iteration) * "_" * string(confidence_percentage) * "_pyclause.jls"
    pc_dir = "../../Data-AnyBURL/5-PC/"
    pc_loc = pc_dir * filename_pc
    pc = open(pc_loc, "r") do io
        deserialize(io)
    end

    return pc
end;

function parse_ruleset(str)
    stripped = strip(str, ['[', ']'])
    parts = split(stripped, ",")
    return [strip(r, [' ', '"']) for r in parts]
end

function parse_csv_triplet_rules(csv_path::String)
    println("📄 Parsing CSV from: ", csv_path)
    df = CSV.read(csv_path, DataFrame)
    return df
end

function parse_rules_list(str::AbstractString)
    # Remove brackets and quotes, then split
    cleaned = replace(str, r"\[|\]|'|\"" => "")
    return split(cleaned, ", ")
end

function get_MAR_from_triplet_csv(pc, csv_path::String, active_rules::Vector{String})
    df = parse_csv_triplet_rules(csv_path)

    marlls = Float64[]
    probs = Float64[]

    for i in 1:nrow(df)
        triplet = df.predicted_triples[i]
        rules_to_clamp_str = df.rules_fired[i]
        rules_to_clamp = parse_rules_list(rules_to_clamp_str)

        col_indices = findall(r -> r in rules_to_clamp, active_rules)
        # println("🔧 Row $i → Triplet: $triplet → Rules to set FALSE: ", rules_to_clamp)
        # println("   → Columns to set FALSE: ", col_indices)

        row_vals = [r in rules_to_clamp ? false : missing for r in active_rules]
        data_new = DataFrame([Symbol(r) => [v] for (r, v) in zip(active_rules, row_vals)])

        marll = MAR(pc, data_new)[1]
        p = 1 - exp(marll)

        push!(marlls, marll)
        push!(probs, p)
    end

    df.MarLLFromNegation = marlls
    df.ProbabilityFromNegation = probs

    output_csv_path = replace(csv_path, ".csv" => "_with_marginals.csv")
    CSV.write(output_csv_path, df)

    return df
end


function get_ruleprogram_subsets(source, filetype, pctype, em_iteration, confidence_percentage, cutoff_prob_threshold, dictionary_loc)
    for s in source
        for ft in filetype
            for pt in pctype
                for emit in em_iteration
                    println(s, " ", ft, " ", pt, " ", emit, " ", confidence_percentage)

                    # Get Data
                    println("Getting data...")
                    data = prepare_data(s, ft, pt, confidence_percentage)
                    # data_tr = transpose_and_convert_to_boolean(data)

                    # Get Ruleset
                    # ruleset = get_rule_files(s, cc)

                    ###################################
                    # Get the PC
                    ###################################
                    println("Getting the PC...")
                    pc = get_pc_learned_using_em(s, ft, pt, emit, confidence_percentage)
                    println("PC", pc)
                    # println("Data reading done...")

                    ###################################
                    # Get active rules
                    ###################################
                    println("Getting active rules...")
                    bm_df = prepare_active_rules(s, ft, pt, emit, confidence_percentage, cutoff_prob_threshold)
                    active_rules = string.(bm_df[:, 1])
                    # active_rules = ["R4532"]
                    active_rules = collect(String.(bm_df[:, 1]))

                    # println("Active Rules: ", active_rules)

                    ###################################
                    # Get MAR
                    ###################################
                    # common_name = string(s) * "_" * string(ft) * "_" * string(pt)* "_" * string(emit) * "_" * string(confidence_percentage)
                    # unfiltered_ruleset_dir = "../Data-AnyBURL/4-Rules/4-2-3-Rules-EVI/AllRules/"
                    # dictionary_loc = "../Test/1-OriginalDataset/" * string(s) * "/predictedTriples_rules_" * string(confidence_percentage)  * ".csv"
                    get_MAR_from_triplet_csv(pc, dictionary_loc, active_rules)
                end
            end
        end
    end
end;



for k in k_start:k_interval:k_end
    dictionary_loc = "../../Tests-AnyBURL/" * string(dataset) * "/mappings_triples_rules_k_" * string(confidence_percentage) * "_" * string(em_iteration[1]) * "/predictedTriples_rules_" * string(confidence_percentage)  * "_k_" * string(k) * ".csv"
    get_ruleprogram_subsets(source, filetype, pctype, em_iteration, confidence_percentage, cutoff_prob_threshold, dictionary_loc)
end

