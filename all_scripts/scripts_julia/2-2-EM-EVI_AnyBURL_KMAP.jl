# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Julia 1.6.7
#     language: julia
#     name: julia-1.6
# ---

# %% [markdown]
# # Uses KMAP algorithm to form subsets with associated probabilities

# %%
using Pkg                # Use the Pkg module 
Pkg.activate("Project.toml")       # For defining environment, . = current directory, .. = immediate previous directory, "path" = some other directory
#Pkg.instantiate()        # Installs and Precompiles all dependencies (We have installed Random already)

# %%
# import Pkg;
# Pkg.precompile()

# %%
using Serialization

using CSV
using DataFrames
using ProbabilisticCircuits
using OrderedCollections

# %% [markdown]
# - UML: k=50, conf=50
# - Kinship: k=50, conf= 20 (reduce k to 10 maybe)
# - WN18RR: k=10, conf = 0

# %% [markdown]
# # Tunable Parameter

# %%
DATASETS = ["Family", "Nations", "UML", "Kinship", "WN18", "WN18RR", "FB15-237", "CODEX-S"]
source = [DATASETS[1]]

# %%
# filetype = ["valid", "test"]; 
filetype = ["train"]; 

# %%
# pctype = ["imputed"]; 
pctype = ["missing"]; 

# %%
em_iteration = [100];

# %%
confidence_percentage = 0
confidence_threshold = confidence_percentage/100

# %% [markdown]
# ### How many rule subsets to produce

# %% does not matter the value of k
k = 100

# %%
THRESHOLD = 0.5

# %% [markdown]
# # Get Data and Preprocess

# %%
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

# %%
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

# %%
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

# %%
function prepare_active_rules(source, filetype, pctype, confidence_percentage)
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

    println("Active Rules: ", active_rules)
    return bm_df
end;

# %% [markdown]
# # Get EM Learned PC

# %%
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

# %% [markdown]
# # kMAP algorithm

# %%
function kMAP(circuit::StructProbCircuit, k::Int = 3)
    @assert circuit isa StructSumNode || circuit isa StructMulNode "The input circuit must be a valid sum or product node"
    kmap_dict = Dict{StructProbCircuit, Vector{Vector{Union{Vector{Int32}, Float64}}}}()
    kmap_ = kmap(circuit, k, kmap_dict)
    return [map(x -> x isa Vector{Int32} ? (x .> 0) : x, row) for row in kmap_]
end

function kmap(node::StructSumNode, k::Int, kmap_dict::Dict{StructProbCircuit, Vector{Vector{Union{Vector{Int32}, Float64}}}})
    if node in keys(kmap_dict)
        return kmap_dict[node]
    end

    kmap_ = Vector{Vector{Union{Vector{Int32}, Float64}}}()
    
    for (i, child) in enumerate(node.children)
        _ = kmap(child, k, kmap_dict)
        kmap_ = max_merge_kmaps(kmap_, kmap_dict[child], node.log_probs[i], k)
    end
    
    kmap_dict[node] = kmap_[:]
    return kmap_
end

function kmap(node::StructMulNode, k::Int, kmap_dict::Dict{StructProbCircuit, Vector{Vector{Union{Vector{Int32}, Float64}}}})
    if node in keys(kmap_dict)
        return kmap_dict[node]
    end

    kmap_ = Vector{Vector{Union{Vector{Int32}, Float64}}}()
    
    # Recursively compute kmap for both children
    _ = kmap(node.prime, k, kmap_dict)
    _ = kmap(node.sub, k, kmap_dict)

    # Merge prime and sub kmaps using cartesian merge
    kmap_ = cart_merge_kmaps(kmap_, kmap_dict[node.prime], kmap_dict[node.sub], k)

    kmap_dict[node] = kmap_[:]
    return kmap_
end

# Unchanged
function kmap(node::StructProbLiteralNode, k::Int, kmap_dict::Dict{StructProbCircuit, Vector{Vector{Union{Vector{Int32}, Float64}}}})
    kmap_ = [[[node.literal], 0.0]]  # Store literal value with log probability 0
    kmap_dict[node] = kmap_[:]
    return kmap_
end

# Unchanged
function max_merge_kmaps(kmap1::Vector{Vector{Union{Vector{Int32}, Float64}}}, kmap2::Vector{Vector{Union{Vector{Int32}, Float64}}}, log_prob::Float64, k::Int)
    kmap_ = kmap1[:]  # Copy existing kmap
    for kmap2_element in kmap2
        push!(kmap_, [kmap2_element[1], kmap2_element[2] + log_prob])
    end
    sort!(kmap_, by=x->x[2], rev=true)  # Sort by log probability
    return kmap_[1:min(k, length(kmap_))]
end

# Unchanged
function cart_merge_kmaps(kmap_::Vector{Vector{Union{Vector{Int32}, Float64}}}, kmap1::Vector{Vector{Union{Vector{Int32}, Float64}}}, kmap2::Vector{Vector{Union{Vector{Int32}, Float64}}}, k::Int)
    kmap_res = kmap_[:]
    for kmap1_ in kmap1
        for kmap2_ in kmap2
            push!(kmap_res, [sort(append!(kmap1_[1][:], kmap2_[1][:]); by=x->abs(x)), kmap1_[2] + kmap2_[2]])
        end
    end
    sort!(kmap_res, by=x->x[2], rev=true)
    return kmap_res[1:min(k, length(kmap_res))]
end

# Compute kMAP for k=3
# println(kMAP(pc, 3))
# new_pc = kMAP(pc, 3)

# %% [markdown]
# # Get K Best Rules Using EVI Query 

# %%
function get_rule_program_subsets_with_probabilities(pc, data, active_rules, k::Int)
    ########################################################################
    # Step - 1: Get the k-best assignments from kMAP
    ########################################################################
    k_best_assignments = kMAP(pc, k)  # Get top-k most probable rule assignments

    ########################################################################
    # Step - 2: Prepare DataFrame to Store Results (Including RuleCount)
    ########################################################################
    result = DataFrame(RuleSubset = Vector{Vector{String}}(undef, k), 
                       RuleCount = zeros(Int, k),  
                       LogLikelihood = zeros(Float64, k), 
                       Probability = zeros(Float64, k))

    ########################################################################
    # Step - 3: Convert kMAP Assignments to Rule Subsets
    ########################################################################
    for i in 1:k
        assignment = k_best_assignments[i]  # One of the k-best assignments
        rule_subset = []  # Stores active rules
        
        for (idx, value) in enumerate(assignment[1])  # assignment[1] contains variable states
            if value == 1  # Only include active variables
                push!(rule_subset, active_rules[idx])   # Map idx to the actual rule ID from active_rules
            end
        end
        
        result.RuleSubset[i] = rule_subset
        result.RuleCount[i] = length(rule_subset)  # Compute RuleCount
        result.LogLikelihood[i] = assignment[2]  # assignment[2] contains log-likelihood
        result.Probability[i] = exp(assignment[2])  # Convert log-likelihood to probability
    end

    ########################################################################
    # Step - 4: Sort by Probability in Descending Order
    ########################################################################
    sort!(result, :Probability, rev=true)

    return result  # Return the final DataFrame containing k-best rule subsets
end


# %% [markdown]
# # Get MAR

# %%
function get_MAR(pc, active_rules, original_ruleset)
    ###############################################
    # Step - 0: Extract active rules from data
    ###############################################
    n = length(active_rules)
    println("Number of active rules: ", n)

    ###############################################
    # Step - 1: Load rule_original and filter
    ###############################################
    rule_original = original_ruleset[in.(original_ruleset.Rule, Ref(active_rules)), :]

    ###############################################
    # Step - 2: Prepare Dummy Dataset
    ###############################################
    data_new = DataFrame([i == j ? true : missing for i in 1:n, j in 1:n], :auto)
    for col in names(data_new)
        data_new[!, col] = convert(Vector{Any}, data_new[!, col])
    end

    ###############################################
    # Step - 3: Get Marginal
    ###############################################
    result = DataFrame(Rule = active_rules, MarLL = MAR(pc, data_new))
    result.Probability = exp.(result.MarLL)

    ###############################################
    # Step - 4: Join and compute errors
    ###############################################
    mar_and_orig = innerjoin(rule_original, result, on = :Rule => :Rule)
    rename!(mar_and_orig, :Rule => :Rules)

    mar_and_orig.StdErr = mar_and_orig.Confidence .- mar_and_orig.Probability
    mar_and_orig.AbsStdErr = abs.(mar_and_orig.StdErr)

    return mar_and_orig
end


# %% [markdown]
# # Main Caller Function 

# %%
function get_ruleprogram_subsets(source, filetype, pctype, em_iteration, confidence_percentage)
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
                    bm_df = prepare_active_rules(s, ft, pt, confidence_percentage)
                    active_rules = string.(bm_df[:, 1])  
                    
                    ###################################
                    # Get kMAP 
                    ###################################
                    println("Getting rule program subsets with kmap...")
                    program_subsets_with_prob = get_rule_program_subsets_with_probabilities(pc, data, active_rules, k)

                    # Store the rule file 
                    rule_file = "ruleset_em_evi_" * string(s) * "_" * string(ft) * "_" * string(pt)
                    rule_file = rule_file * "_" * string(emit) * "_" * string(confidence_percentage) * ".csv"
                    rule_dir = "../../Data-AnyBURL/4-Rules/4-2-3-Rules-EVI/RuleSubsetsWithProb/kmap_pyclause/" 
                    rule_loc = rule_dir * rule_file 
                    CSV.write(rule_loc, program_subsets_with_prob)

                    ###################################
                    # Get MAR 
                    ###################################
                    println("Getting marginals...")
                    mar_file = "ruleset_em_mar_" * string(s) * "_" * string(ft) * "_" * string(pt)
                    mar_file = mar_file * "_" * string(emit) * "_" * string(confidence_percentage) * ".csv"
                    mar_dir = "../../Data-AnyBURL/4-Rules/4-2-3-Rules-EVI/Marginals/kmap_pyclause/" 
                    mar_loc = mar_dir * mar_file 
                    original_ruleset_path = "../../Data-AnyBURL/1-OriginalDataset/$(s)/$(lowercase(s))_10_$(confidence_threshold)_10.csv"
                    println(original_ruleset_path)
                    original_ruleset = CSV.read(original_ruleset_path, DataFrame)
                    current_mar = get_MAR(pc, active_rules, original_ruleset)

                    # Store the MAR 
                    CSV.write(mar_loc, current_mar)
                end
            end 
        end
    end
end; 

# %%
get_ruleprogram_subsets(source, filetype, pctype, em_iteration, confidence_percentage)

# %% [markdown]
# # END 
