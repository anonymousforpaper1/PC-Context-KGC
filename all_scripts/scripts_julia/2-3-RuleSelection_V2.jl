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

# %%
using Pkg                # Use the Pkg module 
Pkg.activate("Project.toml")       # For defining environment, . = current directory, .. = immediate previous directory, "path" = some other directory
# Pkg.instantiate()        # Installs and Precompiles all dependencies (We have installed Random already)

# %%
using Serialization

using CSV
using DataFrames
using OrderedCollections
using ProbabilisticCircuits
#using CUDA
# %% [markdown]
# # Tunable Parameters 

# %%
DATASETS = ["UML", "WN18RR", "FB15-237", "Kinship", "Family", "JF17K", "Nations"]
source = [DATASETS[5]]

# %%
# filetype = ["valid", "test"];
filetype = "train";

# %%
# pctype = ["imputed"];
pctype = "missing";

# %%
em_iteration = 100;

# %%
confidence_percentage = 0
confidence_threshold = confidence_percentage/100

# %%
cutoff_prob_threshold = 0.5;    # May need to adjust this. 0.7 created a HUGE subdirectory.



# %% [markdown]
# # Cond Prob from MAR

# %%
function get_intermediate_cond_prob_table(mar_prob_table, selected_rules_from_prev_walks, pc, cutoff_prob_threshold, storage_info)
    # Pick the first rule based on marginal probability of the remaining rules
    filtered_mar_prob_table = filter(row -> !(row.Rules in selected_rules_from_prev_walks), mar_prob_table)
    sorted_prob_table= sort(filtered_mar_prob_table, :Probability, rev=true)
    best_rule = sorted_prob_table.Rules[1]
    best_ll = sorted_prob_table.MarLL[1]
    best_prob = sorted_prob_table.Probability[1]
    last_selected_prob = best_prob

    ####################################################
    # Preparing the resultant dataframe
    ####################################################
    cond_prob_table = DataFrame(RuleSets=Vector{Vector{String}}[],
                            NumRules=Int[],
                            LogLikelihood=Union{Missing, Float64}[],
                            Probability=Union{Missing, Float64}[])

    # Add the first row (best marginal probability row)
    selected_rules_here = String[]  # Empty list
    push!(selected_rules_here, best_rule)
    if last_selected_prob >= cutoff_prob_threshold
        push!(cond_prob_table, (copy(selected_rules_here), length(selected_rules_here), best_ll, best_prob), promote=true)
    end

    ####################################################
    # Now create rest of the rows
    ####################################################
    # Extract and sort all rules in natural order
    all_possible_rules = sort(mar_prob_table.Rules, by = x -> parse(Int, match(r"\d+", x).match))
    all_rules_after_prev_walks = sort(filtered_mar_prob_table.Rules, by = x -> parse(Int, match(r"\d+", x).match))
    flag = true     # Keeps track if there are no more rules left to add
    sub_walk_no = 1

    while (last_selected_prob >= cutoff_prob_threshold)
        # remaining_rules_after_prev_walk = setdiff(all_possible_rules, selected_rules_from_prev_walks)
        remaining_rules = setdiff(all_rules_after_prev_walks, selected_rules_here)  # Get unselected rules

        if isempty(remaining_rules)
            flag = false
            println("No Rules Left.")
            break  # Exit if no more rules left
        end


        # Create an empty dataframe
        # Here, Symbol(r) => Union{Missing, Bool}[] means: Symbol(r) becomes column names
        # Union{Missing, Bool}[] is for values, so values can be either boolean or missing
        df = DataFrame([Symbol(r) => Union{Missing, Bool}[] for r in all_possible_rules]...)

        # Create rows where selected_rules_here are always TRUE and one additional rule is also TRUE
        for r in remaining_rules
            # If we want a square matrix, many rows will only have missing value. Also the row with the best marginal have to be all missing.
            # row = [rule in selected_rules_here || (rule == r && !(r in selected_rules_from_prev_walks)) ? true : missing for rule in all_possible_rules]
            row = [rule in selected_rules_here || rule == r ? true : missing for rule in all_possible_rules]
            push!(df, row)
        end

        # Run the MAR Query and get the best Probability column
        if nrow(df) == 0
            println("Empty dataframe detected before MAR call.")
            break
        end
        if isempty(df)
            println("DataFrame is completely empty. Exiting loop.")
            break
        end
        marLL_values = MAR(pc, df)
        # marLL_values = replace(marLL_values, NaN => -600, Inf => 0, -Inf => -600)
        # Reason for chosing 700: exp(700) ≈ 1.01 × 10^304 (which is near the maximum representable Float64 value).
        # Reason for chosing 0: exp(0) = 1, which should be the maximum probability

        if any(isnan, marLL_values)
            println("Some values in MAR(pc, df) are NaN. Exiting the loop.")
            break
        end
        if any(isinf, marLL_values)
            println("Some values in MAR(pc, df) are Inf. Exiting the loop.")
            break
        end
        df.marLL = marLL_values
        # df.marLL = clamp.(marLL_values, -600, 0)      # No clamping should be needed when calculation is correct
        df.Probability = exp.(df.marLL)
        sorted_df = sort(df, :Probability, rev=true)

        # Get the highest probability row (first row)
        top_row = sorted_df[1, :]

        # Identify the rule that has `true` apart from already selected rules
        for rule in all_rules_after_prev_walks
            if rule ∉ selected_rules_here && !ismissing(top_row[rule]) && top_row[rule] == true
                push!(selected_rules_here, rule)
                last_selected_prob = top_row[:Probability]
                if last_selected_prob >= cutoff_prob_threshold
                    push!(cond_prob_table, (copy(selected_rules_here), length(selected_rules_here), top_row.marLL, top_row.Probability), promote=true)
                end
                break
            end
        end

        # Store all subwalks (optional, will delete later)
        # subwalk_dfs_dir = "../../Data-AnyBURL/4-Rules/4-2-3-Rules-EVI/RuleSubsets/" * string(cutoff_prob_threshold) * "/SubwalkDFs/"
        # mkpath(subwalk_dfs_dir)
        # println("Making subwalk directory: $subwalk_dfs_dir")
        # # source, support, confidence, walk_no = storage_info
        # dataset, walk_no = storage_info

        # # subwalk_dfs_subdirectory = subwalk_dfs_directory * string(source) * "_" * string(support) * "_" * string(confidence) * "/"
        # subwalk_dfs_subdir = subwalk_dfs_dir * string(dataset) * "/"
        # mkpath(subwalk_dfs_subdir)
        # subwalk_file = string(walk_no) * "_" * string(sub_walk_no) * ".csv"
        # subwalk_loc = subwalk_dfs_subdir * subwalk_file
        # println("Saving subwalk to: $subwalk_loc")
        # CSV.write(subwalk_loc, sorted_df)
        # sub_walk_no += 1
    end
    if flag
        return cond_prob_table
    else
        return DataFrame(RuleSets=Vector{Vector{String}}[],
                        NumRules=Int[],
                        LogLikelihood=Union{Missing, Float64}[],
                        Probability=Union{Missing, Float64}[])
    end
end;

# %%
function get_cond_prob_table(mar_prob_table, pc, cutoff_prob_threshold, dataset)
    # Initialize an empty array to store results
    all_results = Vector{DataFrame}()

    # Track selected rules in a set for faster lookups
    selected_rules_from_all_greedy_walk = Vector{String}()

    # Continue calling the function until an empty DataFrame is returned
    walk_no = 1
    while true
        storage_info = (dataset, walk_no)

        intermediate_df = get_intermediate_cond_prob_table(
            mar_prob_table,
            selected_rules_from_all_greedy_walk,
            pc,
            cutoff_prob_threshold,
            storage_info
        )

        # print("nrow of intermediate df: ", nrow(intermediate_df), "        ")
        walk_no += 1

        # If an empty DataFrame is returned, exit the loop
        if nrow(intermediate_df) == 0
            break
        end

        # Extract the last row's RuleSets (which should be a vector of rules like ["R1", "R10", "R3"])
        last_row = intermediate_df[end, :]
        selected_rules_from_current_greedy_walk = last_row.RuleSets

        # Update the set of selected rules
        append!(selected_rules_from_all_greedy_walk, selected_rules_from_current_greedy_walk)

        # Remove the selected rules from mar_prob_table using a set for faster lookups
        # mar_prob_table = filter(row -> !(row.Rules in selected_rules_main), mar_prob_table)

        # Append the current intermediate_df to all_results (using a vector for memory efficiency)
        push!(all_results, intermediate_df)
    end

    # Combine all DataFrames into one (if needed)
    combined_df = vcat(all_results...)

    # Return the sorted results
    return combined_df
end;

# %%
function filter_condprobtable_by_k(cond_prob_table, k)
    selected_rows = first(cond_prob_table, k)
    return selected_rows
end;

# %%
function filter_condprobtable_by_p(cond_prob_table, p)
    selected_rows = filter(row -> row.Probability >= p, cond_prob_table)
    return selected_rows
end;

# %% [markdown]
# # Main Call

# %%
function main_call(source, confidence_percentage, cutoff_prob_threshold)
    for s in source
        println("Processing source: $s")

        common_name = string(s) * "_" * string(filetype) * "_" * string(pctype)* "_" * string(em_iteration) * "_" * string(confidence_percentage)

        ###################################
        # Get PC
        ###################################
        pc_file = "PC_EM_" * string(s) * "_" * string(filetype) * "_" * string(pctype)
        pc_file = pc_file * "_" * string(em_iteration) * "_" * string(confidence_percentage) * "_pyclause.jls"
        pc_dir = "../../Data-AnyBURL/5-PC/"
        pc_loc = pc_dir * pc_file
        println("Loading PC from: $pc_loc")

        pc = open(pc_loc, "r") do io
            deserialize(io)
        end

        ###################################
        # Get MAR
        ###################################
        mar_file = "ruleset_em_mar_" * string(s) * "_" * string(filetype) * "_" * string(pctype)
        mar_file = mar_file * "_" * string(em_iteration) * "_" * string(confidence_percentage) * ".csv"
        mar_dir = "../../Data-AnyBURL/4-Rules/4-2-3-Rules-EVI/Marginals/kmap_pyclause/"
        mar_loc = mar_dir * mar_file
        println("Loading MAR from: $mar_loc")

        mar = CSV.File(mar_loc) |> DataFrame
        println("Loaded MAR with $(nrow(mar)) rows")

        if nrow(mar) == 0
            println("MAR file is empty for source: $s. Skipping.")
            continue
        end

        ###################################
        # Get Unsorted Conditional Probability Table
        ###################################
        dataset = s # Basically the variables for naming the files
        # cond_prob_table = get_cond_prob_table(mar, pc, first_rule, sorted_mar.MarLL[1], sorted_mar.Probability[1])
        cond_prob_table = get_cond_prob_table(mar, pc, cutoff_prob_threshold, dataset) # Fixed
        cond_prob_table_file = "unsorted_condprob_" * common_name * "_" * string(cutoff_prob_threshold) * ".csv"
        unfiltered_ruleset_dir = "../../Data-AnyBURL/4-Rules/4-2-3-Rules-EVI/AllRules/"
        cond_prob_table_loc = unfiltered_ruleset_dir * cond_prob_table_file
        println("Saving unsorted conditional probability table to: $cond_prob_table_loc")
        CSV.write(cond_prob_table_loc, cond_prob_table)

        ###################################
        # Get Sorted Conditional Probability Table
        ###################################
        sorted_cond_prob_table = sort(cond_prob_table, :Probability, rev=true)
        sorted_cond_prob_table_file = "condprob_" * common_name * "_" * string(cutoff_prob_threshold) * ".csv"
        sorted_cond_prob_table_loc = unfiltered_ruleset_dir * sorted_cond_prob_table_file
        println("Saving sorted conditional probability table to: $sorted_cond_prob_table_loc")
        CSV.write(sorted_cond_prob_table_loc, sorted_cond_prob_table)

        ###################################
        # Filtering by Top-k
        ###################################
        # for k in k_values
        #     println("Filtering by top-k = $k")
        #     filtered_cpt = filter_condprobtable_by_k(sorted_cond_prob_table, k) # Using the sorted cond table
        #     filtered_ruleset_file = "filtered_ruleset_" * common_name * "_k_" * string(k) * ".csv"
        #     filtered_ruleset_dir = "../../Data-AnyBURL/4-Rules/4-2-3-Rules-EVI/FilteredRules/k_selection/"
        #     filtered_ruleset_loc = filtered_ruleset_dir * filtered_ruleset_file
        #     println("Writing filtered ruleset (K) to: $filtered_ruleset_loc")
        #     CSV.write(filtered_ruleset_loc, filtered_cpt)
        # end

        ##################################
        # Filtering by Probability Threshold
        ###################################
        # for p in p_values
        #     println("Filtering by probability threshold = $p")
        #     filtered_cpt = filter_condprobtable_by_p(sorted_cond_prob_table, p) # Using the sorted cond table
        #     filtered_ruleset_file = "filtered_ruleset_" * common_name * "_p_" * string(p) * ".csv"
        #     filtered_ruleset_dir = "../../Data-AnyBURL/4-Rules/4-2-3-Rules-EVI/FilteredRules/prob_selection/"
        #     filtered_ruleset_loc = filtered_ruleset_dir * filtered_ruleset_file
        #     println("Writing filtered ruleset (P) to: $filtered_ruleset_loc")
        #     CSV.write(filtered_ruleset_loc, filtered_cpt)
        # end

        println("Finished processing source: $s")
        println("-"^80)
    end
end

# %%
# main_call(source, rule_supports, rule_confidences, k_values, p_values, cutoff_prob_threshold)
main_call(source, confidence_percentage, cutoff_prob_threshold)

# %% [markdown]
# # END
