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
# # Creates PC files

# %% is_executing=true
using Pkg                # Use the Pkg module
Pkg.activate("Project.toml")       # For defining environment, . = current directory, .. = immediate previous directory, "path" = some other directory
#Pkg.instantiate()        # Installs and Precompiles all dependencies (We have installed Random already)

# Force consistent numerical environment
using LinearAlgebra
#LinearAlgebra.BLAS.set_num_threads(8)  # Single-threaded BLAS
#ENV["JULIA_NUM_THREADS"] = "1"         # Single-threaded Julia
#ENV["OMP_NUM_THREADS"] = "1"           # Single-threaded OpenMP
#ENV["MKL_NUM_THREADS"] = "1"           # Single-threaded MKL

# Set consistent floating point behavior
#Base.Threads.@threads for i in 1:Base.Threads.nthreads()
#    nothing  # Initialize thread pool consistently
#end


#Pkg.develop(path="../../ProbabilisticCircuits.jl-0.3.3")
#Pkg.precompile()
# %%
import Pkg;
Pkg.add("CUDA")
# %%
using Serialization

using CSV
using DataFrames
using ProbabilisticCircuits
using OrderedCollections
using CUDA
# %% [markdown]
# # Tunable Parameter
#####
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

# %%
THRESHOLD = 0.5 # Can be whatever really, since our values are either blank or 1.

# %% [markdown]
# # Get Validation Data and Preprocess
# - We have two data sources: Missing Data and Conditional Probability Imputed Data. Load both.
# - Convert both dataset to Boolean data. Take 0.5 as Threshold.

# %%
function get_source_data(source, filetype, pctype, confidence_percentage)
    if pctype === "missing"
        filepath = "/home/jpatil01/PC-AnyBURL-Knowledge-Graph-Completion/Data-AnyBURL/3-POST-Data-PC/3-3-2-2-BM_PyClause/"
        # filepath = filepath * string(source) * "_bm_int_" * string(filetype) * ".csv"
        filepath = filepath * string(source) * "_bm_int_" * string(filetype) * "_" * string(confidence_percentage) * ".csv"
    elseif pctype === "imputed" # We don't even need this
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

# %% [markdown]
# ### Change threshold here

# %%
function transpose_and_convert_to_boolean(data)
    # Transpose the DataFrame
    data_transposed = DataFrame(permutedims(Matrix(data)), :auto)

    # Convert Float64 to Bool (using a threshold of 0.5)
    for col in names(data_transposed)
        data_transposed[!, col] = data_transposed[!, col] .> THRESHOLD
    end

    return data_transposed
end;

# %% [markdown]
# # Learn PC using EM

function learn_pc_using_builtin_function(data, pctype::String, number_of_iteration::Int)
    println("Numerical environment check:")
    println("  BLAS threads: ", LinearAlgebra.BLAS.get_num_threads())
    println("  Julia threads: ", Threads.nthreads())
    println("  Data type: ", eltype(data))
    # Learn circuit
    if pctype === "missing"
        pc = learn_circuit_miss(data; maxiter = number_of_iteration, verbose = true)
    elseif pctype === "imputed"
        pc = learn_circuit(data; maxiter = number_of_iteration, verbose = false)
    else
        error("Invalid pctype: $pctype. Use 'missing' or 'imputed'.")
    end

    return pc
end

function get_pc(source, filetype, pctype, em_iteration, confidence_percentage)
    for s in source
        for ft in filetype
            for pt in pctype
                for emit in em_iteration
                    println(s, " ", ft, " ", pt, " ", emit, " ", confidence_percentage)

                    start_time = time()

                    data = get_source_data(s, ft, pt, confidence_percentage)
                    println("Time for get_source_data: $(time() - start_time) seconds")

                    start_time = time()
                    data_tr = transpose_and_convert_to_boolean(data)
                    println("Time for transpose_and_convert_to_boolean: $(time() - start_time) seconds")


                    println("Data reading done...")
                    # CSV.write("bm.csv", data_tr)

                    # Get the PC
                    start_time = time()
                    pc = learn_pc_using_builtin_function(data_tr, pt, emit)
                    ("Time for learn_pc_using_builtin_function: $(time() - start_time) seconds")

                    println("PC is learned...")

                    # Store the PC
                    filename_pc = "PC_EM_" * string(s) * "_" * string(ft) * "_" * string(pt)
                    filename_pc = filename_pc * "_" * string(emit) * "_" * string(confidence_percentage) * "_pyclause.jls"
                    pc_dir = "/home/jpatil01/PC-AnyBURL-Knowledge-Graph-Completion/Data-AnyBURL/5-PC/"
                    pc_loc = pc_dir * filename_pc

                    start_time = time()
                    open(pc_loc, "w") do io
                        serialize(io, pc)
                    end
                    println("Time for serializing and saving PC: $(time() - start_time) seconds")
                end
            end
        end
    end
end;

# %%
get_pc(source, filetype, pctype, em_iteration, confidence_percentage)

# %% [markdown]
# # END

# %%
