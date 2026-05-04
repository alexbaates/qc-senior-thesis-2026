import argparse
import json
import os
import sys
import pandas as pd
import traceback
import h5py
import qiskit 
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import CplexOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo
from qaoa import run_qaoa
from qrao import run_qrao
from cvar_qaoa import run_cvar_qaoa
from cvar_vqe import run_cvar_vqe
from vqe import run_vqe
from ma_qaoa import run_ma_qaoa
from ws_qaoa import run_ws_qaoa

ALGORITHMS = ["qaoa", "cvar_qaoa", "ma_qaoa", "ws_qaoa", "vqe", "cvar_vqe", "qrao", "cplex"]

"""
Quadratic Program Generation
"""

# Given the nodes, edges, and weights of a BIG and the penalty value of the corresponding docking
# problem, generates and returns a quadratic program.
def generate_quadratic_program(name, nodes, edges, weights, penalty):
    quadratic_program = QuadraticProgram(f"{name}")
    N = len(nodes)
    for i in range(N):
        quadratic_program.binary_var(name=f"x{i + 1}")
    linear = {f"x{i + 1}": weights[i] for i in range(N)}
    
    # Build an undirected edge set, so (i, j) and (j, i) are treated identically
    edges_set = {tuple(sorted((int(edge[0]), int(edge[1])))) for edge in edges}
    
    quadratic = {}
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if tuple(sorted((i, j))) not in edges_set:
                quadratic[(f"x{i + 1}", f"x{j + 1}")] = -penalty
    quadratic_program.maximize(linear=linear, quadratic=quadratic)
    return quadratic_program

"""
Run and Save CPLEX Results
"""

# Converts a quadratic program to a QUBO, then runs CPLEX and returns the result.
def run_cplex(quadratic_program):
    print(f"Running CPLEX for {quadratic_program.name}...")
    print()
    # Conert the quadratcim program to a QUBO
    converter = QuadraticProgramToQubo()
    qubo = converter.convert(quadratic_program)
    print(f"QUBO:")
    print(qubo)
    print()

    # Solve the QUBO using CPLEX
    optimizer = CplexOptimizer(cplex_parameters={"threads": 1})
    result = optimizer.solve(qubo)

    return result

# Save CPLEX result for a docking problem to .h5 file.
def save_cplex_result_h5(h5_file_path, problem_name, cplex_result):
    with h5py.File(h5_file_path, "a") as res_file:
        # Create or retrieve group for all results
        if "results" not in res_file:
            results_group = res_file.create_group("results")
        else:
            results_group = res_file["results"]
        
        # Create or retrieve group for the given problem
        if problem_name not in results_group:
            problem_group = results_group.create_group(problem_name)
        else:
            problem_group = results_group[problem_name]

        # Save CPLEX result (overwrite if already exists)
        if "cplex" in problem_group:
            print(f"Warning: Overwriting existing CPLEX results for problem '{problem_name}' in {h5_file_path}.")
            del problem_group["cplex"]

        cplex_group = problem_group.create_group("cplex")
        cplex_group.create_dataset("solution", data=cplex_result.x)
        cplex_group.attrs["fval"] = cplex_result.fval
        cplex_group.attrs["status"] = cplex_result.status.name
        cplex_group.attrs["time"] = cplex_result.raw_results.solve_details.time

# Saves CPLEX result for a docking problem to a CSV file.
def save_cplex_result_csv(csv_file_path, problem_name, cplex_result):
    headers = ["name", "solve_time", "result", "cost", "status"]
    result_data = {"name": problem_name, "solve_time": cplex_result.raw_results.solve_details.time, "result": cplex_result.x, "cost": cplex_result.fval, "status": cplex_result.status.name}    
    if not os.path.exists(csv_file_path):
        df = pd.DataFrame([result_data], columns=headers)
        df.to_csv(csv_file_path, index=False)
    else:
        df = pd.read_csv(csv_file_path)
        df = pd.concat([df, pd.DataFrame([result_data], columns=headers)], ignore_index=True)
        df.to_csv(csv_file_path, index=False)

"""
Run and Save Quantum Algorithm Results and Circuits
"""

# Save quantum algorithm result for a docking problem to .h5 file.
def save_quantum_algorithm_result_h5(h5_file_path, problem_name, algorithm, distribution_bin, optimized_params, reps, shots, function_evals, optimization_time, circuit_time, solve_time, total_time, alpha=None):
    outcomes = list(distribution_bin.keys())
    probabilities = list(distribution_bin.values())

    # Save QAOA result to .h5 file
    with h5py.File(h5_file_path, "a") as res_file:
        # Create or retrieve group for all results
        if "results" not in res_file:
            results_group = res_file.create_group("results")
        else:
            results_group = res_file["results"]
        
        # Create or retrieve group for the given problem
        if problem_name not in results_group:
            problem_group = results_group.create_group(problem_name)
        else:
            problem_group = results_group[problem_name]

        if algorithm not in problem_group:
            algorithm_group = problem_group.create_group(algorithm)
        else:
            algorithm_group = problem_group[algorithm]

        if alpha is None:
            target_group = algorithm_group
        else:
            alpha_group_name = f"alpha_{alpha}"
            if alpha_group_name not in algorithm_group:
                target_group = algorithm_group.create_group(alpha_group_name)
            else:
                target_group = algorithm_group[alpha_group_name]

        if "outcomes" in target_group or "probabilities" in target_group or "parameters" in target_group:
            if alpha is None:
                print(f"Warning: Overwriting existing {algorithm} results for problem '{problem_name}' in {h5_file_path}.")
            else:
                print(f"Warning: Overwriting existing {algorithm}/alpha_{alpha} results for problem '{problem_name}' in {h5_file_path}.")

        if "outcomes" in target_group:
            del target_group["outcomes"]
        if "probabilities" in target_group:
            del target_group["probabilities"]
        if "parameters" in target_group:
            del target_group["parameters"]

        target_group.create_dataset("outcomes", data=outcomes)
        target_group.create_dataset("probabilities", data=probabilities)
        target_group.create_dataset("parameters", data=optimized_params)
        target_group.attrs["shots"] = shots
        target_group.attrs["solve_time"] = solve_time
        target_group.attrs["total_time"] = total_time

        if optimization_time is not None:
            target_group.attrs["optimization_time"] = optimization_time
        if circuit_time is not None:
            target_group.attrs["circuit_time"] = circuit_time
        if alpha is not None:
            target_group.attrs["alpha"] = alpha
        if reps is not None:
            target_group.attrs["reps"] = reps
        if function_evals is not None:
            target_group.attrs["function_evals"] = function_evals

# Save quantum algorithm result for a docking problem to a CSV file.
def save_quantum_algorithm_result_csv(csv_file_path, problem_name, distribution_bin, shots, function_evals, optimization_time, circuit_time, solve_time, total_time, alpha=None):
    # Save top 5 results to CSV file
    distribution_bin_sorted = dict(sorted(distribution_bin.items(), key=lambda x: x[1], reverse=True))
    top_5_outcomes = list(distribution_bin_sorted.keys())[:5]
    top_5_probabilities = list(distribution_bin_sorted.values())[:5]

    headers = []
    results_data = {"name": problem_name, "shots": shots, "function_evals": function_evals, "total_time": total_time}

    if alpha is not None:
        if optimization_time is not None and circuit_time is not None:
            results_data["optimization_time"] = optimization_time
            results_data["circuit_time"] = circuit_time
            results_data["solve_time"] = solve_time
            results_data["alpha"] = alpha
            headers = ["name", "alpha", "shots", "function_evals", "optimization_time", "circuit_time", "solve_time", "total_time", 
                        "top_1_outcome", "top_1_probability", 
                        "top_2_outcome", "top_2_probability", 
                        "top_3_outcome", "top_3_probability", 
                        "top_4_outcome", "top_4_probability", 
                        "top_5_outcome", "top_5_probability"]
        else:
            results_data["solve_time"] = solve_time
            results_data["alpha"] = alpha
            headers = ["name", "alpha", "shots", "function_evals", "solve_time", "total_time", "top_1_outcome", "top_1_probability", 
                    "top_2_outcome", "top_2_probability", 
                    "top_3_outcome", "top_3_probability", 
                    "top_4_outcome", "top_4_probability", 
                    "top_5_outcome", "top_5_probability"]
    else:
        if optimization_time is not None and circuit_time is not None:
            results_data["optimization_time"] = optimization_time
            results_data["circuit_time"] = circuit_time
            results_data["solve_time"] = solve_time
            headers = ["name", "shots", "function_evals", "optimization_time", "circuit_time", "solve_time", "total_time", 
                        "top_1_outcome", "top_1_probability", 
                        "top_2_outcome", "top_2_probability", 
                        "top_3_outcome", "top_3_probability", 
                        "top_4_outcome", "top_4_probability", 
                        "top_5_outcome", "top_5_probability"]
        else:
            results_data["solve_time"] = solve_time
            headers = ["name", "shots", "function_evals", "solve_time", "total_time", "top_1_outcome", "top_1_probability", 
                    "top_2_outcome", "top_2_probability", 
                    "top_3_outcome", "top_3_probability", 
                    "top_4_outcome", "top_4_probability", 
                    "top_5_outcome", "top_5_probability"]
                

    for i in range(len(top_5_outcomes)):
        # Ensure bitstring is stored as a clean string (not a list)
        bitstring = top_5_outcomes[i]
        if isinstance(bitstring, list):
            bitstring = ''.join(map(str, bitstring))
        results_data[f"top_{i+1}_outcome"] = bitstring
        results_data[f"top_{i+1}_probability"] = f"{top_5_probabilities[i]:.6f}"
    
    if not os.path.exists(csv_file_path):
        df = pd.DataFrame([results_data], columns=headers)
        df.to_csv(csv_file_path, index=False)
    else:
        df = pd.read_csv(csv_file_path, dtype=str)
        df = pd.concat([df, pd.DataFrame([results_data], columns=headers)], ignore_index=True)
        df.to_csv(csv_file_path, index=False)


"""
Running Algorithms
"""

# Runs a given algorithm on a quadratic program for a docking problem, then then saves the 
# results of each in the .h5 results file.
def run_algorithm(prob_file_path, h5_file_path, csv_file_path, circuits_dir, algorithm, reps, limit=None, shots=10000, param_max_iter=100000000, param_maxfev=1000):
    print(f"Qiskit Version: {qiskit.__version__}")
    print(f"Running algorithm: {algorithm}")
    print()

    # Extract quadratic programs from problem file
    with h5py.File(prob_file_path, "r") as prob_file:
        print("Retrieving quadratic programs...")
        print()
        docking_problems_group = prob_file["/docking_problems"]
        problem_count = 0
        for docking_problem_name in docking_problems_group.keys():
            if limit is not None and problem_count >= limit:
                print(f"Reached problem limit of {limit}. Stopping.")
                print()
                break
            problem_count += 1
            docking_problem_group = docking_problems_group[docking_problem_name]

            # Extract quadratic program data
            problem_name = docking_problem_group.attrs["name"]
            penalty = docking_problem_group.attrs["penalty"]
            nodes_json = docking_problem_group["nodes"].asstr()[()]
            nodes = json.loads(nodes_json)
            edges = docking_problem_group["edges"][:]
            weights = docking_problem_group["weights"][:]

            # Construct the quadratic program
            quadratic_program = generate_quadratic_program(problem_name, nodes, edges, weights, penalty)

            # Run algorithm on the quadratic program
            distribution_bin = None
            optimized_params = None
            function_evals = None
            optimization_time = None
            circuit_time = None
            solve_time = None
            total_time = None

            if algorithm == "cplex":
                result = run_cplex(quadratic_program)
                save_cplex_result_h5(h5_file_path, problem_name, result)
                save_cplex_result_csv(csv_file_path, problem_name, result)

            elif algorithm !="cvar_qaoa"  and algorithm != "cvar_vqe":
                if algorithm == "qaoa":
                    distribution_bin, optimized_params, shots, function_evals, optimization_time, circuit_time, total_time = run_qaoa(circuits_dir, quadratic_program, reps, shots, param_max_iter, param_maxfev)

                elif algorithm == "qrao":
                    distribution_bin, optimized_params, shots, function_evals, solve_time, total_time = run_qrao(circuits_dir, quadratic_program, reps, shots, param_max_iter, param_maxfev)

                elif algorithm == "vqe":
                    distribution_bin, optimized_params, shots, function_evals, optimization_time, circuit_time, total_time = run_vqe(circuits_dir, quadratic_program, reps, shots, param_max_iter, param_maxfev)

                elif algorithm == "ma_qaoa":
                    distribution_bin, optimized_params, shots, function_evals, optimization_time, circuit_time, total_time = run_ma_qaoa(circuits_dir, quadratic_program, reps, shots, param_max_iter, param_maxfev)

                elif algorithm == "ws_qaoa":
                    distribution_bin, optimized_params, shots, function_evals, optimization_time, circuit_time, total_time = run_ws_qaoa(circuits_dir, quadratic_program, reps, shots, param_max_iter, param_maxfev)

                # Save results to .h5 and CSV files
                if optimization_time is not None and circuit_time is not None:
                    solve_time = optimization_time + circuit_time

                save_quantum_algorithm_result_h5(h5_file_path, problem_name, algorithm, distribution_bin, optimized_params, reps, shots, function_evals, optimization_time, circuit_time, solve_time, total_time, alpha=None)
                save_quantum_algorithm_result_csv(csv_file_path, problem_name, distribution_bin, shots, function_evals, optimization_time, circuit_time, solve_time, total_time, alpha=None)

            else:
                alphas = [0.90, 0.75, 0.50, 0.25]
                for alpha in alphas:
                    if algorithm == "cvar_qaoa":
                        distribution_bin, optimized_params, shots, function_evals, optimization_time, circuit_time, total_time = run_cvar_qaoa(circuits_dir, quadratic_program, alpha, reps, shots, param_max_iter, param_maxfev)

                    elif algorithm == "cvar_vqe":
                        distribution_bin, optimized_params, shots, function_evals, optimization_time, circuit_time, total_time = run_cvar_vqe(circuits_dir, quadratic_program, alpha, reps, shots, param_max_iter, param_maxfev)

                    else:
                        raise Exception(f"Invalid algorithm '{algorithm}'. Valid options are: {', '.join(ALGORITHMS)}.")
                    
                    solve_time = optimization_time + circuit_time
                    save_quantum_algorithm_result_h5(h5_file_path, problem_name, algorithm, distribution_bin, optimized_params, reps, shots, function_evals, optimization_time, circuit_time, solve_time, total_time, alpha)
                    save_quantum_algorithm_result_csv(csv_file_path, problem_name, distribution_bin, shots, function_evals, optimization_time, circuit_time, solve_time, total_time, alpha)

"""
Main Script
"""

def main():
    print("\n------------ HUMAN-DESGINED ALGORITHMS TESTING AND CIRCUIT GENERATION ------------\n")
    try:
        parser = argparse.ArgumentParser(description="Run quantum optimization algorithms.")
        parser.add_argument("--probfile", type=str, required=True, help="Path to the .h5 problems file.")
        parser.add_argument("--h5", type=str, required=True, help="Path to the .h5 results file for the algorithm.")
        parser.add_argument("--csv", type=str, required=True, help="Path to the CSV file for saving algorithm results.")
        parser.add_argument("--circuits", type=str, help="Directory path for saving generated quantum circuits in .qpy format.")
        parser.add_argument("--algorithm", type=str, required=True, help="The algorithm to run (e.g., 'cvar_vqe').")
        parser.add_argument("--reps", type=int, help="Number of repetition layers for quantum algorithms.")
        parser.add_argument("--limit", type=int, help="Maximum number of problems to process from the probfile (default: all).")
        parser.add_argument("--shots", type=int, default=10000, help="Number of shots to use when running quantum algorithms.")
        parser.add_argument("--param_max_iter", type=int, default=100000000, help="Maximum number of iterations for circuit parameter optimizer (default: 100000000).")
        parser.add_argument("--param_maxfev", type=int, default=1000, help="Maximum number of function evaluations for circuit parameter optimizer (default: 1000).")
        args = parser.parse_args()

        PROB_FILE_PATH = args.probfile   # relative path to store generated docking problems and results (must be .h5 and parent directory must exist)
        ALGORITHM = args.algorithm  # algorithm to run
        H5_FILE_PATH = args.h5   # relative path to store algorithm results in .h5 format (must be .h5 and parent directory must exist)
        CSV_FILE_PATH = args.csv # relative path to store algorithm results in CSV format (must be .csv and parent directory must exist)
        CIRCUITS_DIR = args.circuits  # relative path to store generated quantum circuits in .qpy format
        REPS = args.reps  # number of repetition layers for quantum algorithms
        LIMIT = args.limit  # max number of problems to process (None = all)
        SHOTS = args.shots  # number of shots to use when running quantum algorithms
        PARAM_MAX_ITER = args.param_max_iter  # max iterations for circuit parameter optimizer
        PARAM_MAXFEV = args.param_maxfev  # max function evaluations for circuit parameter optimizer
        h5_file_dir = os.path.dirname(H5_FILE_PATH) or "."
        csv_file_dir = os.path.dirname(CSV_FILE_PATH) or "."

        if PROB_FILE_PATH is None or ALGORITHM is None or H5_FILE_PATH is None or CSV_FILE_PATH is None:
            raise Exception("All arguments --probfile, --h5, --csv, and --algorithm must be provided.")
        elif not os.path.isfile(PROB_FILE_PATH):
            raise Exception(f"The problem file '{PROB_FILE_PATH}' does not exist.")
        elif not H5_FILE_PATH.endswith(".h5"):
            raise Exception("The --h5 argument must be a path to a .h5 file.")
        elif not CSV_FILE_PATH.endswith(".csv"):
            raise Exception("The --csv argument must be a path to a .csv file.")
        elif os.path.exists(CSV_FILE_PATH):
            raise Exception(f"The CSV file '{CSV_FILE_PATH}' already exists. Please provide a new file path or delete the existing CSV file.")
        elif not os.path.isdir(h5_file_dir):
            raise Exception("Provided directory for saving .h5 results does not exist.")
        elif not os.path.isdir(csv_file_dir):
            raise Exception("Provided directory for saving CSV results does not exist.")
        elif ALGORITHM not in ALGORITHMS:
            raise Exception(f"Invalid algorithm '{ALGORITHM}'. Valid options are: {', '.join(ALGORITHMS)}.")
        elif ALGORITHM != "cplex" and CIRCUITS_DIR is None:
            raise Exception("The --circuits argument must be provided when running quantum algorithms.")
        elif ALGORITHM != "cplex" and not os.path.isdir(CIRCUITS_DIR):
            raise Exception("Provided directory for saving quantum circuits does not exist.")
        else:
            print(f"Quadratic programs stored in the following file will be used: {PROB_FILE_PATH}")
            print()
            run_algorithm(PROB_FILE_PATH, H5_FILE_PATH, CSV_FILE_PATH, CIRCUITS_DIR, ALGORITHM, REPS, LIMIT, shots=SHOTS, param_max_iter=PARAM_MAX_ITER, param_maxfev=PARAM_MAXFEV)
            print("Algorithm run complete.")
            print()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
