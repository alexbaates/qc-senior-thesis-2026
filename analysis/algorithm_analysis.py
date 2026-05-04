import argparse
import csv
import datetime
import json
import os
import re
import uuid
from collections import defaultdict
import h5py
import numpy as np
from qiskit.quantum_info import Statevector
from qiskit import qpy
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

"""
Quadratic Program and QUBO helpers
"""
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

def quadratic_program_to_qubo(quadratic_program):
    converter = QuadraticProgramToQubo()
    qubo = converter.convert(quadratic_program)
    return qubo

def qubo_to_ising(qubo, offset=0.0):
    qubit_op, qubo_offset = qubo.to_ising()
    return qubit_op, qubo_offset + offset


"""
H5 loading helpers
"""
def load_problems_from_h5(h5_path):
    problems = []
    with h5py.File(h5_path, "r") as f:
        docking_problems_group = f["/docking_problems"]
        for problem_key in docking_problems_group.keys():
            group = docking_problems_group[problem_key]
            problem_name = group.attrs["name"]
            penalty = group.attrs["penalty"]
            nodes_json = group["nodes"].asstr()[()]
            nodes = json.loads(nodes_json)
            edges = group["edges"][:]
            weights = group["weights"][:]
            quadratic_program = generate_quadratic_program(problem_name, nodes, edges, weights, penalty)
            problems.append((problem_name, quadratic_program))
    return problems

"""
Circuit helpers
"""
def load_circuit(qpy_path):
    with open(qpy_path, "rb") as f:
        circuits = qpy.load(f)
    return circuits[0]

"""
Results helpers
"""
def filename_matches_algorithm(fname, algorithm, reps):
    # Tokenize filename (strip .h5, split on - and _)
    stem = fname[:-3] if fname.endswith(".h5") else fname
    tokens = re.split(r"[-_]", stem)
    alg_tokens = algorithm.split("_")
    # Check algorithm appears as a contiguous token sequence
    n, m = len(tokens), len(alg_tokens)
    alg_match = any(tokens[i:i + m] == alg_tokens for i in range(n - m + 1))
    if not alg_match:
        return False
    # Check reps if provided
    if reps is not None:
        return f"reps{reps}" in tokens
    return True

def get_results(resdir, problem_name, algorithm, reps=None):
    h5_files = sorted(
        os.path.join(resdir, f)
        for f in os.listdir(resdir)
        if f.endswith(".h5") and filename_matches_algorithm(f, algorithm, reps)
    )
    if not h5_files:
        reps_str = f" reps={reps}" if reps is not None else ""
        print(f"Warning: No .h5 results file found for algorithm '{algorithm}'{reps_str} in '{resdir}'.")
        return
    for h5_path in h5_files:
        with h5py.File(h5_path, "r") as f:
            if "results" not in f:
                continue
            if problem_name not in f["results"]:
                continue
            print(f"Results found for {algorithm} ran on {problem_name}")
            if algorithm == "cplex":
                cplex_group = f["results"][problem_name]["cplex"]
                data = {}
                for key in cplex_group.keys():
                    data[key] = cplex_group[key][()]
                for attr_key, attr_val in cplex_group.attrs.items():
                    data[attr_key] = attr_val
                return data
            return
    reps_str = f" reps={reps}" if reps is not None else ""
    print(f"Warning: No results found for problem '{problem_name}' with algorithm '{algorithm}'{reps_str} in '{resdir}'.")

"""
Expected probability amplitude calculation
"""

def reorder_cplex_solution(solution):
    # CPLEX solution is in reverse variable order compared to Qiskit's qubit ordering
    # return solution[::-1]
    return solution

def ground_state_vector_to_bitarray(vector, tol=1e-6):
    amplitudes = np.abs(vector)
    max_amp = np.max(amplitudes)
    if max_amp < 1.0 - tol:
        raise ValueError(
            f"Ground state vector is not a pure basis state: max amplitude is {max_amp:.6f}, "
            "expected ~1.0. The Ising Hamiltonian may not be diagonal."
        )
    index = int(np.argmax(amplitudes))
    n_qubits = int(np.log2(len(vector)))
    return np.array([(index >> i) & 1 for i in range(n_qubits)], dtype=int)

def get_ground_state_vectors(qubit_op, degeneracy_tol=1e-10):
    matrix = qubit_op.to_matrix()
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    min_eigenvalue = eigenvalues[0]
    ground_indices = np.where(np.abs(eigenvalues - min_eigenvalue) <= degeneracy_tol)[0]
    return eigenvectors[:, ground_indices]

def parse_algorithm_label(algorithm_label):
    # Expected formats:
    #   {algorithm}_reps{N}              e.g. ma_qaoa_reps3
    #   {algorithm}_reps{N}_alpha{V}     e.g. cvar_qaoa_reps3_alpha0.5
    match = re.match(r'^(.+)_reps(\d+)(?:_alpha([\d.]+))?$', algorithm_label)
    if match:
        algorithm = match.group(1)
        reps = int(match.group(2))
        alpha = float(match.group(3)) if match.group(3) is not None else None
        return algorithm, reps, alpha
    return algorithm_label, None, None

def save_success_probabilities(outputdir, rows, algorithm, reps, alpha=None):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_chars = uuid.uuid4().hex[:6]
    alpha_string = f"_alpha{alpha}" if alpha is not None else ""
    filename = f"success_probabilities_{algorithm}_reps{reps}{alpha_string}_{timestamp}_{random_chars}.csv"
    filepath = os.path.join(outputdir, filename)
    if os.path.exists(filepath):
        raise FileExistsError(f"Output file already exists: '{filepath}'")
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["problem_name", "algorithm", "reps", "alpha", "success_probability", "n_ground_states"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved success probabilities to: {filepath}")


def best_objective_value_from_outcomes(quadratic_program, outcomes):
    best = None
    for outcome in outcomes:
        # Outcome bitstring: left-to-right maps to x1, x2, ..., xN
        x = np.array([int(c) for c in outcome], dtype=float)
        val = quadratic_program.objective.evaluate(x)
        if best is None or val > best:
            best = val
    return float(best)

def load_cplex_solution_obj(cplex_h5_paths, problem_name, quadratic_program):
    """Search cplex_h5_paths for a CPLEX solution for problem_name and evaluate against the QP objective."""
    for cplex_h5_path in cplex_h5_paths:
        with h5py.File(cplex_h5_path, "r") as f:
            if "results" not in f or problem_name not in f["results"] or "cplex" not in f["results"][problem_name]:
                continue
            solution = f["results"][problem_name]["cplex"]["solution"][:]
        x = np.array(solution, dtype=float)
        return float(quadratic_program.objective.evaluate(x))
    raise KeyError(f"No CPLEX results found for problem '{problem_name}' in any CPLEX file")

def load_algorithm_outcomes(resdir, problem_name, algorithm, reps, alpha=None):
    h5_files = sorted(
        os.path.join(resdir, f)
        for f in os.listdir(resdir)
        if f.endswith(".h5") and filename_matches_algorithm(f, algorithm, reps)
    )
    if not h5_files:
        raise FileNotFoundError(
            f"No .h5 results file found for algorithm '{algorithm}' reps={reps} in '{resdir}'"
        )
    for h5_path in h5_files:
        with h5py.File(h5_path, "r") as f:
            if "results" not in f or problem_name not in f["results"]:
                continue
            prob_group = f["results"][problem_name]
            if algorithm not in prob_group:
                continue
            alg_group = prob_group[algorithm]
            if alpha is not None:
                alpha_key = f"alpha_{alpha}"
                if alpha_key not in alg_group:
                    continue
                alg_group = alg_group[alpha_key]
            if "outcomes" not in alg_group:
                continue
            return alg_group["outcomes"].asstr()[:]
    raise KeyError(
        f"No outcomes found for problem '{problem_name}' algorithm '{algorithm}' reps={reps}"
        + (f" alpha={alpha}" if alpha is not None else "")
        + f" in '{resdir}'"
    )

def relative_solution_quality(alg_obj_value, cplex_obj_value):
    return (alg_obj_value / cplex_obj_value) * 100.0

def save_relative_solution_quality(outputdir, rows, algorithm, reps, alpha=None):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_chars = uuid.uuid4().hex[:6]
    alpha_string = f"_alpha{alpha}" if alpha is not None else ""
    filename = f"relative_solution_quality_{algorithm}_reps{reps}{alpha_string}_{timestamp}_{random_chars}.csv"
    filepath = os.path.join(outputdir, filename)
    if os.path.exists(filepath):
        raise FileExistsError(f"Output file already exists: '{filepath}'")
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["problem_name", "algorithm", "reps", "alpha", "best_objective_value", "cplex_fval", "relative_solution_quality"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved relative solution quality to: {filepath}")


def parse_rsq_filename(filepath):
    """Parse algorithm, reps, alpha from a [avg_]relative_solution_quality_*.csv filename."""
    basename = os.path.basename(filepath)
    name_to_parse = basename[4:] if basename.startswith("avg_") else basename
    m = re.match(r'^relative_solution_quality_(.+?)_reps(\d+)(?:_alpha([\d.]+))?_\d{8}_\d{6}_[0-9a-f]{6}\.csv$', name_to_parse)
    if not m:
        raise ValueError(
            f"Cannot parse algorithm/reps/alpha from filename: '{basename}'. "
            "Expected format: [avg_]relative_solution_quality_<alg>_reps<N>[_alpha<V>]_<YYYYMMDD>_<HHMMSS>_<hex6>.csv"
        )
    algorithm = m.group(1)
    reps = int(m.group(2))
    alpha = float(m.group(3)) if m.group(3) is not None else None
    return algorithm, reps, alpha

def load_already_computed_problems(csv_path):
    """Read a partial RSQ CSV and return the set of problem_names already present."""
    problems = set()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            problems.add(row["problem_name"])
    return problems

def append_relative_solution_quality(filepath, rows):
    """Append new rows to an existing RSQ CSV (header is not re-written)."""
    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["problem_name", "algorithm", "reps", "alpha", "best_objective_value", "cplex_fval", "relative_solution_quality"])
        writer.writerows(rows)
    print(f"Appended {len(rows)} rows to: {filepath}")


def load_algorithm_outcomes_and_probabilities(resdir, problem_name, algorithm, reps, alpha=None):
    h5_files = sorted(
        os.path.join(resdir, f)
        for f in os.listdir(resdir)
        if f.endswith(".h5") and filename_matches_algorithm(f, algorithm, reps)
    )
    if not h5_files:
        raise FileNotFoundError(
            f"No .h5 results file found for algorithm '{algorithm}' reps={reps} in '{resdir}'"
        )
    for h5_path in h5_files:
        with h5py.File(h5_path, "r") as f:
            if "results" not in f or problem_name not in f["results"]:
                continue
            prob_group = f["results"][problem_name]
            if algorithm not in prob_group:
                continue
            alg_group = prob_group[algorithm]
            if alpha is not None:
                alpha_key = f"alpha_{alpha}"
                if alpha_key not in alg_group:
                    continue
                alg_group = alg_group[alpha_key]
            if "outcomes" not in alg_group or "probabilities" not in alg_group:
                continue
            outcomes = alg_group["outcomes"].asstr()[:]
            probabilities = alg_group["probabilities"][:]
            return outcomes, probabilities
    raise KeyError(
        f"No outcomes/probabilities found for problem '{problem_name}' algorithm '{algorithm}' reps={reps}"
        + (f" alpha={alpha}" if alpha is not None else "")
        + f" in '{resdir}'"
    )


def average_objective_value_from_outcomes(quadratic_program, outcomes, probabilities):
    # Probability-weighted average of objective values, excluding zero-probability outcomes.
    total = 0.0
    total_prob = 0.0
    for outcome, prob in zip(outcomes, probabilities):
        if prob <= 0.0:
            continue
        x = np.array([int(c) for c in outcome], dtype=float)
        val = quadratic_program.objective.evaluate(x)
        total += prob * val
        total_prob += prob
    if total_prob == 0.0:
        raise ValueError("All outcomes have zero probability; cannot compute average objective value.")
    return total / total_prob


def save_avg_relative_solution_quality(outputdir, rows, algorithm, reps, alpha=None):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_chars = uuid.uuid4().hex[:6]
    alpha_string = f"_alpha{alpha}" if alpha is not None else ""
    filename = f"avg_relative_solution_quality_{algorithm}_reps{reps}{alpha_string}_{timestamp}_{random_chars}.csv"
    filepath = os.path.join(outputdir, filename)
    if os.path.exists(filepath):
        raise FileExistsError(f"Output file already exists: '{filepath}'")
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["problem_name", "algorithm", "reps", "alpha", "average_objective_value", "cplex_fval", "relative_solution_quality"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved average relative solution quality to: {filepath}")


def append_avg_relative_solution_quality(filepath, rows):
    # Append new rows to an existing avg RSQ CSV (header is not re-written).
    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["problem_name", "algorithm", "reps", "alpha", "average_objective_value", "cplex_fval", "relative_solution_quality"])
        writer.writerows(rows)
    print(f"Appended {len(rows)} rows to: {filepath}")


def success_probability(qubit_op, circuit):
    circ = circuit.remove_final_measurements(inplace=False)
    psi = Statevector(circ)
    psi_vec = psi.data
    n_circ_qubits = circ.num_qubits
    n_op_qubits = qubit_op.num_qubits
    if n_circ_qubits != n_op_qubits:
        raise ValueError(
            f"Circuit has {n_circ_qubits} qubit(s) but Ising Hamiltonian has {n_op_qubits} qubit(s). "
            "EPA requires matching dimensions — skipping (QRAO circuits are not compatible)."
        )
    ground_vecs = get_ground_state_vectors(qubit_op)
    n_ground_states = ground_vecs.shape[1]
    total_prob = 0.0
    for i in range(n_ground_states):
        g = ground_vecs[:, i]
        total_prob += abs(np.dot(np.conj(g), psi_vec)) ** 2
    return float(total_prob), n_ground_states

def main():
    try:
        parser = argparse.ArgumentParser(description="Analyze algorithm results.")
        parser.add_argument("--probdir", type=str, required=True, help="Directory containing .h5 problem files")
        parser.add_argument("--circdir", type=str, default=None, help="Directory containing .qpy circuit files (required for --metric epa)")
        parser.add_argument("--resdir", type=str, default=None, help="Directory containing algorithm results .h5 files (required for --metric rsq)")
        parser.add_argument("--outputdir", type=str, required=True, help="Directory to save output files")
        parser.add_argument("--resume-from", type=str, default=None, dest="resume_from", help="Path to a partial RSQ CSV to resume; algorithm/reps/alpha are inferred from its filename (only valid with --metric rsq)")
        parser.add_argument("--algorithm", type=str, default=None, choices=["qaoa", "ma_qaoa", "ws_qaoa", "cvar_qaoa", "vqe", "cvar_vqe", "qrao"], help="Algorithm to analyze (required unless --resume-from is set)")
        parser.add_argument("--reps", type=int, default=None, help="Number of repetitions to analyze (required unless --resume-from is set)")
        parser.add_argument("--alpha", type=float, default=None, help="Alpha value to filter on (only for cvar_qaoa and cvar_vqe)")
        parser.add_argument("--metric", type=str, required=True, choices=["epa", "rsq"], help="Metric to compute (currently supported: 'epa', 'rsq')")
        parser.add_argument("--n", type=int, default=None, help="Stop after processing N problems")
        parser.add_argument("--avg", action="store_true", default=False, help="Compute probability-weighted average RSQ instead of best-value RSQ (only valid with --metric rsq)")
        args = parser.parse_args()

        # Check always-required directories
        for arg_name, dir_path in [("--probdir", args.probdir), ("--outputdir", args.outputdir)]:
            if not os.path.isdir(dir_path):
                parser.error(f"{arg_name}: '{dir_path}' is not an existing directory")

        # Metric-specific argument validation
        if args.avg and args.metric != "rsq":
            parser.error("--avg is only valid when --metric is 'rsq'")
        if args.metric == "epa":
            if not args.circdir:
                parser.error("--circdir is required when --metric is 'epa'")
            if not os.path.isdir(args.circdir):
                parser.error(f"--circdir: '{args.circdir}' is not an existing directory")
        if args.metric == "rsq":
            if not args.resdir:
                parser.error("--resdir is required when --metric is 'rsq'")
            if not os.path.isdir(args.resdir):
                parser.error(f"--resdir: '{args.resdir}' is not an existing directory")

        # Resolve effective algorithm/reps/alpha and the set of already-computed problems
        valid_algorithms = ["qaoa", "ma_qaoa", "ws_qaoa", "cvar_qaoa", "vqe", "cvar_vqe", "qrao"]
        already_computed = set()
        if args.resume_from is not None:
            if args.metric != "rsq":
                parser.error("--resume-from is only valid when --metric is 'rsq'")
            if not os.path.isfile(args.resume_from):
                parser.error(f"--resume-from: '{args.resume_from}' does not exist or is not a file")
            effective_algorithm, effective_reps, effective_alpha = parse_rsq_filename(args.resume_from)
            resume_is_avg = os.path.basename(args.resume_from).startswith("avg_")
            if args.avg and not resume_is_avg:
                parser.error("--avg was set but --resume-from points to a non-avg RSQ CSV. Pass the avg_relative_solution_quality_*.csv file to resume an avg run.")
            if not args.avg and resume_is_avg:
                parser.error("--resume-from points to an avg RSQ CSV but --avg was not set. Add --avg to resume this file.")
            if effective_algorithm not in valid_algorithms:
                parser.error(
                    f"Algorithm '{effective_algorithm}' inferred from --resume-from filename is not recognised. "
                    f"Expected one of: {valid_algorithms}"
                )
            if args.algorithm is not None:
                print(f"Warning: --algorithm '{args.algorithm}' ignored; using '{effective_algorithm}' inferred from --resume-from filename.")
            if args.reps is not None:
                print(f"Warning: --reps {args.reps} ignored; using {effective_reps} inferred from --resume-from filename.")
            if args.alpha is not None:
                print(f"Warning: --alpha {args.alpha} ignored; using {effective_alpha} inferred from --resume-from filename.")
            already_computed = load_already_computed_problems(args.resume_from)
            print(
                f"Resuming '{os.path.basename(args.resume_from)}': "
                f"{len(already_computed)} problems already computed "
                f"(algorithm={effective_algorithm}, reps={effective_reps}, alpha={effective_alpha})"
            )
        else:
            if args.algorithm is None:
                parser.error("--algorithm is required when --resume-from is not specified")
            if args.reps is None:
                parser.error("--reps is required when --resume-from is not specified")
            effective_algorithm = args.algorithm
            effective_reps = args.reps
            effective_alpha = args.alpha

        # Find all .h5 problem files in probdir
        h5_prob_files = sorted(
            os.path.join(args.probdir, f)
            for f in os.listdir(args.probdir)
            if f.endswith(".h5")
        )
        if not h5_prob_files:
            raise FileNotFoundError(f"No .h5 files found in --probdir '{args.probdir}'")

        # Discover CPLEX h5 files for RSQ metric
        cplex_h5_paths = []
        if args.metric == "rsq":
            cplex_h5_paths = sorted(
                os.path.join(args.resdir, f)
                for f in os.listdir(args.resdir)
                if f.endswith(".h5") and "cplex" in f.lower()
            )
            if not cplex_h5_paths:
                raise FileNotFoundError(f"No CPLEX .h5 files (containing 'cplex' in name) found in --resdir '{args.resdir}'")
            print(f"Found {len(cplex_h5_paths)} CPLEX file(s): {[os.path.basename(p) for p in cplex_h5_paths]}")

        # Build circuit_map for EPA metric
        circuit_map = {}
        if args.metric == "epa":
            for dirpath, _, filenames in os.walk(args.circdir):
                for fname in filenames:
                    if fname.endswith(".qpy") and fname.startswith("DP12_"):
                        parts = fname[:-4].split("_")
                        if len(parts) >= 3:
                            prob_name = f"{parts[0]}_{parts[1]}"
                            algorithm_label = "_".join(parts[2:])
                            circuit_map.setdefault(prob_name, []).append(
                                (os.path.join(dirpath, fname), algorithm_label)
                            )

        # Process each problem file
        rows = []
        n_processed = 0
        done = False
        for h5_path in h5_prob_files:
            if done:
                break
            print(f"\nProcessing problem file: {h5_path}")
            problems = load_problems_from_h5(h5_path)
            for problem_name, quadratic_program in problems:
                if args.n is not None and n_processed >= args.n:
                    done = True
                    break
                if args.metric == "epa":
                    if problem_name not in circuit_map:
                        print(f"Warning: No .qpy circuits found for problem '{problem_name}' in --circdir '{args.circdir}'. Skipping.")
                        continue
                    qubo = quadratic_program_to_qubo(quadratic_program)
                    qubit_op, _ = qubo_to_ising(qubo)
                    for circuit_path, algorithm_label in sorted(circuit_map[problem_name]):
                        alg_parsed, reps_parsed, alpha_parsed = parse_algorithm_label(algorithm_label)
                        if alg_parsed != effective_algorithm:
                            continue
                        if reps_parsed != effective_reps:
                            continue
                        if effective_alpha is not None and alpha_parsed != effective_alpha:
                            continue
                        circuit = load_circuit(circuit_path)
                        try:
                            sp, n_ground_states = success_probability(qubit_op, circuit)
                        except ValueError as e:
                            print(f"Warning: Skipping '{os.path.basename(circuit_path)}' for problem '{problem_name}': {e}")
                            continue
                        rows.append({
                            "problem_name": problem_name,
                            "algorithm": alg_parsed,
                            "reps": reps_parsed,
                            "alpha": alpha_parsed,
                            "success_probability": sp,
                            "n_ground_states": n_ground_states,
                        })
                    n_processed += 1
                elif args.metric == "rsq":
                    if problem_name in already_computed:
                        continue
                    if args.avg:
                        try:
                            outcomes, probabilities = load_algorithm_outcomes_and_probabilities(args.resdir, problem_name, effective_algorithm, effective_reps, effective_alpha)
                        except (FileNotFoundError, KeyError) as e:
                            print(f"Warning: {e}. Skipping.")
                            continue
                        try:
                            avg_obj = average_objective_value_from_outcomes(quadratic_program, outcomes, probabilities)
                        except ValueError as e:
                            print(f"Warning: {e}. Skipping.")
                            continue
                        try:
                            cplex_obj = load_cplex_solution_obj(cplex_h5_paths, problem_name, quadratic_program)
                        except KeyError as e:
                            print(f"Warning: {e}. Skipping.")
                            continue
                        rsq = relative_solution_quality(avg_obj, cplex_obj)
                        rows.append({
                            "problem_name": problem_name,
                            "algorithm": effective_algorithm,
                            "reps": effective_reps,
                            "alpha": effective_alpha,
                            "average_objective_value": avg_obj,
                            "cplex_fval": cplex_obj,
                            "relative_solution_quality": rsq,
                        })
                    else:
                        try:
                            outcomes = load_algorithm_outcomes(args.resdir, problem_name, effective_algorithm, effective_reps, effective_alpha)
                        except (FileNotFoundError, KeyError) as e:
                            print(f"Warning: {e}. Skipping.")
                            continue
                        best_obj = best_objective_value_from_outcomes(quadratic_program, outcomes)
                        try:
                            cplex_obj = load_cplex_solution_obj(cplex_h5_paths, problem_name, quadratic_program)
                        except KeyError as e:
                            print(f"Warning: {e}. Skipping.")
                            continue
                        rsq = relative_solution_quality(best_obj, cplex_obj)
                        rows.append({
                            "problem_name": problem_name,
                            "algorithm": effective_algorithm,
                            "reps": effective_reps,
                            "alpha": effective_alpha,
                            "best_objective_value": best_obj,
                            "cplex_fval": cplex_obj,
                            "relative_solution_quality": rsq,
                        })
                    n_processed += 1

        if args.metric == "epa":
            save_success_probabilities(args.outputdir, rows, effective_algorithm, effective_reps, effective_alpha)
        elif args.metric == "rsq":
            if args.avg:
                if args.resume_from is not None:
                    append_avg_relative_solution_quality(args.resume_from, rows)
                else:
                    save_avg_relative_solution_quality(args.outputdir, rows, effective_algorithm, effective_reps, effective_alpha)
            else:
                if args.resume_from is not None:
                    append_relative_solution_quality(args.resume_from, rows)
                else:
                    save_relative_solution_quality(args.outputdir, rows, effective_algorithm, effective_reps, effective_alpha)

    except Exception as e:
        print(f"Error: {e}")
        raise SystemExit(1)

if __name__ == "__main__":
    main()

