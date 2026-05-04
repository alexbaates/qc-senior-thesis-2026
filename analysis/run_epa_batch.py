"""
Batch sampling-EPA computation: processes a slice of all docking problems and computes
the sampling-based expected probability amplitude for all algorithm/reps/alpha
combinations found in --resdir, sharing the per-problem Hamiltonian computation.
"""

import argparse
import csv
import datetime
import json
import os
import re
import sys
import uuid

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from algorithm_analysis import (
    generate_quadratic_program,
    quadratic_program_to_qubo,
    qubo_to_ising,
)


"""
Problem enumeration
"""

def enumerate_all_problems(probdir):
    # Returns a deterministically ordered flat list of (h5_path, problem_key).
    all_problems = []
    for fname in sorted(os.listdir(probdir)):
        if not fname.endswith(".h5"):
            continue
        h5_path = os.path.join(probdir, fname)
        with h5py.File(h5_path, "r") as f:
            if "/docking_problems" not in f:
                continue
            for key in sorted(f["/docking_problems"].keys()):
                all_problems.append((h5_path, key))
    return all_problems


def load_problem(h5_path, key):
    with h5py.File(h5_path, "r") as f:
        group = f["/docking_problems"][key]
        problem_name = group.attrs["name"]
        penalty = group.attrs["penalty"]
        nodes = json.loads(group["nodes"].asstr()[()])
        edges = group["edges"][:]
        weights = group["weights"][:]
    qp = generate_quadratic_program(problem_name, nodes, edges, weights, penalty)
    return problem_name, qp


"""
Results index
"""

def scan_result_files(resdir):
    # Scans resdir for non-CPLEX algorithm result .h5 files.
    # Returns a list of (h5_path, algorithm, reps, alpha) tuples,
    # one entry per unique (algorithm, reps, alpha) combination found.
    combos = []
    for fname in sorted(os.listdir(resdir)):
        if not fname.endswith(".h5") or "cplex" in fname.lower():
            continue
        h5_path = os.path.join(resdir, fname)
        m = re.search(r"reps(\d+)", fname)
        reps = int(m.group(1)) if m else None
        try:
            with h5py.File(h5_path, "r") as f:
                if "results" not in f:
                    continue
                prob_keys = list(f["results"].keys())
                if not prob_keys:
                    continue
                prob_grp = f["results"][prob_keys[0]]
                for alg in prob_grp.keys():
                    if alg == "cplex":
                        continue
                    alg_grp = prob_grp[alg]
                    alpha_keys = sorted(k for k in alg_grp.keys() if k.startswith("alpha_"))
                    if alpha_keys:
                        for ak in alpha_keys:
                            alpha = float(ak.replace("alpha_", ""))
                            combos.append((h5_path, alg, reps, alpha))
                    else:
                        combos.append((h5_path, alg, reps, None))
        except Exception as e:
            print(f"Warning: could not scan '{fname}': {e}")
    return combos

"""
Ground state calculation
"""

def ground_state_bitstrings(qubit_op, tol=1e-10):
    # Returns (bitstrings, n_ground) for the ground states of a diagonal Ising Hamiltonian.
    # Each bitstring is a str of '0'/'1' with left=qubit0=x1, matching the outcome format
    # stored in algorithm result h5 files.
    matrix = qubit_op.to_matrix()
    diag = np.diagonal(matrix).real
    min_val = np.min(diag)
    gnd_idxs = np.where(np.abs(diag - min_val) <= tol)[0]
    n_qubits = qubit_op.num_qubits
    bitstrings = [
        "".join(str((int(idx) >> k) & 1) for k in range(n_qubits))
        for idx in gnd_idxs
    ]
    return bitstrings, len(gnd_idxs)



"""
Main EPA batch processing
"""

def main():
    parser = argparse.ArgumentParser(
        description="Batch sampling-EPA computation for all algorithm combinations."
    )
    parser.add_argument("--probdir", required=True, help="Directory with .h5 problem files")
    parser.add_argument("--resdir", required=True, help="Directory with algorithm result .h5 files")
    parser.add_argument("--outputdir", required=True, help="Directory to write output CSVs")
    parser.add_argument("--batch_id", type=int, required=True, help="0-based batch index (SLURM_ARRAY_TASK_ID)")
    parser.add_argument("--batch_size", type=int, default=100, help="Problems per batch (default: 100)")
    args = parser.parse_args()

    for name, path in [
        ("--probdir", args.probdir),
        ("--resdir", args.resdir),
        ("--outputdir", args.outputdir),
    ]:
        if not os.path.isdir(path):
            parser.error(f"{name}: '{path}' is not a directory")

    tag = f"[batch {args.batch_id}]"

    print(f"{tag} Scanning result files from {args.resdir} ...", flush=True)
    result_combos = scan_result_files(args.resdir)
    print(f"{tag} {len(result_combos)} (algorithm, reps, alpha) combinations found.", flush=True)
    if not result_combos:
        print(f"{tag} No result files found; exiting.")
        return

    print(f"{tag} Enumerating problems in {args.probdir} ...", flush=True)
    all_problems = enumerate_all_problems(args.probdir)
    print(f"{tag} {len(all_problems)} problems total.", flush=True)

    start = args.batch_id * args.batch_size
    end = min(start + args.batch_size, len(all_problems))
    if start >= len(all_problems):
        print(f"{tag} start={start} >= total={len(all_problems)}; nothing to do.")
        return

    batch = all_problems[start:end]
    print(f"{tag} Processing problems {start}–{end - 1} ({len(batch)} problems).", flush=True)

    rows = []
    for i, (h5_path, key) in enumerate(batch):
        # Load problem
        try:
            problem_name, qp = load_problem(h5_path, key)
        except Exception as e:
            print(f"{tag}  [skip] load error key='{key}': {e}", flush=True)
            continue

        # Compute Ising Hamiltonian and ground state bitstrings ONCE per problem
        try:
            qubo = quadratic_program_to_qubo(qp)
            qubit_op, _ = qubo_to_ising(qubo)
            gnd_bitstrings, n_ground = ground_state_bitstrings(qubit_op)
            gnd_set = set(gnd_bitstrings)
        except Exception as e:
            print(f"{tag}  [skip] Hamiltonian error for '{problem_name}': {e}", flush=True)
            continue

        # Compute sampling EPA for every (algorithm, reps, alpha) combination
        for res_h5_path, alg, reps, alpha in result_combos:
            try:
                with h5py.File(res_h5_path, "r") as f:
                    if "results" not in f or problem_name not in f["results"]:
                        continue
                    prob_grp = f["results"][problem_name]
                    if alg not in prob_grp:
                        continue
                    alg_grp = prob_grp[alg]
                    if alpha is not None:
                        alpha_key = f"alpha_{alpha}"
                        if alpha_key not in alg_grp:
                            continue
                        alg_grp = alg_grp[alpha_key]
                    if "outcomes" not in alg_grp or "probabilities" not in alg_grp:
                        continue
                    outcomes = alg_grp["outcomes"].asstr()[:]
                    probabilities = alg_grp["probabilities"][:]
            except Exception as e:
                print(f"{tag}    [skip] {os.path.basename(res_h5_path)} '{problem_name}': {e}", flush=True)
                continue

            prob_dict = dict(zip(outcomes, probabilities))
            sp = float(sum(prob_dict.get(bitstring, 0.0) for bitstring in gnd_set))

            rows.append({
                "problem_name": problem_name,
                "algorithm": alg,
                "reps": reps,
                "alpha": alpha,
                "success_probability": sp,
                "n_ground_states": n_ground,
            })

        if (i + 1) % 20 == 0 or (i + 1) == len(batch):
            print(f"{tag}  [{i + 1}/{len(batch)}] done '{problem_name}'", flush=True)

    # Save CSV
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rand = uuid.uuid4().hex[:6]
    fname = f"success_probabilities_batch{args.batch_id:04d}_{timestamp}_{rand}.csv"
    fpath = os.path.join(args.outputdir, fname)
    with open(fpath, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "problem_name", "algorithm", "reps", "alpha",
                "success_probability", "n_ground_states",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"{tag} Saved {len(rows)} rows → {fpath}", flush=True)


if __name__ == "__main__":
    main()
