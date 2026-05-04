import numpy as np
import json
import h5py
import argparse
import os
import time
import qiskit_optimization
from qiskit_optimization.converters import QuadraticProgramToQubo

# Convert a Qiskit QuadraticProgram to an Ising Hamiltonian. Returns tuple: (ising_matrix, 
# ising_offset, pauli_dict).
def quadratic_program_to_ising(quadratic_program, offset=0.0):
    print(f"Converting the quadratic program for {quadratic_program.name} to an Ising Hamiltonian...")

    converter = QuadraticProgramToQubo()
    qubo = converter.convert(quadratic_program)

    qubit_op, qubo_offset = qubo.to_ising()

    n = qubo.get_num_vars()

    ising_offset = float(np.real(qubo_offset)) + offset
    raw_pauli_dict = {}

    # Aggregate coefficients exactly as returned by qiskit.
    for pauli, coeff in zip(qubit_op.paulis, qubit_op.coeffs):
        pauli_str = str(pauli)
        coeff_real = float(np.real(coeff))
        raw_pauli_dict[pauli_str] = raw_pauli_dict.get(pauli_str, 0.0) + coeff_real

    # Keep only non-identity operator terms in pauli_dict.
    # Any identity contribution is absorbed into ising_offset to avoid double counting
    # between the dense matrix and the separate offset field.
    pauli_dict = {}

    H = np.zeros((n, n))

    # Build the Ising matrix from Z / ZZ terms in the Pauli representation.
    for pauli_str, coeff in raw_pauli_dict.items():
        z_positions = [idx for idx, char in enumerate(pauli_str) if char == 'Z']

        if len(z_positions) == 0:
            ising_offset += coeff
        elif len(z_positions) == 1:
            pauli_dict[pauli_str] = coeff
            H[z_positions[0], z_positions[0]] += coeff
        elif len(z_positions) == 2:
            pauli_dict[pauli_str] = coeff
            i, j = z_positions
            H[i, j] += coeff
            H[j, i] += coeff

    return H, ising_offset, pauli_dict

# Convert Pauli dictionary to arrays of Pauli strings and weights. Returns the tuple
# (paulis_array, weights_array).
def pauli_dict_to_arrays(pauli_dict):
    paulis = []
    weights = []
    
    for pauli_str in sorted(pauli_dict.keys()):
        paulis.append(pauli_str)
        weights.append(pauli_dict[pauli_str])
    
    return np.array(paulis, dtype=str), np.array(weights, dtype=float)

# Convert Pauli Hamiltonian dict to dense matrix representation.
def ising_pauli_dict_to_dense_matrix(pauli_dict, n_qubits):
    # Pauli matrices
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    pauli_map = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
    
    # Initialize full Hamiltonian
    H_full = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    
    for pauli_str, coeff in pauli_dict.items():
        # Build tensor product of Pauli matrices
        term = np.array([[1]], dtype=complex)
        for pauli_char in pauli_str[::-1]:  # reverse for tensor order
            term = np.kron(term, pauli_map[pauli_char])
        H_full += coeff * term
    
    return H_full

# Saves the Hamiltonian in .npz format.
def save_hamiltonian_to_npz(filename, pauli_dict, n_qubits, ising_offset=0.0, compute_dense=True, overwrite=True):
    print(f"Saving Hamiltonian to {filename}...")
    # Build complete Pauli dict including the constant (identity) term so that
    # paulis+weights fully describe the Hamiltonian, consistent with the dense
    # matrix and matching the format produced by making_molecules.py.
    complete_pauli_dict = dict(pauli_dict)
    if ising_offset != 0.0:
        identity_str = 'I' * n_qubits
        complete_pauli_dict[identity_str] = (
            complete_pauli_dict.get(identity_str, 0.0) + ising_offset
        )

    # Convert Pauli strings from Qiskit little-endian to big-endian (standard /
    # PennyLane) convention expected by mps2qc and the TensorRL pipeline.
    sorted_keys = sorted(complete_pauli_dict.keys())
    paulis_be = [p[::-1] for p in sorted_keys]
    weights_list = [complete_pauli_dict[p] for p in sorted_keys]

    data_dict = {
        'paulis': paulis_be,
        'weights': np.array(weights_list, dtype=float),
        'energy_shift': 0.0,
    }

    if compute_dense:
        # ising_pauli_dict_to_dense_matrix accepts Qiskit-convention (little-
        # endian) strings but its internal [::-1] kron loop produces a matrix
        # in big-endian basis ordering, consistent with paulis_be above.
        hamiltonian_dense = ising_pauli_dict_to_dense_matrix(
            complete_pauli_dict, n_qubits
        )

        data_dict['hamiltonian'] = hamiltonian_dense
        data_dict['eigvals'] = np.linalg.eigvalsh(np.real(hamiltonian_dense))
    else:
        # The offset is already folded into paulis/weights as an identity
        # term, so energy_shift stays 0.0 — no separate correction needed.
        pass

    if os.path.exists(filename):
        if overwrite:
            print(f"Warning: {filename} already exists and will be overwritten.")
        else:
            raise FileExistsError(
                f"Output file already exists: {filename}. Use --overwrite to overwrite."
            )

    np.savez(filename, **data_dict)
    print(f"Saved to {filename}")
    print(f"  - {len(paulis_be)} Pauli terms")
    print(f"  - {n_qubits} qubits")
    print(f"  - Offset: {ising_offset}")

def verify_saved_hamiltonian(filename):
    print(f"\n{'='*70}")
    print(f"Loading and verifying: {filename}")
    print(f"{'='*70}")
    
    data = np.load(filename, allow_pickle=True)
    
    print(f"\nContents of {filename}:")
    print(f"  Keys: {list(data.files)}")

    if 'energy_shift' in data.files:
        print(f"  Energy shift: {data['energy_shift']}")
    
    paulis = data['paulis']
    weights = data['weights']
    print(f"\n  Number of Pauli terms: {len(paulis)}")
    if 'n_qubits' in data.files:
        print(f"  Number of qubits: {data['n_qubits'].item()}")
    elif len(paulis) > 0:
        print(f"  Number of qubits (inferred): {len(paulis[0])}")
    
    print(f"\n  Pauli terms (first 10):")
    for i, (p, w) in enumerate(zip(paulis[:10], weights[:10])):
        print(f"    {p}: {w:.6f}")
    
    if 'hamiltonian' in data.files:
        ham = data['hamiltonian']
        print(f"\n  Dense Hamiltonian shape: {ham.shape}")
        print(f"  Dense Hamiltonian (first 5x5):\n{ham[:5, :5]}")

def generate_quadratic_program(name, nodes, edges, weights, penalty):
    quadratic_program = qiskit_optimization.QuadraticProgram(f"{name}")
    N = len(nodes)
    for i in range(N):
        quadratic_program.binary_var(name=f"x{i + 1}")
    linear = {f"x{i + 1}": weights[i] for i in range(N)}

    edge_ids = {int(v) for edge in edges for v in edge}
    if edge_ids:
        min_edge_id = min(edge_ids)
        max_edge_id = max(edge_ids)
        if min_edge_id < 0 or max_edge_id >= N:
            raise ValueError(
                "Edges must be 0-indexed node IDs in [0, N-1]. "
                f"Got min={min_edge_id}, max={max_edge_id}, N={N}."
            )
    
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

def load_quadratic_program_from_h5(prob_file_path, docking_problem_name):
    if not isinstance(prob_file_path, str):
        raise TypeError("prob_file_path must be a file path string")

    with h5py.File(prob_file_path, "r") as prob_file:
        print("Retrieving quadratic programs...")
        print()

        if "/docking_problems" not in prob_file:
            raise ValueError("/docking_problems group was not found in the provided .h5 file")

        docking_problems_group = prob_file["/docking_problems"]

        matching_group_keys = []
        for docking_problem_group_key in docking_problems_group.keys():
            docking_problem_group = docking_problems_group[docking_problem_group_key]
            problem_name = docking_problem_group.attrs["name"]

            if problem_name == docking_problem_name:
                matching_group_keys.append(docking_problem_group_key)

        if len(matching_group_keys) == 0:
            raise ValueError(
                f"Docking problem with name '{docking_problem_name}' was not found in {prob_file_path}"
            )

        if len(matching_group_keys) > 1:
            raise ValueError(
                f"Multiple docking problems found with name '{docking_problem_name}'. "
                f"Matched groups: {matching_group_keys}"
            )

        matched_group = docking_problems_group[matching_group_keys[0]]
        problem_name = matched_group.attrs["name"]
        penalty = matched_group.attrs["penalty"]
        nodes_json = matched_group["nodes"].asstr()[()]
        nodes = json.loads(nodes_json)
        edges = matched_group["edges"][:]
        weights = matched_group["weights"][:]

        quadratic_program = generate_quadratic_program(problem_name, nodes, edges, weights, penalty)

        return quadratic_program

def load_all_problems_from_h5(prob_file_path):
    """Load all docking problem data from an .h5 file. Returns a list of
    (name, nodes, edges, weights, penalty) tuples."""
    problems = []
    with h5py.File(prob_file_path, "r") as prob_file:
        if "/docking_problems" not in prob_file:
            raise ValueError("/docking_problems group was not found in the provided .h5 file")
        docking_problems_group = prob_file["/docking_problems"]
        for key in docking_problems_group.keys():
            group = docking_problems_group[key]
            name = group.attrs["name"]
            penalty = group.attrs["penalty"]
            nodes = json.loads(group["nodes"].asstr()[()])
            edges = group["edges"][:]
            weights = group["weights"][:]
            problems.append((name, nodes, edges, weights, penalty))
    return problems


def process_one_problem(problem_name, nodes, edges, weights, penalty, output_dir, overwrite, index=None, total=None):
    """Convert one docking problem to an Ising .npz file. Prints a single summary line."""
    problem_start_time = time.time()

    quadratic_program = generate_quadratic_program(problem_name, nodes, edges, weights, penalty)
    ising_matrix, ising_offset, pauli_dict = quadratic_program_to_ising(quadratic_program)

    n_qubits = ising_matrix.shape[0]
    output_filename = f"qubo_{n_qubits}q_geom_docking_problem_{problem_name}_ising.npz"
    output_path = os.path.join(output_dir, output_filename)

    save_hamiltonian_to_npz(
        filename=output_path,
        pauli_dict=pauli_dict,
        n_qubits=n_qubits,
        ising_offset=ising_offset,
        compute_dense=True,
        overwrite=overwrite,
    )

    prefix = f"[{index}/{total}] " if index is not None else ""
    print(f"{prefix}{problem_name}: {n_qubits} qubits, {len(pauli_dict)} Pauli terms -> {output_path}")
    problem_end_time = time.time()
    problem_time = problem_end_time - problem_start_time
    print(f"Time taken for {problem_name}: {problem_time:.2f}s")
    print()
    return output_path


def main():
    full_start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--probfile", required=True, type=str, help="Path to .h5 file containing docking quadratic programs")
    parser.add_argument("--output", required=True, type=str, help="Directory to save output .npz files")
    parser.add_argument("--dp", required=False, default=None, type=str, help="Name of a single target docking problem (omit to process all problems in the file)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .npz files. Without this flag, encountering an existing file is an error.")
    args = parser.parse_args()

    if not args.probfile.lower().endswith(".h5"):
        raise ValueError("--probfile must point to an .h5 file")

    if not os.path.isfile(args.probfile):
        raise ValueError(f"Problem file not found: {args.probfile}")

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    if args.dp:
        # Single problem mode
        print()
        print(f"Loading docking problem '{args.dp}' from {args.probfile}...")
        print()

        quadratic_program = load_quadratic_program_from_h5(args.probfile, args.dp)
        ising_matrix, ising_offset, pauli_dict = quadratic_program_to_ising(quadratic_program)

        n_qubits = ising_matrix.shape[0]
        output_filename = f"qubo_{str(n_qubits)}q_geom_docking_problem_{args.dp}_ising.npz"
        output_path = os.path.join(output_dir, output_filename)

        save_hamiltonian_to_npz(
            filename=output_path,
            pauli_dict=pauli_dict,
            n_qubits=n_qubits,
            ising_offset=ising_offset,
            compute_dense=True,
            overwrite=args.overwrite,
        )

        print()
        print(f"Saved Hamiltonian for docking problem '{args.dp}' to: {output_path}")
        print()

        verify_saved_hamiltonian(output_path)

    else:
        # Batch mode: process all problems
        print()
        print(f"Loading all docking problems from {args.probfile}...")
        print()

        problems = load_all_problems_from_h5(args.probfile)
        total = len(problems)
        print(f"Found {total} docking problems. Processing...")
        print()

        failed = []

        for i, (name, nodes, edges, weights, penalty) in enumerate(problems, start=1):
            try:
                process_one_problem(
                    problem_name=name,
                    nodes=nodes,
                    edges=edges,
                    weights=weights,
                    penalty=penalty,
                    output_dir=output_dir,
                    overwrite=args.overwrite,
                    index=i,
                    total=total,
                )
            except Exception as e:
                print(f"[{i}/{total}] ERROR - {name}: {e}")
                failed.append((name, str(e)))

        print()
        if failed:
            print(f"Batch complete. {total - len(failed)}/{total} succeeded. {len(failed)} failed:")
            print()
            for name, err in failed:
                print(f"  - {name}: {err}")
        else:
            print(f"Batch complete. All {total} problems processed successfully.")
            print()

    full_end_time = time.time()
    total_time = full_end_time - full_start_time
    print(f"Total time taken: {total_time:.2f}s")


if __name__ == "__main__":
    main()


