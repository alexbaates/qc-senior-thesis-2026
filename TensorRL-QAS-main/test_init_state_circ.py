"""
For each .qpy circuit in --circuits, match it to its Hamiltonian .npz in --hamfiles
via the shared DP12_XXXX identifier in the filename. Run the circuit with Qiskit's
AerSimulator (matrix_product_state), evaluate the Ising cost of each sampled bitstring
using the Pauli strings from the Hamiltonian, and save the top-N results per circuit
together with the ground state energy to a combined CSV.
"""

import argparse
import csv
import os
import re
import sys
import time

import numpy as np
from qiskit import qpy
from qiskit.primitives import BackendSamplerV2
from qiskit_aer import AerSimulator


DP_RE = re.compile(r'(DP\d+_[A-Z0-9]+)', re.IGNORECASE)


# Return the first 'DP12_XXXX' token from a filename, or None.
def extract_dp_id(filename):
    m = DP_RE.search(os.path.basename(filename))
    return m.group(1) if m else None


# Map DP id to absolute path for every .npz in hamfiles_dir.
def build_ham_index(hamfiles_dir):
    index = {}
    for fname in os.listdir(hamfiles_dir):
        if fname.endswith('.npz'):
            dp_id = extract_dp_id(fname)
            if dp_id:
                index[dp_id.upper()] = os.path.join(hamfiles_dir, fname)
    return index


# Return the minimum eigenvalue stored in the .npz.
def get_ground_state_energy(data):
    if 'min_eig' in data:
        return float(data['min_eig'])
    return float(data['eigvals'].min())


# Compute the shot-estimated expectation value as a weighted average of 
# Ising costs over all sampled bitstrings.
def compute_expectation_value(counts, paulis, weights, energy_shift):
    total_shots = sum(counts.values())
    expval = 0.0
    for bitstring, count in counts.items():
        cost = compute_ising_cost(bitstring, paulis, weights, energy_shift)
        expval += (count / total_shots) * cost
    return expval


# Evaluate the Ising/QUBO cost of a classical bitstring.
def compute_ising_cost(bitstring, paulis, weights, energy_shift):
    cost = float(energy_shift)
    for pauli, weight in zip(paulis, weights):
        if any(c in ('X', 'Y') for c in pauli):
            continue
        term = float(weight)
        for gate, bit in zip(pauli, bitstring):
            if gate == 'Z':
                term *= 1 - 2 * int(bit)
        cost += term
    return cost


def load_circuit(qpy_path):
    with open(qpy_path, 'rb') as f:
        circuits = qpy.load(f)
    if len(circuits) != 1:
        raise ValueError(
            f"Expected exactly one circuit in '{qpy_path}', found {len(circuits)}."
        )
    return circuits[0]


# Add measurements if absent, run with AerSimulator MPS, return counts.
def run_circuit(circuit, shots):
    qc = circuit.copy()
    has_measurements = any(inst.operation.name == 'measure' for inst in qc.data)
    if not has_measurements:
        qc.measure_all()

    backend = AerSimulator(method='matrix_product_state')
    sampler = BackendSamplerV2(backend=backend, options={'default_shots': shots})

    t0 = time.time()
    job = sampler.run([(qc,)], shots=shots)
    elapsed = time.time() - t0

    counts = job.result()[0].data.meas.get_counts()
    return counts, elapsed


# Return the top_n results sorted by Ising cost (ascending = lowest energy first).
# Each entry: (rank, bitstring, count, probability, cost)
def top_results(counts, paulis, weights, energy_shift, top_n):
    total_shots = sum(counts.values())
    rows = []
    for bitstring, count in counts.items():
        prob = count / total_shots
        cost = compute_ising_cost(bitstring, paulis, weights, energy_shift)
        rows.append((bitstring, count, prob, cost))

    rows.sort(key=lambda r: r[3])  # sort by cost ascending
    return [
        (i + 1, r[0], r[1], r[2], r[3])
        for i, r in enumerate(rows[:top_n])
    ]


def main():
    parser = argparse.ArgumentParser(
        description='Run all .qpy circuits in a directory and evaluate Ising costs.'
    )
    parser.add_argument('--circuits', required=True,
                        help='Directory containing .qpy circuit files, or a single .qpy file.')
    parser.add_argument('--hamfiles', default=None,
                        help='Directory containing .npz Hamiltonian files (matched by DP id).')
    parser.add_argument('--hamfile', default=None,
                        help='Single .npz Hamiltonian file to use for all circuits.')
    parser.add_argument('--output', required=True,
                        help='Path to the output CSV file.')
    parser.add_argument('--shots', type=int, default=10000,
                        help='Number of shots per circuit (default: 10000).')
    parser.add_argument('--n', type=int, default=5,
                        help='Number of top results per circuit (default: 5).')
    args = parser.parse_args()

    if args.shots <= 0:
        raise ValueError('--shots must be a positive integer.')
    if args.n <= 0:
        raise ValueError('--n must be a positive integer.')
    if args.hamfile is None and args.hamfiles is None:
        raise ValueError('Provide either --hamfile (single .npz) or --hamfiles (directory).')
    if args.hamfile is not None and args.hamfiles is not None:
        raise ValueError('Provide only one of --hamfile or --hamfiles, not both.')

    # --circuits can be a single .qpy file or a directory
    if os.path.isfile(args.circuits):
        if not args.circuits.endswith('.qpy'):
            raise ValueError(f'--circuits file must be a .qpy file: {args.circuits}')
        circuits_dir = os.path.dirname(os.path.abspath(args.circuits))
        qpy_files = [os.path.basename(args.circuits)]
    elif os.path.isdir(args.circuits):
        circuits_dir = args.circuits
        qpy_files = sorted(f for f in os.listdir(circuits_dir) if f.endswith('.qpy'))
        if not qpy_files:
            raise ValueError(f'No .qpy files found in {args.circuits}')
    else:
        raise ValueError(f'--circuits is not a valid file or directory: {args.circuits}')

    single_ham_data = None
    if args.hamfile is not None:
        if not os.path.isfile(args.hamfile):
            raise ValueError(f'--hamfile is not a valid file: {args.hamfile}')
        single_ham_data = np.load(args.hamfile, allow_pickle=True)
        ham_index = {}
    else:
        if not os.path.isdir(args.hamfiles):
            raise ValueError(f'--hamfiles is not a valid directory: {args.hamfiles}')
        ham_index = build_ham_index(args.hamfiles)

    total = len(qpy_files)
    print(f'Found {total} .qpy file(s)')
    print()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    fieldnames = [
        'circuit_id', 'rank', 'bitstring',
        'count', 'probability', 'cost', 'expectation_value', 'ground_state_energy',
    ]

    failed = []

    with open(args.output, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for i, fname in enumerate(qpy_files, start=1):
            qpy_path = os.path.join(circuits_dir, fname)
            dp_id = extract_dp_id(fname)
            circuit_id = dp_id if dp_id is not None else os.path.splitext(fname)[0]

            if single_ham_data is not None:
                data = single_ham_data
            else:
                if dp_id is None:
                    print(f'[{i}/{total}] WARNING: cannot extract DP id from {fname} — skipping')
                    failed.append((fname, 'No DP id in filename'))
                    continue

                ham_path = ham_index.get(dp_id.upper())
                if ham_path is None:
                    print(f'[{i}/{total}] WARNING: no matching Hamiltonian for {dp_id} — skipping')
                    failed.append((fname, f'No Hamiltonian match for {dp_id}'))
                    continue

            try:
                if single_ham_data is None:
                    data = np.load(ham_path, allow_pickle=True)
                paulis = data['paulis']
                weights = np.real(data['weights'])
                energy_shift = float(data['energy_shift']) if 'energy_shift' in data else 0.0
                ground_state = get_ground_state_energy(data)

                circuit = load_circuit(qpy_path)
                counts, elapsed = run_circuit(circuit, args.shots)

                results = top_results(counts, paulis, weights, energy_shift, args.n)
                expval = compute_expectation_value(counts, paulis, weights, energy_shift)

                for rank, bitstring, count, prob, cost in results:
                    writer.writerow({
                        'circuit_id': circuit_id,
                        'rank': rank,
                        'bitstring': bitstring,
                        'count': count,
                        'probability': f'{prob:.8f}',
                        'cost': f'{cost:.6f}',
                        'expectation_value': f'{expval:.6f}',
                        'ground_state_energy': f'{ground_state:.6f}',
                    })

                best_cost = results[0][4]
                print(f'[{i}/{total}] {circuit_id}: best_cost={best_cost:.4f}, '
                      f'expval={expval:.4f}, '
                      f'ground_state={ground_state:.4f}, '
                      f'time={elapsed:.2f}s')

            except Exception as e:
                print(f'[{i}/{total}] ERROR — {fname}: {e}')
                failed.append((fname, str(e)))

    print()
    succeeded = total - len(failed)
    if failed:
        print(f'Done. {succeeded}/{total} succeeded, {len(failed)} failed:')
        for fname, err in failed:
            print(f'  - {fname}: {err}')
    else:
        print(f'Done. {succeeded}/{total} succeeded.')
    print(f'Results saved to {args.output}')


if __name__ == '__main__':
    main()
