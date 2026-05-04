import argparse
import os
import re
import time
import h5py
import numpy as np
from qiskit import QuantumCircuit, qpy
from qiskit.primitives import BackendSamplerV2
from qiskit_aer import AerSimulator


# Reconstruct a Qiskit circuit from TensorRL best tracker state_tensor.
def reconstruct_qiskit_circuit(state_tensor, include_measurements=True):
    if state_tensor.ndim != 3:
        raise ValueError(
            f"Expected state_tensor to have 3 dimensions (layers, rows, qubits). Got shape {state_tensor.shape}."
        )

    num_qubits = state_tensor.shape[2]
    qc = QuantumCircuit(num_qubits)

    for local_state in state_tensor:
        # CNOT mask is [0:num_qubits, :]. Row = target, col = control.
        cnot_mask = local_state[:num_qubits, :]
        cnot_targets, cnot_controls = np.nonzero(cnot_mask == 1)
        for target, control in zip(cnot_targets, cnot_controls):
            qc.cx(int(control), int(target))

        # Rotation axis flags are [num_qubits:num_qubits+3, :].
        rot_flags = local_state[num_qubits : num_qubits + 3, :]
        # Rotation angles are [num_qubits+3:num_qubits+6, :], aligned to axis index.
        thetas = local_state[num_qubits + 3 :, :]

        axes, qubits = np.nonzero(rot_flags == 1)
        for axis, qubit in zip(axes, qubits):
            theta = float(thetas[axis][qubit])
            if axis == 0:
                qc.rx(theta, int(qubit))
            elif axis == 1:
                qc.ry(theta, int(qubit))
            elif axis == 2:
                qc.rz(theta, int(qubit))
            else:
                raise ValueError(f"Unexpected rotation axis index: {axis}")

    if include_measurements:
        qc.measure_all()

    return qc


PROBLEM_RE = re.compile(r'(DP\d+_[A-Z0-9]+)')

# Extract problem name (e.g. 'DP12_8HJE') from a TensorRL .npy filename.
def extract_problem_name(fname):
    m = PROBLEM_RE.search(fname)
    if m:
        return m.group(1)
    # Fall back to filename stem if pattern not found
    return os.path.splitext(fname)[0]


def ensure_measurements(circuit):
    has_measurements = any(inst.operation.name == "measure" for inst in circuit.data)
    if not has_measurements:
        raise ValueError("Circuit has no measurements.")

# Run circuit with AerSimulator and return (distribution_bin, circuit_time).
def run_circuit(circuit, shots):
    ensure_measurements(circuit)
    backend = AerSimulator(method="matrix_product_state")
    sampler = BackendSamplerV2(backend=backend, options={"default_shots": shots})
    pub = (circuit,)
    t0 = time.time()
    job = sampler.run([pub], shots=shots)
    result = job.result()
    circuit_time = time.time() - t0
    counts = result[0].data.meas.get_counts()
    total = sum(counts.values())
    distribution_bin = {bs[::-1]: count / total for bs, count in counts.items()}
    return distribution_bin, circuit_time

# Save circuit sampling result to h5 in the same format as main.py.
def save_result_h5(h5_path, problem_name, distribution_bin, params, shots, circuit_time, total_time):
    outcomes = list(distribution_bin.keys())
    probabilities = list(distribution_bin.values())
    with h5py.File(h5_path, "a") as f:
        results_group = f.require_group("results")
        problem_group = results_group.require_group(problem_name)
        alg_group = problem_group.require_group("tensorrl")

        for key in ("outcomes", "probabilities", "parameters"):
            if key in alg_group:
                del alg_group[key]

        alg_group.create_dataset("outcomes", data=outcomes)
        alg_group.create_dataset("probabilities", data=probabilities)
        alg_group.create_dataset("parameters", data=params)
        alg_group.attrs["shots"] = shots
        alg_group.attrs["circuit_time"] = circuit_time
        alg_group.attrs["solve_time"] = circuit_time
        alg_group.attrs["total_time"] = total_time


def main():
    parser = argparse.ArgumentParser(
        description="Batch-convert TensorRL best_circuit_<seed>.npy files into Qiskit .qpy files."
    )
    parser.add_argument("--inputdir", required=True, help="Directory containing best_circuit_<seed>.npy files")
    parser.add_argument("--outputdir", required=True, help="Directory to write output .qpy files")
    parser.add_argument(
        "--shots",
        type=int,
        default=10000,
        help="Number of shots for circuit sampling (default: 10000). Only used with --outputh5.",
    )
    parser.add_argument(
        "--outputh5",
        default=None,
        help="Path to shared .h5 file for sampling results. If omitted, circuits are converted to .qpy only.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.inputdir):
        parser.error(f"--inputdir: '{args.inputdir}' is not a directory")

    npy_files = sorted(f for f in os.listdir(args.inputdir) if f.endswith(".npy"))
    if not npy_files:
        print(f"No .npy files found in '{args.inputdir}'.")
        return

    os.makedirs(args.outputdir, exist_ok=True)

    n_ok = 0
    n_fail = 0
    for fname in npy_files:
        input_path = os.path.join(args.inputdir, fname)
        output_path = os.path.join(args.outputdir, os.path.splitext(fname)[0] + ".qpy")
        try:
            tracker = np.load(input_path, allow_pickle=True).item()
            if "state_tensor" not in tracker or tracker["state_tensor"] is None:
                raise ValueError("Missing or None 'state_tensor' in file.")
            state_tensor = np.array(tracker["state_tensor"])
            qc = reconstruct_qiskit_circuit(state_tensor, include_measurements=True)

            with open(output_path, "wb") as f:
                qpy.dump(qc, f)

            if args.outputh5 is not None:
                total_start = time.time()
                problem_name = extract_problem_name(fname)
                distribution_bin, circuit_time = run_circuit(qc, args.shots)
                total_time = time.time() - total_start
                theta = tracker.get("theta_tensor")
                params = np.array(theta) if theta is not None else np.array([])
                save_result_h5(
                    args.outputh5,
                    problem_name,
                    distribution_bin,
                    params,
                    args.shots,
                    circuit_time,
                    total_time,
                )

            n_ok += 1
        except Exception as e:
            print(f"Warning: skipping '{fname}': {e}")
            n_fail += 1

    print(f"{n_ok}/{n_ok + n_fail} circuits converted successfully → '{args.outputdir}'")


if __name__ == "__main__":
    main()
