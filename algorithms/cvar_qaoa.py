# This file is based on code from https://github.com/SMU-Quantum/quantum-optimization-algorithms,
# licensed under the MIT License (Copyright (c) 2025 Monit Sharma).
# Modifications have been made by Alexandra Xiulan Bates, 2026.

import time
import os
import numpy as np

# quantum imports
from qiskit.circuit import QuantumCircuit,ParameterVector
from qiskit_optimization.algorithms import CplexOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.quantum_info import SparsePauliOp
from qiskit import qpy
# Pre-defined ansatz circuit and operator class for Hamiltonian
from qiskit.circuit.library import EfficientSU2, PauliEvolutionGate

# SciPy minimizer routine
from scipy.optimize import minimize
from qiskit.primitives import BackendSamplerV2
from qiskit_aer import AerSimulator

# Define CVaR function
def compute_cvar(probabilities, values, alpha):
    sorted_indices = np.argsort(values)
    probs = np.array(probabilities)[sorted_indices]
    vals = np.array(values)[sorted_indices]
    cvar = 0
    total_prob = 0
    for p, v in zip(probs, vals):
        if total_prob + p > alpha:
            p = alpha - total_prob
        total_prob += p
        cvar += p * v
        if total_prob >= alpha:
            break
    return cvar / alpha

# Define the cost function
def eval_bitstring(H, x):
    spins = np.array([(-1) ** int(b) for b in x[::-1]])
    value = 0.0
    for pauli, coeff in zip(H.paulis, H.coeffs):
        z_indices = np.where(pauli.z)[0]
        contribution = coeff.real * np.prod(spins[z_indices])
        value += contribution
    return value

class CVaRObjective:
    def __init__(self, H, offset, alpha, sampler, ansatz):
        self.H = H
        self.offset = offset
        self.alpha = alpha
        self.sampler = sampler
        self.ansatz = ansatz
        self.history = []
        self.cost_history_dict = {"prev_vector": None, "iters": 0, "cost_history": []}

    def evaluate(self, params):
        assigned_circuit = self.ansatz.assign_parameters(params)
        assigned_circuit.measure_all()

        # Run circuit
        job = self.sampler.run([assigned_circuit])
        result = job.result()
        counts = result[0].data.meas.get_counts()
        total_shots = sum(counts.values())
        probabilities = [v / total_shots for v in counts.values()]
        bitstrings = list(counts.keys())
        values = [eval_bitstring(self.H, b) + self.offset for b in bitstrings]

        cvar = compute_cvar(probabilities, values, self.alpha)
        self.history.append(cvar)
        self.cost_history_dict["iters"] += 1
        self.cost_history_dict["cost_history"].append(cvar)
        print(f"Iters. done: {self.cost_history_dict['iters']} [Current cost: {cvar}]")
        return cvar
    
# Build ansatz

# function to make qaoa circuit
def generate_sum_x_pauli_str(length):
    ret = []
    for i in range(length):
        paulis = ['I'] * length
        paulis[i] = 'X'
        ret.append(''.join(paulis))

    return ret

def qaoa_circuit(quadratic_program_ham: SparsePauliOp, depth: int = 1) -> QuantumCircuit:
    r"""
    Input:
    - quadratic_program_ham: Quadratic_program Hamiltonian to construct the QAOA circuit.
    Standard procedure would be:
    ```
        hamiltonian, offset = qubo.to_ising()
        qc = qaoa_circuit_from_qubo(hamiltonian)
    ```

    Returns:
    - qc: A QuantumCircuit object representing the QAOA circuit e^{-i\beta H_M} e^{-i\gamma H_C}.
    """
    num_qubits = quadratic_program_ham.num_qubits

    gamma = ParameterVector(name=r'$\gamma$', length=depth)
    beta = ParameterVector(name=r'$\beta$', length=depth)

    mixer_ham = SparsePauliOp(generate_sum_x_pauli_str(num_qubits))
    
    qc = QuantumCircuit(num_qubits)
    qc.h(range(num_qubits))

    for p in range(depth):
        exp_gamma = PauliEvolutionGate(quadratic_program_ham, time=gamma[p])
        exp_beta = PauliEvolutionGate(mixer_ham, time=beta[p])
        qc.append(exp_gamma, qargs=range(num_qubits))
        qc.append(exp_beta, qargs=range(num_qubits))

    return qc

def run_cvar_qaoa(circuits_dir, quadratic_program, alpha, reps=2, shots=10000, param_max_iter=100000000, maxfev=1000):
    print(f"Running CVaR QAOA for {quadratic_program.name}...")
    print()
    full_start_time = time.time()

    backend = AerSimulator(method='matrix_product_state')
    sampler = BackendSamplerV2(backend=backend, options={"default_shots": shots})

    converter = QuadraticProgramToQubo()
    qubo = converter.convert(quadratic_program)
    print("QUBO:")
    print(qubo)
    print()

    """
    Finish
    """

    num_vars = qubo.get_num_vars()
    print('Number of variables:', num_vars)
    print('Number of repetitions:', reps)
    # converting hamiltonian
    qubitOp, offset = qubo.to_ising()

    num_qubits = qubitOp.num_qubits
    print('Number of qubits:', num_qubits)
    ansatz = qaoa_circuit(qubitOp, depth=reps)
    ansatz = ansatz.decompose()
    #ansatz = EfficientSU2(num_qubits, reps=reps).decompose()
    num_params = ansatz.num_parameters

    # Initial parameters
    initial_params = 2 * np.pi * np.random.random(num_params)

    # Optimization
    results_summary = {}

    print(f"\nStarting optimization for alpha = {alpha}")
    objective = CVaRObjective(qubitOp, offset, alpha, sampler, ansatz)

    optimization_time_start = time.time()

    res = minimize(
        objective.evaluate,
        initial_params,
        method="cobyla",
        options={"maxiter": param_max_iter, "maxfev": maxfev}
    )


    optimization_time_end = time.time()

    # Number of objective function evaluations performed by the optimizer
    function_evals = getattr(res, "nfev", objective.cost_history_dict["iters"])

    print(res)
    print()

    # Retrieve results
    optimized_params = res.x
    final_circuit = ansatz.assign_parameters(optimized_params)
    final_circuit.measure_all()

    circuit_time_start = time.time()
    job = sampler.run([final_circuit], shots=shots)
    circuit_time_end = time.time()

    result = job.result()
    counts = result[0].data.meas.get_counts()
    shots = sum(counts.values())

    # Normalize and retrieve distribution
    distribution_bin = {k: v / shots for k, v in counts.items()}
    sorted_keys = sorted(distribution_bin, key=distribution_bin.get, reverse=True)

    # Get top 4 results
    top_4_results = sorted_keys[:4]
    top_4_probabilities = [distribution_bin[k] for k in top_4_results]

    # Print top 4 results
    print("\nTop 4 Results:")
    for bitstring, probability in zip(top_4_results, top_4_probabilities):
        print(f"Bitstring: {bitstring}, Probability: {probability:.6f}")

    # Convert bitstrings to solutions
    print("\nConverted Solutions:")
    for bitstring in top_4_results:
        solution = converter.interpret([int(b) for b in bitstring])
        cost = quadratic_program.objective.evaluate(solution)
        feasible = quadratic_program.get_feasibility_info(solution)[0]
        print(f"Solution: {solution}, Cost: {cost}, Feasible: {feasible}")

    full_end_time = time.time()

    # Store results summary
    results_summary[alpha] = {
        "top_4_results": top_4_results,
        "top_4_probabilities": top_4_probabilities,
    }

    # Summary of results for all alphas
    print("\nResults Summary:")
    for alpha, summary in results_summary.items():
        print(f"\nAlpha = {alpha}")
        print("Top 4 Results:", summary["top_4_results"])
        print("Top 4 Probabilities:", summary["top_4_probabilities"])

    print("-----------------------------------------------------")
    print()
    print("Time taken for optimization:", optimization_time_end - optimization_time_start, "seconds")
    print("Time taken for circuit execution:", circuit_time_end - circuit_time_start, "seconds")
    print("Total time taken:", full_end_time - full_start_time, "seconds")
    print()

    # Save the optimized CVaR QAOA circuit as a .qpy file
    qpy_file_path = os.path.join(circuits_dir, f"{quadratic_program.name}_cvar_qaoa_reps{reps}_alpha{alpha}.qpy")

    if os.path.exists(qpy_file_path):
        print(f"Warning: {qpy_file_path} already exists and will be overwritten.")
        os.remove(qpy_file_path)  # Remove existing file if it exists
    
    with open(qpy_file_path, "wb") as qpy_file:
        qpy.dump(final_circuit, qpy_file)

    print(f"CVAR_QAOA circuit for {quadratic_program.name} with reps={reps} and alpha={alpha} saved to {qpy_file_path}.")
    print()

    # Reverse the final distribution list of binary string key pairs
    distribution_bin_reversed = {key[::-1]: val for key, val in distribution_bin.items()}

    # Calculate optimization and solve times
    optimization_time = optimization_time_end - optimization_time_start
    circuit_time = circuit_time_end - circuit_time_start
    total_time = full_end_time - full_start_time

    return distribution_bin_reversed, optimized_params, shots, function_evals, optimization_time, circuit_time, total_time

