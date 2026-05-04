# This file is based on code from https://github.com/SMU-Quantum/quantum-optimization-algorithms,
# licensed under the MIT License (Copyright (c) 2025 Monit Sharma).
# Modifications have been made by Alexandra Xiulan Bates, 2026.

import time
import h5py
import os
from datetime import datetime

# basic imports
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# quantum imports
from qiskit_optimization import QuadraticProgram 
from qiskit_optimization.applications import Maxcut, Knapsack
from qiskit.circuit import Parameter,QuantumCircuit
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.algorithms import CplexOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector
from qiskit import qpy
# Pre-defined ansatz circuit and operator class for Hamiltonian
from qiskit.circuit.library import EfficientSU2
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_optimization.algorithms import MinimumEigenOptimizer

# SciPy minimizer routine
from scipy.optimize import minimize
from qiskit.primitives import BackendEstimatorV2, BackendSamplerV2
from qiskit_aer import AerSimulator



# Define the cost function

def cost_func(params, cost_history_dict, ansatz, hamiltonian, estimator):
    """Return estimate of energy from estimator

    Parameters:
        params (ndarray): Array of ansatz parameters
        cost_history_dict (dict): Dictionary for storing intermediate results
        ansatz (QuantumCircuit): Parameterized ansatz circuit
        hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
        estimator (EstimatorV2): Estimator primitive instance
        cost_history_dict: Dictionary for storing intermediate results

    Returns:
        float: Energy estimate
    """
    pub = (ansatz, [hamiltonian], [params])
    result = estimator.run(pubs=[pub]).result()
    energy = result[0].data.evs[0]

    cost_history_dict["iters"] += 1
    cost_history_dict["prev_vector"] = params
    cost_history_dict["cost_history"].append(energy)
    print(f"Iters. done: {cost_history_dict['iters']} [Current cost: {energy}]")

    return energy


# auxiliary functions to sample most likely bitstring
def to_bitstring(integer, num_bits):
    result = np.binary_repr(integer, width=num_bits)
    return [int(digit) for digit in result]


def run_vqe(circuits_dir, quadratic_program, reps=2, shots=10000, param_max_iter=100000000, maxfev=1000):
    print(f"Running VQE for {quadratic_program.name}...")
    print()
    full_start_time = time.time()

    backend = AerSimulator(method='matrix_product_state')
    estimator = BackendEstimatorV2(backend=backend)
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

    # make the ansatz circuit
    ansatz = EfficientSU2(qubitOp.num_qubits,reps=reps)
    ansatz = ansatz.decompose()
    num_params = ansatz.num_parameters

    print('Number of parameters:', num_params)

    # number of qubits, circuit depth, gate counts, 2 qubit gate count
    print('Number of qubits:', ansatz.num_qubits)
    print('Circuit depth:', ansatz.depth())
    print('Gate counts:', dict(ansatz.count_ops()))

    # print new line
    print()
    print("-----------------------------------------------------")

    cost_history_dict = {
        "prev_vector": None,
        "iters": 0,
        "cost_history": [],
    }

    # Initial parameters
    x0 = 2 * np.pi * np.random.random(num_params)


    # optimization loop
    optimization_time_start = time.time()
   
    res = minimize(
            cost_func,
            x0,
            args=(cost_history_dict, ansatz, qubitOp, estimator),
            method="cobyla",
            options={"maxiter": param_max_iter, "maxfev": maxfev}
        )
    
    optimization_time_end = time.time()

    # Number of objective function evaluations performed by the optimizer
    function_evals = getattr(res, "nfev", cost_history_dict["iters"])
    
    print(res)
    print()

    optimized_params = res.x

    # sanity check

    all(cost_history_dict["prev_vector"] == res.x)
    cost_history_dict["iters"] == res.nfev

    # get the results
    ansatz = ansatz.assign_parameters(res.x)
    ansatz.measure_all()

    pub = (ansatz,)
    
    circuit_time_start = time.time()
    job = sampler.run([pub], shots=shots)
    circuit_time_end = time.time()

    counts_int = job.result()[0].data.meas.get_int_counts()
    counts_bin = job.result()[0].data.meas.get_counts()
    shots = sum(counts_int.values())
    final_distribution_int = {key: val/shots for key, val in counts_int.items()}
    final_distribution_bin = {key: val/shots for key, val in counts_bin.items()}

    keys = list(final_distribution_int.keys())
    values = list(final_distribution_int.values())

    # Find the indices of the top 4 values
    top_4_indices = np.argsort(np.abs(values))[::-1][:4]
    top_4_results = []
    # Print the top 4 results with their probabilities
    print("Top 4 Results:")
    for idx in top_4_indices:
        bitstring = to_bitstring(keys[idx], num_vars)
        bitstring.reverse()
        top_4_results.append(bitstring)
        print(f"Bitstring: {bitstring}, Probability: {values[idx]:.6f}")

    print()
    print("--------------------")

    # Iterate through the list of bitstrings and evaluate for each
    for bitstring in top_4_results:
        result = converter.interpret(bitstring)  # Interpret the bitstring
        cost = quadratic_program.objective.evaluate(result)  # Evaluate the cost for the bitstring
        feasible =quadratic_program.get_feasibility_info(result)[0]
        
        # Print the results
        print("Result:", result)
        print("Result value:", cost)
        print("Feasible solution:", feasible)

    full_end_time = time.time()

    print()
    print("-----------------------------------------------------")
    print("Time taken for optimization:", optimization_time_end - optimization_time_start, "seconds")
    print("Time taken for circuit execution:", circuit_time_end - circuit_time_start, "seconds")
    print("Total time taken:", full_end_time - full_start_time, "seconds")

    # Save the optimized VQE circuit as a .qpy file
    qpy_file_path = os.path.join(circuits_dir, f"{quadratic_program.name}_vqe_reps{reps}.qpy")

    if os.path.exists(qpy_file_path):
        print(f"Warning: {qpy_file_path} already exists and will be overwritten.")
        os.remove(qpy_file_path)  # Remove existing file if it exists
    
    with open(qpy_file_path, "wb") as qpy_file:
        qpy.dump(ansatz, qpy_file)
    print(f"VQE circuit for {quadratic_program.name} with reps={reps} saved to {qpy_file_path}.")
    print()

    # Reverse the final distribution list of binary string key pairs
    final_distribution_bin_reversed = {key[::-1]: val for key, val in final_distribution_bin.items()}

    # Calculate optimization and solve times
    optimization_time = optimization_time_end - optimization_time_start
    circuit_time = circuit_time_end - circuit_time_start
    total_time = full_end_time - full_start_time

    return final_distribution_bin_reversed, optimized_params, shots, function_evals, optimization_time, circuit_time, total_time