# This file is based on code from https://github.com/SMU-Quantum/quantum-optimization-algorithms,
# licensed under the MIT License (Copyright (c) 2025 Monit Sharma).
# Modifications have been made by Alexandra Xiulan Bates, 2026.

# Basic imports
import os
import time
import numpy as np

# Quantum imports
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.circuit.library import QAOAAnsatz
from qiskit import qpy

# SciPy minimizer routine
from scipy.optimize import minimize
from qiskit.primitives import BackendEstimatorV2, BackendSamplerV2
from qiskit_aer import AerSimulator


# Estimate cost function
def cost_func_estimator(params, objective_func_vals, cost_history_dict, ansatz, hamiltonian, estimator):

    # transform the observable defined on virtual qubits to
    # an observable defined on all physical qubits
    isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)

    pub = (ansatz, isa_hamiltonian, params)
    job = estimator.run([pub])

    results = job.result()[0]
    cost = results.data.evs

    cost_history_dict["iters"] += 1
    cost_history_dict["prev_vector"] = params
    cost_history_dict["cost_history"].append(cost)
    print(f"Iters. done: {cost_history_dict['iters']} [Current cost: {cost}]")

    objective_func_vals.append(cost)

    return cost

def to_bitstring(integer, num_bits):
    result = np.binary_repr(integer, width=num_bits)
    return [int(digit) for digit in result]

def run_qaoa(circuits_dir, quadratic_program, reps=3, shots=10000, param_max_iter=100000000, maxfev=1000):
    print(f"Running QAOA for {quadratic_program.name}...")
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
    num_vars = qubo.get_num_vars()
    qubitOp, offset = qubo.to_ising()

    # QAOA algorithm

    print(f"Number of repetitions: {reps}")
    circuit = QAOAAnsatz(cost_operator=qubitOp, reps=reps)
    circuit = circuit.decompose(reps=3)

    initial_gamma = np.pi
    initial_beta = np.pi/2

    # Note: init_params assignment order has been corrected from the source code
    init_params = []
    for p in circuit.parameters:
        if p.name.startswith('β'):
            init_params.append(initial_beta)
        elif p.name.startswith('γ'):
            init_params.append(initial_gamma)

    # number of qubits, circuit depth, gate counts, 2 qubit gate count
    print("Number of variables:", num_vars)
    print("Number of parameters:", len(init_params))
    print('Number of qubits:', circuit.num_qubits)
    print('Circuit depth:', circuit.depth())
    print('Gate counts:', dict(circuit.count_ops()))
    # print new line
    print()
    print("-----------------------------------------------------")
    print()

    # Run QAOA circuit
    objective_func_vals = [] # Store the objective function values

    cost_history_dict = {
        "prev_vector": None,
        "iters": 0,
        "cost_history": [],
    }

    optimization_time_start = time.time()

    result = minimize(
        cost_func_estimator,
        init_params,
        args= (objective_func_vals, cost_history_dict, circuit, qubitOp, estimator),
        method="Powell",
        tol=1e-3,
        options={"maxiter": param_max_iter, "maxfev": maxfev}
    )

    optimization_time_end = time.time()

    # Number of objective function evaluations performed by the optimizer
    function_evals = getattr(result, "nfev", cost_history_dict["iters"])

    print(result)
    print()
 
    # Retrieve optimized parameters

    optimized_params = result.x

    # Post processing
    optimized_circuit = circuit.assign_parameters(result.x)
    optimized_circuit.measure_all()
    pub = (optimized_circuit,)

    circuit_time_start = time.time()

    job = sampler.run([pub], shots=shots)

    circuit_time_end = time.time()

    counts_int = job.result()[0].data.meas.get_int_counts()
    counts_bin = job.result()[0].data.meas.get_counts()
    shots = sum(counts_int.values())
    # final_distribution_int = {key: val/shots for key, val in counts_int.items()}
    final_distribution_bin = {key: val/shots for key, val in counts_bin.items()}

    full_end_time = time.time()

    print("-----------------------------------------------------")
    print()
    print("Time taken for optimization:", optimization_time_end - optimization_time_start, "seconds")
    print("Time taken for circuit execution:", circuit_time_end - circuit_time_start, "seconds")
    print("Total time taken:", full_end_time - full_start_time, "seconds")
    print()

    # Save the optimized QAOA circuit as a .qpy file
    qpy_file_path = os.path.join(circuits_dir, f"{quadratic_program.name}_qaoa_reps{reps}.qpy")

    if os.path.exists(qpy_file_path):
        print(f"Warning: {qpy_file_path} already exists and will be overwritten.")
        os.remove(qpy_file_path)  # Remove existing file if it exists
    
    with open(qpy_file_path, "wb") as qpy_file:
        qpy.dump(optimized_circuit, qpy_file)
    print(f"QAOA circuit for {quadratic_program.name} with reps={reps} saved to {qpy_file_path}.")
    print()

    # Reverse the final distribution list of binary string key pairs
    final_distribution_bin_reversed = {key[::-1]: val for key, val in final_distribution_bin.items()}

    # Calculate optimization and solve times
    optimization_time = optimization_time_end - optimization_time_start
    circuit_time = circuit_time_end - circuit_time_start
    total_time = full_end_time - full_start_time

    return final_distribution_bin_reversed, optimized_params, shots, function_evals, optimization_time, circuit_time, total_time