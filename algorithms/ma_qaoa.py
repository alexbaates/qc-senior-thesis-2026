# This file is based on code from https://github.com/SMU-Quantum/quantum-optimization-algorithms,
# licensed under the MIT License (Copyright (c) 2025 Monit Sharma).
# Modifications have been made by Alexandra Xiulan Bates, 2026.

# basic imports
import time
import os
import numpy as np

# quantum imports
from qiskit_optimization.algorithms import CplexOptimizer
from qiskit.circuit import Parameter,QuantumCircuit
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.quantum_info import Statevector
from qiskit import qpy

# Pre-defined ansatz circuit and operator class for Hamiltonian
from qiskit.circuit.library import RZGate, RZZGate, RXGate

# SciPy minimizer routine
from scipy.optimize import minimize
from qiskit.primitives import BackendEstimatorV2, BackendSamplerV2
from qiskit_aer import AerSimulator

def build_ma_qaoa_circuit(pauli_strings, coefficients, reps):
    # Number of qubits
    num_qubits = len(pauli_strings[0])  # Length of the Pauli string determines the number of qubits

    # Create a quantum circuit
    qc = QuantumCircuit(num_qubits)

    # Step 1: Apply equal superposition (Hadamard gates on all qubits)
    qc.h(range(num_qubits))

    # Initialize counter for unique parameters
    theta_counter = 0

    # Function to add a single layer of gates based on Pauli strings and coefficients
    def add_qaoa_layer(qc, theta_counter):
        for pauli_string, coeff in zip(pauli_strings, coefficients):
            # Identify indices with 'Z'
            z_indices = [i for i, char in enumerate(reversed(pauli_string)) if char == 'Z']
            
            # If there's a single 'Z', add an RZ gate with a unique parameter
            if len(z_indices) == 1:
                qubit = z_indices[0]
                theta = Parameter(f"θ_{theta_counter}")
                theta_counter += 1
                angle = 2 * coeff * theta
                qc.append(RZGate(angle), [qubit])
            
            # If there are two 'Z's, add an RZZ gate with a unique parameter
            elif len(z_indices) == 2:
                qubit1, qubit2 = z_indices
                theta = Parameter(f"θ_{theta_counter}")
                theta_counter += 1
                angle = 2 * coeff * theta
                qc.append(RZZGate(angle), [qubit1, qubit2])
        
        # Apply RX gates on all qubits with unique parameters
        for qubit in range(num_qubits):
            theta = Parameter(f"θ_{theta_counter}")
            theta_counter += 1
            angle = 2 * theta
            qc.append(RXGate(angle), [qubit])
        
        return theta_counter

    # Step 2: Add the repeating portion of the circuit
    for _ in range(reps):
        theta_counter = add_qaoa_layer(qc, theta_counter)

    return qc

def cost_func_exact(params, cost_history_dict, objective_func_vals, ansatz, hamiltonian, estimator):

    bound = ansatz.assign_parameters(params)
    psi = Statevector.from_instruction(bound)

    cost = psi.expectation_value(hamiltonian).real

    cost_history_dict["iters"] += 1
    cost_history_dict["prev_vector"] = params
    cost_history_dict["cost_history"].append(cost)

    print(f"Iters. done: {cost_history_dict['iters']} [Current cost: {cost}]")

    objective_func_vals.append(cost)

    return cost

def cost_and_grad(params,cost_history_dict, objective_func_vals, ansatz, hamiltonian, estimator):
    shift = np.pi / 2
    grad = np.zeros_like(params)

    # compute base cost once
    base_cost = cost_func_exact(params, cost_history_dict, objective_func_vals, ansatz, hamiltonian, estimator)

    for i in range(len(params)):
        plus = params.copy()
        minus = params.copy()

        plus[i] += shift
        minus[i] -= shift

        forward = cost_func_exact(plus, cost_history_dict, objective_func_vals, ansatz, hamiltonian, estimator)
        backward = cost_func_exact(minus, cost_history_dict, objective_func_vals, ansatz, hamiltonian, estimator)

        grad[i] = 0.5 * (forward - backward)

    return base_cost, grad


def cost_func_estimator(params, cost_history_dict, objective_func_vals, ansatz, hamiltonian, estimator):

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

# def cost_func(params, ansatz, hamiltonian, estimator):
#     """Return estimate of energy from estimator

#     Parameters:
#         params (ndarray): Array of ansatz parameters
#         ansatz (QuantumCircuit): Parameterized ansatz circuit
#         hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
#         estimator (EstimatorV2): Estimator primitive instance
#         cost_history_dict: Dictionary for storing intermediate results

#     Returns:
#         float: Energy estimate
#     """
#     pub = (ansatz, [hamiltonian], [params])
#     result = estimator.run(pubs=[pub]).result()
#     energy = result[0].data.evs[0]

#     cost_history_dict["iters"] += 1
#     cost_history_dict["prev_vector"] = params
#     cost_history_dict["cost_history"].append(energy)
#     print(f"Iters. done: {cost_history_dict['iters']} [Current cost: {energy}]")

#     return energy

def to_bitstring(integer, num_bits):
    result = np.binary_repr(integer, width=num_bits)
    return [int(digit) for digit in result]

def run_ma_qaoa(circuits_dir, quadratic_program, reps=3, shots=10000, param_max_iter=100000000, maxfev=1000):
    print(f"Running MA-QAOA for {quadratic_program.name}...")
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
    print(f"Number of variables: {num_vars}")

    qubitOp, offset = qubo.to_ising()

    print(f"Number of repetitions: {reps}")

    """
    Make the circuit based on Pauli Z interactions, and
    assign each of the gate a separate parameter
    """
    # print("Qubit Operator:", qubitOp)
    # Separate the SparsePauliOp into Pauli strings and coefficients
    pauli_strings = [str(op) for op in qubitOp.paulis]
    coefficients = [float(coeff.real) for coeff in qubitOp.coeffs]  # Use .real since the coefficients are complex but with no imaginary part

    # The list `pauli_strings` already contains all the Pauli strings
    print(f"Total number of Pauli strings: {len(pauli_strings)}")
    print(f"Pauli strings: {pauli_strings}")
    print(f"Coefficients: {coefficients}")

    ma_qaoa_circuit = build_ma_qaoa_circuit(pauli_strings, coefficients, reps)

    ma_qaoa_circuit = ma_qaoa_circuit.decompose(reps=2)

    num_params = ma_qaoa_circuit.num_parameters
    print("Number of parameters:", num_params)
    print('Number of qubits:', ma_qaoa_circuit.num_qubits)
    print('Circuit depth:', ma_qaoa_circuit.depth())
    print('Gate counts:', dict(ma_qaoa_circuit.count_ops()))
    # # print new line
    print()
    print("-----------------------------------------------------")

    cost_history_dict = {
        "prev_vector": None,
        "iters": 0,
        "cost_history": [],
    }

    x0 = 2 * np.pi * np.random.random(num_params)
    objective_func_vals = [] # Store the objective function values

    optimization_time_start = time.time()
   
    result = minimize(
        cost_func_exact,
        x0,
        args=(cost_history_dict, objective_func_vals, ma_qaoa_circuit, qubitOp, estimator),
        method="Powell",
        options={"maxiter": param_max_iter, "maxfev": maxfev},
        tol=1e-3
    )

    optimization_time_end = time.time()

    # Number of objective function evaluations performed by the optimizer
    function_evals = getattr(result, "nfev", cost_history_dict["iters"])

    # post processing
    optimized_params = result.x
    optimized_circuit = ma_qaoa_circuit.assign_parameters(optimized_params)
    optimized_circuit.measure_all()
    optimized_circuit.draw('mpl', idle_wires=False,fold=-1)

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

    print()
    print("-----------------------------------------------------")
    print("Optimization time:", optimization_time_end - optimization_time_start, "seconds")
    print("Circuit execution time:", circuit_time_end - circuit_time_start, "seconds")
    print("Total time taken:", full_end_time - full_start_time, "seconds")

    # Save the optimized MA-QAOA circuit as a .qpy file
    qpy_file_path = os.path.join(circuits_dir, f"{quadratic_program.name}_ma_qaoa_reps{reps}.qpy")

    if os.path.exists(qpy_file_path):
        print(f"Warning: {qpy_file_path} already exists and will be overwritten.")
        os.remove(qpy_file_path)  # Remove existing file if it exists
    
    with open(qpy_file_path, "wb") as qpy_file:
        qpy.dump(optimized_circuit, qpy_file)
    print(f"MA-QAOA circuit for {quadratic_program.name} with reps={reps} saved to {qpy_file_path}.")
    print()

    # Reverse the final distribution list of binary string key pairs
    final_distribution_bin_reversed = {key[::-1]: val for key, val in final_distribution_bin.items()}

    # Calculate optimization and solve times
    optimization_time = optimization_time_end - optimization_time_start
    circuit_time = circuit_time_end - circuit_time_start
    total_time = full_end_time - full_start_time

    return final_distribution_bin_reversed, optimized_params, shots, function_evals, optimization_time, circuit_time, total_time