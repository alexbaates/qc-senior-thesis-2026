# This file is based on code from https://github.com/SMU-Quantum/quantum-optimization-algorithms,
# licensed under the MIT License (Copyright (c) 2025 Monit Sharma).
# Modifications have been made by Alexandra Xiulan Bates, 2026.

"""
Note: CPLEX does not work in this code for generating warm-start thetas on
the MVWCP problem Hamiltonians.
"""

# basic imports
import time
import os
import copy
import numpy as np
import cvxpy as cp

# quantum imports
from qiskit.circuit import Parameter,QuantumCircuit
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.problems.variable import VarType
from qiskit_optimization import QuadraticProgram
from qiskit.circuit.library import QAOAAnsatz
from qiskit import qpy
# SciPy minimizer routine
from scipy.optimize import minimize
from qiskit.primitives import BackendEstimatorV2, BackendSamplerV2
from qiskit_aer import AerSimulator
# Use scipy instead of CPLEX for non-convex relaxation
from scipy.optimize import minimize as sp_minimize


# relaxing the quadratic_program
def relax_quadratic_program(quadratic_program) -> QuadraticProgram:
    """Change all variables to continuous."""
    relaxed_quadratic_program = copy.deepcopy(quadratic_program)
    for variable in relaxed_quadratic_program.variables:
        variable.vartype = VarType.CONTINUOUS

    return relaxed_quadratic_program


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

def to_bitstring(integer, num_bits):
    result = np.binary_repr(integer, width=num_bits)
    return [int(digit) for digit in result]

def convexification(relaxed_quadratic_program):
    # Convexify the relaxed QP so a standard solver can handle it
    Q_matrix = relaxed_quadratic_program.objective.quadratic.to_array().astype(float)
    Q_sym = (Q_matrix + Q_matrix.T) / 2
    min_eig = np.linalg.eigvalsh(Q_sym)[0]

    if min_eig < 0:
        shift = -min_eig + 1e-6
        for i in range(relaxed_quadratic_program.get_num_vars()):
            var_name = relaxed_quadratic_program.variables[i].name
            relaxed_quadratic_program.objective.quadratic[var_name, var_name] += shift
            relaxed_quadratic_program.objective.linear[var_name] -= shift

    # Solve the now-convex continuous relaxation with scipy
    n = relaxed_quadratic_program.get_num_vars()
    Q_conv = relaxed_quadratic_program.objective.quadratic.to_array().astype(float)
    Q_conv = (Q_conv + Q_conv.T) / 2
    c_lin = relaxed_quadratic_program.objective.linear.to_array().astype(float)

    result_relax = sp_minimize(
        lambda x: x @ Q_conv @ x + c_lin @ x,
        x0=np.full(n, 0.5),  # start at midpoint to avoid vertex bias
        jac=lambda x: 2 * Q_conv @ x + c_lin,
        bounds=[(0, 1)] * n,
        method="L-BFGS-B",
    )

    c_stars = np.clip(result_relax.x, 0.0, 1.0)
    return c_stars

def run_ws_qaoa(circuits_dir, quadratic_program, reps=3, shots=10000, param_max_iter=100000000, maxfev=1000):
    print(f"Running WS-QAOA for {quadratic_program.name}...")
    print()
    full_start_time = time.time()

    backend = AerSimulator(method='matrix_product_state')
    estimator = BackendEstimatorV2(backend=backend)
    sampler = BackendSamplerV2(backend=backend, options={"default_shots": shots})

    # convert the quadratic_program to a qubo
    converter = QuadraticProgramToQubo()
    qubo = converter.convert(quadratic_program)
    print("QUBO:")
    print(qubo)
    print()

    num_vars = qubo.get_num_vars()
    print(f"Number of variables: {num_vars}")

    qubitOp, offset = qubo.to_ising()

    # relax the quadratic_program
    # relaxed_quadratic_program = relax_quadratic_program(qubo)

    """
    EDITED CODE for finding c_stars using scipy instead of CPLEX for non-convex relaxation
    """
    # solve it classically
    # sol = CplexOptimizer().solve(quadratic_program)  # CPLEX fails on non-convex QUADRATIC_PROGRAMs

    """
    New method using convexification of relaxed QUBO to find c_stars (different from source code)
    """

    relaxed_qubo = relax_quadratic_program(qubo)
    c_stars = convexification(relaxed_qubo)

    print("Relaxed solution (c_stars):", c_stars)

    print("Any values between 0.1 and 0.9?", any(0.1 < c < 0.9 for c in c_stars))
    print()

    """
    Warm Start QAOA
    """

    thetas = [2 * np.arcsin(np.sqrt(c_star)) for c_star in c_stars]

    init_qc = QuantumCircuit(qubitOp.num_qubits)
    for idx, theta in enumerate(thetas):
        init_qc.ry(theta, idx)

    beta = Parameter("β")

    ws_mixer = QuantumCircuit(qubitOp.num_qubits)
    for idx, theta in enumerate(thetas):
        ws_mixer.ry(-theta, idx)
        ws_mixer.rz(-2 * beta, idx)
        ws_mixer.ry(theta, idx)

    print(f"Number of repetitions: {reps}")
    circuit = QAOAAnsatz(cost_operator=qubitOp, reps=reps,initial_state=init_qc, mixer_operator=ws_mixer)

    circuit = circuit.decompose(reps=2)
    circuit.draw('mpl',fold=-1)

    print("Number of parameters:", len(circuit.parameters))

    initial_gamma = np.pi
    initial_beta = np.pi/2

    # Note: init_params assignment order has been corrected from the source code
    init_params = []
    for p in circuit.parameters:
        if p.name.startswith('β'):
            init_params.append(initial_beta)
        elif p.name.startswith('γ'):
            init_params.append(initial_gamma)

    print("Initial parameters:", init_params)

    print("Number of parameters:", len(init_params))
    # number of qubits, circuit depth, gate counts, 2 qubit gate count
    print('Number of qubits:', circuit.num_qubits)
    print('Circuit depth:', circuit.depth())
    print('Gate counts:', dict(circuit.count_ops()))
    # print new line
    print()
    print("-----------------------------------------------------")

    # cost function

    cost_history_dict = {
        "prev_vector": None,
        "iters": 0,
        "cost_history": [],
    }

    objective_func_vals = [] # Store the objective function values

    optimization_time_start = time.time()

    result = minimize(
        cost_func_estimator,
        init_params,
        args= (cost_history_dict, objective_func_vals, circuit, qubitOp, estimator),
        method="Powell",
        tol=1e-3,
        options={"maxiter": param_max_iter, "maxfev": maxfev}
    )

    optimization_time_end = time.time()

    # Number of objective function evaluations performed by the optimizer
    function_evals = getattr(result, "nfev", cost_history_dict["iters"])

    print(result)
    print()

    optimized_params = result.x

    # post processing
    optimized_circuit = circuit.assign_parameters(optimized_params)
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

    print()
    print("-----------------------------------------------------")
    print("Optimization Time:", optimization_time_end - optimization_time_start, "seconds")
    print("Circuit Execution Time:", circuit_time_end - circuit_time_start, "seconds")
    print("Total time taken:", full_end_time - full_start_time, "seconds")

    # Save the optimized WS-QAOA circuit as a .qpy file
    qpy_file_path = os.path.join(circuits_dir, f"{quadratic_program.name}_ws_qaoa_reps{reps}.qpy")

    if os.path.exists(qpy_file_path):
        print(f"Warning: {qpy_file_path} already exists and will be overwritten.")
        os.remove(qpy_file_path)  # Remove existing file if it exists
    
    with open(qpy_file_path, "wb") as qpy_file:
        qpy.dump(optimized_circuit, qpy_file)
    print(f"WS-QAOA circuit for {quadratic_program.name} with reps={reps} saved to {qpy_file_path}.")
    print()

    # Reverse the final distribution list of binary string key pairs
    final_distribution_bin_reversed = {key[::-1]: val for key, val in final_distribution_bin.items()}

    # Calculate optimization and solve times
    optimization_time = optimization_time_end - optimization_time_start
    circuit_time = circuit_time_end - circuit_time_start
    total_time = full_end_time - full_start_time

    return final_distribution_bin_reversed, optimized_params, shots, function_evals, optimization_time, circuit_time, total_time
