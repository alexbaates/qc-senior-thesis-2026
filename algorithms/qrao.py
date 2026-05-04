# This file is based on code from https://github.com/SMU-Quantum/quantum-optimization-algorithms,
# licensed under the MIT License (Copyright (c) 2025 Monit Sharma).
# Modifications have been made by Alexandra Xiulan Bates, 2026.

# basic imports
import time
# import warnings
# warnings.filterwarnings("ignore")
import qiskit 
import os

# quantum imports
from qiskit_optimization.algorithms import CplexOptimizer
from qiskit_algorithms import VQE
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_algorithms.optimizers import POWELL
from qiskit import qpy
from qiskit.circuit.library import EfficientSU2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer.primitives import EstimatorV2, SamplerV2
from qiskit_aer import AerSimulator

# qrao imports
from qiskit_optimization.algorithms.qrao import QuantumRandomAccessEncoding
from qiskit.circuit.library import EfficientSU2
from qiskit_optimization.algorithms.qrao import (
    QuantumRandomAccessOptimizer
)
from qiskit_optimization.algorithms.qrao import MagicRounding


def run_qrao(circuits_dir, quadratic_program, reps=5, shots=10000, param_max_iter=100000000, maxfev=1000):
    print(f"Running QRAO for {quadratic_program.name}...")
    print()
    full_start_time = time.time()

    backend = AerSimulator(method='matrix_product_state')
    estimator = EstimatorV2.from_backend(backend)
    sampler = SamplerV2.from_backend(backend)
    sampler.__dict__["default_shots"] = shots
    pass_manager = generate_preset_pass_manager(backend=backend)

    converter = QuadraticProgramToQubo()
    qubo = converter.convert(quadratic_program)
    print("QUBO:")
    print(qubo)
    print()

    num_vars = qubo.get_num_vars()
    print('Number of variables:', num_vars)
    print('Number of repetitions:', reps)

    # converting hamiltonian
    encoding = QuantumRandomAccessEncoding(max_vars_per_qubit=3)
    encoding.encode(qubo)

    print(
        "We achieve a compression ratio of "
        f"({encoding.num_vars} binary variables : {encoding.num_qubits} qubits) "
        f"= {encoding.compression_ratio}.\n"
    )

    ansatz = EfficientSU2(num_qubits=encoding.num_qubits,entanglement='linear', reps=reps)
    ansatz = ansatz.decompose(reps=2)
    print('Number of qubits:', ansatz.num_qubits)
    print('ansatz depth:', ansatz.depth())
    print('Gate counts:', dict(ansatz.count_ops()))
    vqe = VQE(
        ansatz=ansatz,
        optimizer=POWELL(maxiter=param_max_iter, maxfev=maxfev),
        estimator=estimator,
    )

    # Use magic rounding
    magic_rounding = MagicRounding(sampler=sampler, pass_manager=pass_manager)

    # Construct the optimizer
    qrao = QuantumRandomAccessOptimizer(min_eigen_solver=vqe, rounding_scheme=magic_rounding)

    solve_time_start = time.time()
    results = qrao.solve(qubo)
    solve_time_end = time.time()

    # Extract optimizer evaluation count from the relaxed VQE solve, if available.
    relaxed_result = getattr(results, "relaxed_result", None)
    optimizer_result = getattr(relaxed_result, "optimizer_result", None)
    function_evals = getattr(optimizer_result, "nfev", None)

    # print(
    #     f"The objective function value: {results.fval}\n"
    #     f"x: {results.x}\n"
    #     f"relaxed function value: {-1 * results.relaxed_fval}\n"
    # )

    # Extract the x values from the top 10 samples
    # top_x_values = [sample.x.tolist() for sample in results.samples[:10]]
    # top_x_probabilities = [sample.probability for sample in results.samples[:10]]


    # top_x = [sample.x for sample in results.samples[:10]]
    # top_x_probabilities = [sample.probability for sample in results.samples[:10]]

    # for i in range(len(top_x)):
    #     print(f"Top {i+1} x value: {top_x[i]}, probability: {top_x_probabilities[i]}")
    # print()


    # for i, bitlist in enumerate(top_x_values):
    #     # Convert the QUBO bitstring to the original problem
    #     x = converter.interpret(bitlist)
    #     print(f"Top {i+1} Interpreted result: {x}")
        
    #     # Check if it's feasible
    #     is_feasible = quadratic_program.is_feasible(x)
    #     print(f"Is the result feasible? {is_feasible}")
        
    #     # Get the market share cost from this x
    #     cost = quadratic_program.objective.evaluate(x)
    #     print(f"Cost of the interpreted result: {cost}")
    #     print("-" * 50)  # Separator for readability

    print("QRAO Finished")

    print("----------------------------------------------")


    # Extract the final distribution over bitstrings from the samples
    final_distribution_bin = {}
    for sample in results.samples:
        bitstring = ''.join(str(int(b)) for b in sample.x)
        final_distribution_bin[bitstring] = sample.probability

    print("Probability distribution (top 10 results):")
    for bitstring, prob in sorted(final_distribution_bin.items(), key=lambda x: -x[1])[:10]:
        print(f"  {bitstring}: {prob:.6f}")

    # Extract optimized parameters and construct the QRAO circuit
    optimized_params = results.relaxed_result.optimal_point
    optimized_circuit = ansatz.assign_parameters(optimized_params)
    optimized_circuit.measure_all()

    full_end_time = time.time()
    print(f"Solve time: {solve_time_end - solve_time_start:.2f} seconds")
    print(f"Total execution time: {full_end_time - full_start_time:.2f} seconds")

    # Save the QRAO circuit as a .qpy file
    qpy_file_path = os.path.join(circuits_dir, f"{quadratic_program.name}_qrao_reps{reps}.qpy")

    if os.path.exists(qpy_file_path):
        print(f"Warning: {qpy_file_path} already exists and will be overwritten.")
        os.remove(qpy_file_path)  # Remove existing file if it exists
    
    with open(qpy_file_path, "wb") as qpy_file:
        qpy.dump(optimized_circuit, qpy_file)
    print(f"QRAO circuit for {quadratic_program.name} with reps={reps} saved to {qpy_file_path}.")
    print()

    # Calculate optimization and solve times
    solve_time = solve_time_end - solve_time_start
    total_time = full_end_time - full_start_time

    return final_distribution_bin, optimized_params, shots, function_evals, solve_time, total_time