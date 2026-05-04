import argparse
import os
import traceback
import h5py
from qiskit import qpy

def print_circuit_from_qpy(file_path):
    with open(file_path, "rb") as qpy_file:
        circuit = qpy.load(qpy_file)[0]

    print(f"Circuit name: {circuit.name}")
    print(f"Number of qubits: {circuit.num_qubits}")
    print(f"Circuit depth: {circuit.depth()}")
    print(f"Gate counts: {dict(circuit.count_ops())}")
    print("\n--- Circuit Diagram ---")
    print(circuit)
    print("\n--- Circuit Details ---")
    for i, instruction in enumerate(circuit.data):
        gate = instruction.operation
        qubits = instruction.qubits
        print(f"Gate {i}: {gate.name} on qubits {qubits}")
    print()

def unique_rotation_gate_parameters(file_path):
    with open(file_path, "rb") as qpy_file:
        circuit = qpy.load(qpy_file)[0]
    
    rotation_gates = {'rx', 'ry', 'rz', 'rxx', 'ryy', 'rzz', 'rxyz', 'u1', 'u2', 'u3'}
    unique_params = set()
    
    for instruction in circuit.data:
        gate = instruction.operation
        if gate.name.lower() in rotation_gates:
            params = gate.params if hasattr(gate, 'params') else []
            for param in params:
                # Convert to string with high precision to handle floating point comparison
                param_str = f"{float(param):.15g}"
                unique_params.add(param_str)
    
    return len(unique_params)

def print_rotation_gate_parameters(file_path):
    with open(file_path, "rb") as qpy_file:
        circuit = qpy.load(qpy_file)[0]
    
    # List of rotation gate names to look for
    rotation_gates = {'rx', 'ry', 'rz', 'rxx', 'ryy', 'rzz', 'rxyz', 'u1', 'u2', 'u3'}
    
    print(f"Rotation Gate Parameters for: {circuit.name}")
    print("-" * 60)
    
    gate_index = 0
    for i, instruction in enumerate(circuit.data):
        gate = instruction.operation
        gate_name = gate.name.lower()
        
        # Check if this is a rotation gate
        if gate_name in rotation_gates:
            qubits = [q for q in instruction.qubits]
            params = gate.params if hasattr(gate, 'params') else []
            
            print(f"Gate {gate_index} (instruction {i}):")
            print(f"  Type: {gate.name}")
            print(f"  Qubits: {qubits}")
            print(f"  Parameters: {params}")
            gate_index += 1
    
    if gate_index == 0:
        print("No rotation gates found in circuit.")
    print()
    print(f"Number of unique rotation gate parameters: {unique_rotation_gate_parameters(file_path)}")
    print()


def _format_dataset_value(dataset):
    value = dataset[()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "dtype") and value.dtype.kind == "S":
        return [v.decode("utf-8") for v in value.tolist()]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _print_group_contents(group, indent=0):
    spacer = " " * indent

    if group.attrs:
        print(f"{spacer}Attributes:")
        for attr_name in sorted(group.attrs.keys()):
            print(f"{spacer}  {attr_name}: {group.attrs[attr_name]}")

    datasets = [name for name, obj in group.items() if isinstance(obj, h5py.Dataset)]
    if datasets:
        print(f"{spacer}Datasets:")
        for dataset_name in sorted(datasets):
            dataset = group[dataset_name]
            print(f"{spacer}  {dataset_name}: {_format_dataset_value(dataset)}")

    subgroups = [name for name, obj in group.items() if isinstance(obj, h5py.Group)]
    for subgroup_name in sorted(subgroups):
        print(f"{spacer}{subgroup_name}:")
        _print_group_contents(group[subgroup_name], indent=indent + 2)


def print_algorithm_results_from_h5(file_path):
    with h5py.File(file_path, "r") as res_file:
        if "results" not in res_file:
            raise Exception(f"No 'results' group found in {file_path}.")

        results_group = res_file["results"]
        print(f"Algorithm Results from: {file_path}")
        print("=" * 80)

        for problem_name in sorted(results_group.keys()):
            print(f"Problem: {problem_name}")
            problem_group = results_group[problem_name]

            for algorithm_name in sorted(problem_group.keys()):
                print(f"  Algorithm: {algorithm_name}")
                algorithm_group = problem_group[algorithm_name]
                _print_group_contents(algorithm_group, indent=4)
                print()

        print("=" * 80)
        print("Finished printing algorithm results.")
        print()

def main():
    try:
        parser = argparse.ArgumentParser(description="Print circuit (.qpy) or algorithm results (.h5).")
        input_group = parser.add_mutually_exclusive_group(required=True)
        input_group.add_argument("--circuit", type=str, help="Path to the .qpy file containing the quantum circuit.")
        input_group.add_argument("--results", type=str, help="Path to the .h5 algorithm results file.")
        args = parser.parse_args()

        if args.circuit is not None:
            circuit_path = args.circuit
            if not os.path.isfile(circuit_path):
                raise Exception(f"The circuit file '{circuit_path}' does not exist.")

            print(f"Loading circuit from: {circuit_path}")
            print()
            print_circuit_from_qpy(circuit_path)
            print_rotation_gate_parameters(circuit_path)

        if args.results is not None:
            results_path = args.results
            if not os.path.isfile(results_path):
                raise Exception(f"The results file '{results_path}' does not exist.")

            print_algorithm_results_from_h5(results_path)
    except Exception as e:
        traceback.print_exc()

if __name__ == "__main__":
    main()