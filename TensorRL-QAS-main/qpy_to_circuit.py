"""
Load and print a Qiskit circuit from a .qpy file.
"""

import argparse
import sys
from pathlib import Path
from qiskit import qpy


def load_and_print_circuit(qpy_file: str) -> None:
    """Load a .qpy file and print the circuit."""
    qpy_path = Path(qpy_file)
    
    if not qpy_path.exists():
        print(f"Error: File not found: {qpy_file}")
        sys.exit(1)
    
    if qpy_path.suffix != ".qpy":
        print("Warning: File does not have .qpy extension")
    
    try:
        with open(qpy_path, "rb") as f:
            circuits = qpy.load(f)
    except Exception as e:
        print(f"Error loading .qpy file: {e}")
        sys.exit(1)
    
    if not circuits:
        print("No circuits found in .qpy file")
        return
    
    # If multiple circuits, print all of them
    if isinstance(circuits, list):
        for i, circuit in enumerate(circuits):
            print(f"\n{'='*80}")
            print(f"Circuit {i + 1}:")
            print(f"{'='*80}")
            print(circuit)
            print(f"\nCircuit Info:")
            print(f"  - Qubits: {circuit.num_qubits}")
            print(f"  - Clbits: {circuit.num_clbits}")
            print(f"  - Depth: {circuit.depth()}")
            print(f"  - Gate counts: {dict(circuit.count_ops())}")
    else:
        # Single circuit
        print(circuits)
        print(f"\nCircuit Info:")
        print(f"  - Qubits: {circuits.num_qubits}")
        print(f"  - Clbits: {circuits.num_clbits}")
        print(f"  - Depth: {circuits.depth()}")
        print(f"  - Gate counts: {dict(circuits.count_ops())}")


def main():
    parser = argparse.ArgumentParser(
        description="Load and print a Qiskit circuit from a .qpy file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: python qpy_to_circuit.py --input circuit.qpy"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the .qpy file to load"
    )
    
    args = parser.parse_args()
    
    load_and_print_circuit(args.input)


if __name__ == "__main__":
    main()
