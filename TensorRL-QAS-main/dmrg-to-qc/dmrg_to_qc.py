import argparse
import logging
import os
import sys
import time

import jax
import mps2qc
import numpy as np
import tnqc_ansatze as tnqc
from qiskit import qpy, qasm2
from stiefel_opt import StiefelAdam
from qiskit.quantum_info import Statevector, Operator
from qiskit.converters import circuit_to_dag, dag_to_circuit

def trimmed_circuit(qc, max_depth):
    """
    Trim a quantum circuit to a maximum depth using its DAG representation.

    Args:
        qc (QuantumCircuit): The input quantum circuit.
        max_depth (int): The maximum allowed depth.

    Returns:
        QuantumCircuit: The trimmed quantum circuit.
    """
    # Convert the circuit to a DAG representation
    dag = circuit_to_dag(qc)
    
    # Create a new empty DAG to store the trimmed circuit
    trimmed_dag = dag.copy_empty_like()
    
    # Iterate through layers of the DAG
    layers = list(dag.layers())
    for i, layer in enumerate(layers):
        if i >= max_depth:
            break  # Stop adding layers once we reach `max_depth`
        for node in layer['graph'].nodes():
            # Check if the node is an operation node (DAGOpNode)
            if isinstance(node, dag.op_nodes().__iter__().__next__().__class__):  # Dynamically check for DAGOpNode type
                trimmed_dag.apply_operation_back(node.op, node.qargs, node.cargs)
    
    # Convert the trimmed DAG back to a QuantumCircuit
    trimmed_circuit = dag_to_circuit(trimmed_dag)
    
    return trimmed_circuit

jax.config.update("jax_enable_x64", True)

logging.basicConfig(handlers=[logging.StreamHandler(sys.stdout)],
                    format='%(asctime)s [%(module)s] %(message)s',
                    datefmt='%Y-%m-%d,%H:%M:%S',
                    level=logging.INFO)

logging.getLogger("qiskit").setLevel(logging.WARNING)
logging.getLogger("jax").setLevel(logging.WARNING)

DIV_STR = "-"*80

def main(hamiltonian_path: str,
         dmrg_opts: dict, 
         ansatz: dict, 
         optimizer_opts: dict,
         su4: int):
    """
    Main function to perform DMRG and convert the resulting MPS to a quantum circuit.
    Parameters:
        hamiltonian_path (str): Path to the file containing the Hamiltonian data in 'npz' format.
        dmrg_opts (dict): Options for the DMRG algorithm (using quimb's DMRG).
        ansatz (dict): Ansatz configuration for the quantum circuit.
        optimizer_opts (dict): Options for the optimizer used in the MPS to QC conversion.
    Returns:
        dict: A dictionary containing the following keys:
            - 'psi_dmrg': The ground state MPS obtained from DMRG.
            - 'dmrg_metadata': Metadata from the DMRG run.
            - 'qc_mps': The optimized quantum circuit tensor network.
            - 'opt_params': Optimized parameters for the quantum circuit.
            - 'loss_history': History of the loss function during optimization.
            - 'qc': The final Qiskit quantum circuit.
    """

    # Extract the Hamiltonian from the data
    data = np.load(hamiltonian_path)
    # print(data.keys())
    # exit()
    # The [::-1] is probably because of qiskit's ordering when saving Pauli strings?
    # It is needed to get the correct Hamiltonian comparing with the one in the data
    # pauli_dict = {k[::-1]: np.real_if_close(v) for k, v in zip(data['paulis'], data['weights'])}
    pauli_dict = {k: np.real_if_close(v) for k, v in zip(data['paulis'], data['weights'])}

    # Construct the MPO of the Hamiltonian
    ham_mpo, _ = mps2qc.mpo_from_paulis(pauli_dict)

    # Extract number of qubits from the provided Hamiltonian
    num_qubits = ham_mpo.L

    # Check that the MPO hamiltonian is correct
    if num_qubits < 10:
        logging.info(f" ║ H_dense - H_mpo ║  = {np.linalg.norm(ham_mpo.to_dense() - data['hamiltonian'])}")
        
    
    logging.info(DIV_STR)
    logging.info("Running DMRG")
    psi_dmrg, dmrg_metadata = mps2qc.gs_dmrg(ham_mpo, **dmrg_opts)
    logging.info(DIV_STR)
    
    logging.info(DIV_STR)
    logging.info("Fitting ansatz quantum circuit TN to DMRG-MPS")
    qc_mps, loss_history, opt_params = mps2qc.mps_to_qc(psi_dmrg, 
                                                        ansatz=ansatz,
                                                        optimizer_opts=optimizer_opts)   
    logging.info(DIV_STR)

    logging.info(DIV_STR)
    logging.info(f"Energy [dmrg] 〈H〉= {mps2qc.compute_energy(psi_dmrg, ham_mpo)}")
    logging.info(f"Energy [qc-tn]〈H〉= {mps2qc.compute_energy(qc_mps, ham_mpo)}")
    logging.info(f"Energy [qc-tn] - Energy [dmrg] 〈H〉= {abs(mps2qc.compute_energy(qc_mps, ham_mpo) - mps2qc.compute_energy(psi_dmrg, ham_mpo))}")

    logging.info(DIV_STR)
    
    logging.info(DIV_STR)
    logging.info("Building qiskit quantum circuit")
    ansatz['num_qubits'] = num_qubits

    # print(ansatz)
    if su4:
        basis_gates = ['rxx', 'ryy', 'rzz', 'rx', 'ry', 'rz']
    else:
        basis_gates = ['cx', 'rx', 'ry', 'rz']

    qc = tnqc.qiskit_circ_from_tn_params(ansatz, 
                                         opt_params,
                                         transpile_opts={'optimization_level': 3,
                                                         'basis_gates': basis_gates})
    logging.info(DIV_STR)
    # exit()
    
    
    results = {'psi_dmrg': psi_dmrg,
               'dmrg_metadata': dmrg_metadata,
               'qc_mps': qc_mps,
               'opt_params': opt_params,
               'loss_history': loss_history,
               'qc': qc}
    
    return results

def _process_one_file(hamiltonian_path, bond_dims, num_layers, init_circ_dir, overwrite,
                      index=None, total=None, SU4=0):
    """Run the full DMRG->QC pipeline on one .npz file. Saves .qpy and .qasm outputs.

    Returns (TN_init_circuit, energy_orig, energy_loaded, sanity_ok, qpy_path),
    or (None, None, None, None, qpy_path) if skipped because output already exists.
    """
    file_start_time = time.time()
    mol_config = os.path.splitext(os.path.basename(hamiltonian_path))[0]
    suffix    = "_su4" if SU4 else ""
    qpy_path  = os.path.join(init_circ_dir, f"init_{mol_config}_TNbond{bond_dims}{suffix}.qpy")
    qasm_path = os.path.join(init_circ_dir, f"init_{mol_config}_TNbond{bond_dims}{suffix}.qasm")
    prefix = f"[{index}/{total}] " if index is not None else ""

    if os.path.exists(qpy_path) and not overwrite:
        print(f"{prefix}Skipping {mol_config}: output already exists at {qpy_path}")
        print()
        return None, None, None, None, qpy_path

    if os.path.exists(qpy_path) and overwrite:
        print(f"Warning: {qpy_path} already exists and will be overwritten.")

    dmrg_opts      = {'bond_dims': [bond_dims], 'num_sweeps': 2}
    qc_tn_ansatz   = {'structure': 'brickwork', 'num_layers': num_layers}
    optimizer      = StiefelAdam(learning_rate=3e-3, beta1=0.9, beta2=0.999, eps=1e-8)
    optimizer_opts = {'method': optimizer, 'maxiter': 2000, 'tol': 1e-8}

    res = main(hamiltonian_path, dmrg_opts, qc_tn_ansatz, optimizer_opts, SU4)
    TN_init_circuit = res['qc']

    os.makedirs(init_circ_dir, exist_ok=True)
    qasm2.dump(TN_init_circuit, qasm_path)
    with open(qpy_path, "wb") as f:
        qpy.dump(TN_init_circuit, f)
    loaded_circuit = qasm2.load(qasm_path, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)

    # Sanity check: energy from original circuit vs. round-tripped circuit
    data    = np.load(hamiltonian_path, allow_pickle=True)
    ham_mat = Operator(data['hamiltonian']).reverse_qargs().to_matrix()
    state_orig   = np.asmatrix(Statevector(TN_init_circuit))
    state_loaded = np.asmatrix(Statevector(loaded_circuit))
    energy_orig   = float(np.real(((state_orig   @ ham_mat) @ state_orig.getH())[0, 0]))
    energy_loaded = float(np.real(((state_loaded @ ham_mat) @ state_loaded.getH())[0, 0]))
    sanity_ok = np.abs(energy_orig - energy_loaded) <= 1e-6

    file_time  = time.time() - file_start_time
    sanity_str = "OK" if sanity_ok else "MISMATCH"
    print(f"{prefix}{mol_config}: Gates={TN_init_circuit.count_ops()}, "
          f"Depth={TN_init_circuit.depth()}, Energy={energy_orig:.6f}, "
          f"Sanity={sanity_str} -> {qpy_path}")
    print(f"  Time: {file_time:.2f}s")
    print()

    return TN_init_circuit, energy_orig, energy_loaded, sanity_ok, qpy_path


if __name__ == '__main__':

    full_start_time = time.time()

    parser = argparse.ArgumentParser(description="DMRG to quantum circuit conversion")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--hamfile', type=str,
                       help='Path to a single .npz Hamiltonian file (single mode)')
    group.add_argument('--inputdir', type=str,
                       help='Directory of .npz Hamiltonian files to process in batch')
    parser.add_argument('--outputdir', type=str, required=True,
                        help='Directory to save output .qpy/.qasm circuit files')
    parser.add_argument('--bond', type=int, default=2,
                        help='Bond dimension for DMRG (default: 2)')
    parser.add_argument('--pqc', type=int, default=1,
                        help='Number of PQC layers (default: 1)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing .qpy/.qasm output files')
    args = parser.parse_args()

    bond_dims = args.bond
    num_layers = args.pqc
    SU4 = 0

    INIT_CIRC_DIR = args.outputdir

    if args.hamfile:
        # Process single file
        print('-x-x-x-x-x-x-x-x-')
        print(f'Hamiltonian file: {args.hamfile}')
        print(f'DMRG is generating MPS with bond dimension {bond_dims}')
        print(f'MPS is transformed into a PQC with layers {num_layers}')
        print('-x-x-x-x-x-x-x-x-')
        print()

        TN_init_circuit, energy_orig, energy_loaded, sanity_ok, qpy_path = _process_one_file(
            args.hamfile, bond_dims, num_layers, INIT_CIRC_DIR, args.overwrite, SU4=SU4
        )

        if TN_init_circuit is not None:
            print(TN_init_circuit)
            print('-------------')
            print()
            print('Gates:', TN_init_circuit.count_ops(), 'Depth:', TN_init_circuit.depth())
            print()
            print('-X-X-X-X-X- SOME SANITY CHECK -X-X-X-X-X-')
            print('THE OBTAINED ENERGY FROM (QISKIT) CIRCUIT: ', energy_orig)
            print('THE OBTAINED ENERGY FROM LOADED CIRCUIT:   ', energy_loaded)
            if sanity_ok:
                print('They are same! Everything is working as it is supposed to be!')
            else:
                print('Something is not right!')

    else:
        # Batch mode - process all .npz files in the input directory
        inputdir = args.inputdir
        if not os.path.isdir(inputdir):
            raise ValueError(f"--inputdir is not a valid directory: {inputdir}")

        npz_files = sorted(f for f in os.listdir(inputdir) if f.endswith('.npz'))
        if not npz_files:
            raise ValueError(f"No .npz files found in {inputdir}")

        total = len(npz_files)
        print()
        print(f'Processing {total} .npz files from {inputdir}')
        print(f'DMRG bond dimension: {bond_dims} | PQC layers: {num_layers}')
        print()

        failed  = []
        skipped = 0

        for i, fname in enumerate(npz_files, start=1):
            hamiltonian_path = os.path.join(inputdir, fname)
            try:
                result = _process_one_file(
                    hamiltonian_path, bond_dims, num_layers, INIT_CIRC_DIR, args.overwrite,
                    index=i, total=total, SU4=SU4
                )
                if result[0] is None:
                    skipped += 1
            except Exception as e:
                print(f"[{i}/{total}] ERROR - {fname}: {e}")
                print()
                failed.append((fname, str(e)))

        succeeded = total - len(failed) - skipped
        print()
        if failed:
            print(f"Batch complete. {succeeded}/{total} processed, {skipped} skipped, "
                  f"{len(failed)} failed:")
            print()
            for fname, err in failed:
                print(f"  - {fname}: {err}")
        else:
            print(f"Batch complete. {succeeded}/{total} processed, {skipped} skipped.")
            print()

    full_end_time = time.time()
    total_time = full_end_time - full_start_time
    print(f"Total run time: {total_time:.2f}s")

