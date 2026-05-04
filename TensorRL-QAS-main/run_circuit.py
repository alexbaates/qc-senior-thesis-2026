import argparse
import json
import os
import tempfile
import time
from datetime import datetime, timezone

from qiskit import qpy
from qiskit.primitives import BackendSamplerV2
from qiskit_aer import AerSimulator


def load_single_circuit(qpy_path):
	with open(qpy_path, "rb") as qpy_file:
		circuits = qpy.load(qpy_file)

	if len(circuits) != 1:
		raise ValueError(
			f"Expected exactly one circuit in '{qpy_path}', found {len(circuits)}."
		)
	return circuits[0]


def ensure_measurements(circuit):
	has_measurements = any(inst.operation.name == "measure" for inst in circuit.data)
	if not has_measurements:
		raise ValueError(
			"The provided circuit has no measurements. "
			"Please provide a measured circuit."
		)


def print_top_results(counts, top_n=5):
	total_shots = sum(counts.values())
	if total_shots == 0:
		raise ValueError("No measurement counts were returned by the sampler.")

	rows = []
	for bitstring, count in counts.items():
		reversed_bitstring = bitstring[::-1]
		probability = count / total_shots
		rows.append((reversed_bitstring, count, probability))

	rows.sort(key=lambda r: (r[2], r[1], r[0]), reverse=True)
	top_rows = rows[:top_n]

	print(f"\nTop {top_n} results (sorted by probability):")
	print(f"{'Rank':<5} {'Bitstring':<20} {'Count':>10} {'Probability':>14}")
	print("-" * 55)
	for idx, (bitstring, count, probability) in enumerate(top_rows, start=1):
		print(f"{idx:<5} {bitstring:<20} {count:>10} {probability:>14.8f}")
	print()

def save_results_json(counts, circuit_path, shots, num_qubits, circuit_time, output_dir):
	total_shots = sum(counts.values())
	results = {}
	for bitstring, count in counts.items():
		reversed_bitstring = bitstring[::-1]
		results[reversed_bitstring] = {
			"count": count,
			"probability": count / total_shots,
		}

	data = {
		"circuit_path": circuit_path,
		"shots": shots,
		"num_qubits": num_qubits,
		"circuit_time": circuit_time,
		"timestamp": datetime.now(timezone.utc).isoformat(),
		"results": results,
	}

	basename = os.path.splitext(os.path.basename(circuit_path))[0]
	filename = f"{basename}_{shots}shots.json"
	os.makedirs(output_dir, exist_ok=True)
	out_path = os.path.join(output_dir, filename)

	fd, tmp_path = tempfile.mkstemp(dir=output_dir, suffix=".json.tmp")
	try:
		with os.fdopen(fd, "w") as f:
			json.dump(data, f, indent=2)
		os.replace(tmp_path, out_path)
	except BaseException:
		os.unlink(tmp_path)
		raise

	print(f"Results saved to {out_path}")


def main():
	parser = argparse.ArgumentParser(
		description="Run a single QPY circuit with classical sampling and print top results."
	)
	parser.add_argument(
		"--circuit",
		required=True,
		help="Path to the .qpy file containing exactly one circuit.",
	)
	parser.add_argument(
		"--shots",
		type=int,
		default=10000,
		help="Number of shots for sampling (default: 10000).",
	)
	parser.add_argument(
		"--n",
		type=int,
		default=5,
		help="Number of top results to print (default: 5).",
	)
	parser.add_argument(
		"--output-dir",
		required=True,
		help="Directory to save the JSON results file.",
	)
	args = parser.parse_args()

	if args.shots <= 0:
		raise ValueError("--shots must be a positive integer.")
	if args.n <= 0:
		raise ValueError("--n must be a positive integer.")

	circuit = load_single_circuit(args.circuit)
	ensure_measurements(circuit)

	backend = AerSimulator(method="matrix_product_state")
	sampler = BackendSamplerV2(backend=backend, options={"default_shots": args.shots})

	pub = (circuit,)

	circuit_time_start = time.time()
	job = sampler.run([pub], shots=args.shots)
	circuit_time_end = time.time()

	counts = job.result()[0].data.meas.get_counts()

	circuit_time = circuit_time_end - circuit_time_start

	print_top_results(counts, top_n=args.n)
	save_results_json(counts, args.circuit, args.shots, circuit.num_qubits, circuit_time, args.output_dir)

if __name__ == "__main__":
	main()
