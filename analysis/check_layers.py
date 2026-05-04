import argparse
import glob
import os
from qiskit import qpy

ALGORITHMS = ["cvar_qaoa", "cvar_vqe", "ma_qaoa", "qaoa", "qrao", "vqe", "ws_qaoa"]
REPS = [1, 2, 3]


def main():
    parser = argparse.ArgumentParser(
        description="Report max circuit depth per (algorithm, reps) directory."
    )
    parser.add_argument("--inputdir", required=True,
                        help="Root directory containing algorithm subdirectories.")
    args = parser.parse_args()

    print(f"{'Algorithm':<12} {'Reps':<6} {'Max depth':>10} {'N circuits':>12} {'Circuit with max depth'}")
    print("-" * 90)

    for alg in ALGORITHMS:
        for reps in REPS:
            reps_dir = os.path.join(args.inputdir, alg, f"reps{reps}")
            if not os.path.isdir(reps_dir):
                continue

            qpy_files = sorted(glob.glob(os.path.join(reps_dir, "*.qpy")))
            if not qpy_files:
                print(f"{alg:<12} {reps:<6} {'(no .qpy files)':>10}")
                continue

            max_depth = -1
            max_file = None
            n_circuits = 0

            for qpy_path in qpy_files:
                with open(qpy_path, "rb") as f:
                    circuits = qpy.load(f)
                for circuit in circuits:
                    depth = circuit.depth()
                    n_circuits += 1
                    if depth > max_depth:
                        max_depth = depth
                        max_file = os.path.basename(qpy_path)

            print(f"{alg:<12} {reps:<6} {max_depth:>10} {n_circuits:>12}   {max_file}")


if __name__ == "__main__":
    main()
