"""
Validates that the algorithm name encoded in each result .h5 filename matches
the algorithm name(s) found inside the file.

Expected filename pattern:  ..._<algorithm>_reps<N>_...h5
Expected internal structure: f["results"][<problem>][<algorithm>]

"""

import argparse
import os
import re
import sys

import h5py

KNOWN_ALGORITHMS = [
    "cvar_qaoa",
    "cvar_vqe",
    "ws_qaoa",
    "ma_qaoa",
    "qaoa",
    "qrao",
    "vqe",
]

def extract_alg_from_filename(fname):
    base = os.path.splitext(fname)[0]
    # Try known algorithms (sorted longest-first to avoid partial matches)
    for alg in sorted(KNOWN_ALGORITHMS, key=len, reverse=True):
        if re.search(rf"_{re.escape(alg)}_reps\d+", base):
            return alg
    # Generic fallback
    m = re.search(r"_([a-z][a-z0-9_]*)_reps\d+", base)
    return m.group(1) if m else None


def internal_algorithms(h5_path):
    algs = set()
    with h5py.File(h5_path, "r") as f:
        if "results" not in f:
            return algs
        for prob_key in f["results"].keys():
            for alg in f["results"][prob_key].keys():
                if alg != "cplex":
                    algs.add(alg)
    return algs


def main():
    parser = argparse.ArgumentParser(description="Validate result .h5 filename vs internal algorithm names.")
    parser.add_argument("--resdir", required=True, help="Directory containing algorithm result .h5 files")
    args = parser.parse_args()

    if not os.path.isdir(args.resdir):
        parser.error(f"--resdir: '{args.resdir}' is not a directory")

    fnames = sorted(
        f for f in os.listdir(args.resdir)
        if f.endswith(".h5") and "cplex" not in f.lower()
    )
    if not fnames:
        print("No non-CPLEX .h5 files found.")
        sys.exit(0)

    n_ok = 0
    n_warn = 0
    n_err = 0

    for fname in fnames:
        h5_path = os.path.join(args.resdir, fname)
        alg_from_name = extract_alg_from_filename(fname)

        try:
            algs_in_file = internal_algorithms(h5_path)
        except Exception as e:
            print(f"[ERROR]  {fname}  — could not read file: {e}")
            n_err += 1
            continue

        if not algs_in_file:
            print(f"[WARN]   {fname}  — no algorithm groups found inside file")
            n_warn += 1
            continue

        if alg_from_name is None:
            print(f"[WARN]   {fname}  — could not parse algorithm from filename; file contains: {sorted(algs_in_file)}")
            n_warn += 1
            continue

        if alg_from_name not in algs_in_file:
            print(f"[MISMATCH] {fname}")
            print(f"           filename says : {alg_from_name}")
            print(f"           file contains : {sorted(algs_in_file)}")
            n_err += 1
        elif len(algs_in_file) > 1:
            print(f"[WARN]   {fname}  — filename says '{alg_from_name}' but file also contains: {sorted(algs_in_file - {alg_from_name})}")
            n_warn += 1
        else:
            print(f"[OK]     {fname}  — {alg_from_name}")
            n_ok += 1

    print(f"\nSummary: {n_ok} OK, {n_warn} warnings, {n_err} errors  ({len(fnames)} files checked)")
    sys.exit(0 if n_err == 0 else 1)


if __name__ == "__main__":
    main()
