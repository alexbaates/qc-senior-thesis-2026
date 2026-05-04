"""
Checks how many unique Hamiltonians are in a directory of .npz files produced
by qubo_to_ising.py. Uniqueness is checked two ways:
  1. By DP id in the filename (e.g. DP12_1A0I).
  2. By the sorted eigenvalue spectrum (rounded to 6 decimal places), which is
     a robust fingerprint — two Hamiltonians are the same iff they share an
     identical spectrum.
"""

import argparse
import hashlib
import os
import re
import numpy as np


DP_RE = re.compile(r'(DP\d+_[A-Z0-9]+)', re.IGNORECASE)


def extract_dp_id(filename):
    m = DP_RE.search(os.path.basename(filename))
    return m.group(1).upper() if m else None


# Hash the sorted, rounded eigenvalue array as a uniqueness fingerprint.
def eigval_fingerprint(eigvals, decimals=6):
    rounded = np.round(np.sort(eigvals), decimals=decimals)
    return hashlib.md5(rounded.tobytes()).hexdigest()


# Hash the sorted (pauli_string, rounded_weight) pairs as a uniqueness fingerprint.
def pauli_fingerprint(paulis, weights, decimals=6):
    pairs = sorted(zip(paulis, np.round(weights, decimals=decimals)))
    s = str(pairs).encode('utf-8')
    return hashlib.md5(s).hexdigest()


# Compare Pauli strings and weights between two .npz files.
def compare_pauli_terms(path_a, path_b, decimals=6):
    data_a = np.load(path_a, allow_pickle=True)
    data_b = np.load(path_b, allow_pickle=True)

    paulis_a = list(data_a['paulis'])
    weights_a = np.real(data_a['weights'])
    paulis_b = list(data_b['paulis'])
    weights_b = np.real(data_b['weights'])

    if len(paulis_a) != len(paulis_b):
        return False, f'Different number of Pauli terms: {len(paulis_a)} vs {len(paulis_b)}'

    pairs_a = dict(zip(paulis_a, weights_a))
    pairs_b = dict(zip(paulis_b, weights_b))

    if set(pairs_a.keys()) != set(pairs_b.keys()):
        only_a = set(pairs_a) - set(pairs_b)
        only_b = set(pairs_b) - set(pairs_a)
        return False, f'Different Pauli strings. Only in A: {only_a}. Only in B: {only_b}.'

    mismatches = []
    for pauli in pairs_a:
        wa = round(pairs_a[pauli], decimals)
        wb = round(pairs_b[pauli], decimals)
        if wa != wb:
            mismatches.append(f'  {pauli}: {wa} vs {wb}')

    if mismatches:
        return False, 'Same Pauli strings but different weights:\n' + '\n'.join(mismatches)

    return True, 'Pauli strings and weights are identical.'


def main():
    parser = argparse.ArgumentParser(
        description='Count unique Hamiltonians in a directory of .npz files.'
    )
    parser.add_argument('--inputdir', required=True,
                        help='Directory containing .npz Hamiltonian files.')
    args = parser.parse_args()

    if not os.path.isdir(args.inputdir):
        raise ValueError(f'--inputdir is not a valid directory: {args.inputdir}')

    npz_files = sorted(f for f in os.listdir(args.inputdir) if f.endswith('.npz'))
    if not npz_files:
        raise ValueError(f'No .npz files found in {args.inputdir}')

    total = len(npz_files)
    print(f'Found {total} .npz file(s) in {args.inputdir}')
    print()

    # Check unique DP ids
    dp_ids = {}
    no_id = []
    for fname in npz_files:
        dp_id = extract_dp_id(fname)
        if dp_id is None:
            no_id.append(fname)
        else:
            dp_ids.setdefault(dp_id, []).append(fname)

    duplicate_ids = {k: v for k, v in dp_ids.items() if len(v) > 1}

    print(f'Unique DP ids:  {len(dp_ids)}')
    if no_id:
        print(f'Files with no DP id: {len(no_id)}')
        for f in no_id:
            print(f'  - {f}')
    if duplicate_ids:
        print(f'Duplicate DP ids ({len(duplicate_ids)}):')
        for dp_id, files in duplicate_ids.items():
            print(f'  {dp_id}:')
            for f in files:
                print(f'    - {f}')
    else:
        print('No duplicate DP ids.')
    print()

    # Check unique eigenvalue spectra
    fingerprints = {}
    errors = []
    for fname in npz_files:
        path = os.path.join(args.inputdir, fname)
        try:
            data = np.load(path, allow_pickle=True)
            if 'eigvals' in data:
                eigvals = data['eigvals']
            elif 'hamiltonian' in data:
                eigvals = np.linalg.eigvalsh(np.real(data['hamiltonian']))
            else:
                errors.append((fname, 'No eigvals or hamiltonian key'))
                continue
            fp = eigval_fingerprint(eigvals)
            fingerprints.setdefault(fp, []).append(fname)
        except Exception as e:
            errors.append((fname, str(e)))

    unique_spectra = len(fingerprints)
    duplicate_spectra = {fp: files for fp, files in fingerprints.items() if len(files) > 1}

    print(f'Unique eigenvalue spectra: {unique_spectra}')
    if duplicate_spectra:
        print(f'Hamiltonians with identical spectra ({len(duplicate_spectra)} groups):')
        for fp, files in duplicate_spectra.items():
            print(f'  Fingerprint {fp[:8]}...:')
            for f in files:
                print(f'    - {f}')
            # Pairwise Pauli comparison within each group
            paths = [os.path.join(args.inputdir, f) for f in files]
            for j in range(len(paths)):
                for k in range(j + 1, len(paths)):
                    exact, detail = compare_pauli_terms(paths[j], paths[k])
                    label = 'IDENTICAL' if exact else 'DIFFERENT'
                    print(f'    [{label}] {files[j]} vs {files[k]}')
                    print(f'      {detail}')
    else:
        print('No duplicate spectra — all Hamiltonians are distinct.')
    print()

    if errors:
        print(f'Errors ({len(errors)}):')
        for fname, err in errors:
            print(f'  - {fname}: {err}')
    
    print(f'Summary: {total} files, {len(dp_ids)} unique DP ids, {unique_spectra} unique Hamiltonian spectra.')


if __name__ == '__main__':
    main()
