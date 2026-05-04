import argparse
from collections import defaultdict
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import random
import string
import traceback
import h5py
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
from scipy.spatial import ConvexHull
from interactions import find_complex_interactions
import atom as atm
from problem import DockingProblem
import problem as prob

"""
Constants
"""

QUBITS = 12 # number of qubits in the generated quantum circuits
C = 5.5   # combined epsilon and tau values
PENALTY = 6 # chosen based on [https://arxiv.org/pdf/2308.04098]
C_MIN = 0   # adjust for testing
C_MAX = 12 # adjust for testing
PENALTY_MIN = 6 # adjust for testing
PENALTY_MAX = 6 # adjust for testing

"""
Functions
"""

def write_interactions_to_hdf5(pdb_dir_path, inter_file_path, pdb_files):
    with h5py.File(inter_file_path, "w") as inter_file:
        print("Detecting and storing interactions...")

        # Create complexes group
        complexes_group = inter_file.create_group("complexes")

        # Iterate through PDB files and create Complex groups for valid complexes
        complex_id = 0
        for pdb_file in pdb_files:
            interactions = find_complex_interactions(pdb_dir_path + "/" + pdb_file)
            if interactions is not None:    # checks for valid complex
                complex_id += 1
                complex = pdb_file.removesuffix(".pdb")
                complex_group = complexes_group.create_group(complex)
                complex_group.attrs["name"] = complex
                complex_group.attrs["id"] = complex_id

                # Hydrogen Bonds (Ligand Donor)
                hbonds_ldon = interactions["hbonds_ldon"]
                hbonds_ldon_group = complex_group.create_group("hbonds_ldon")
                if hbonds_ldon is not None:
                    for i in range(len(hbonds_ldon)):
                        hbond_ldon = hbonds_ldon[i]
                        hbond_ldon_group = hbonds_ldon_group.create_group("hbond_ldon_" + str(i))
                        hbond_ldon_group.attrs["type"] = hbond_ldon["type"]
                        hbond_ldon_group.create_dataset("protein_atoms", data=np.array(hbond_ldon["protein_atoms"], dtype=np.int32))
                        hbond_ldon_group.create_dataset("ligand_atoms", data=np.array(hbond_ldon["ligand_atoms"], dtype=np.int32))
                        hbond_ldon_group.create_dataset("distance", data=float(hbond_ldon["distance"]))

                # Hydrogen Bonds (Protein Donor)
                hbonds_pdon = interactions["hbonds_pdon"]
                hbonds_pdon_group = complex_group.create_group("hbonds_pdon")
                if hbonds_pdon is not None:
                    for i in range(len(hbonds_pdon)):
                        hbond_pdon = hbonds_pdon[i]
                        hbond_pdon_group = hbonds_pdon_group.create_group("hbond_pdon_" + str(i))
                        hbond_pdon_group.attrs["type"] = hbond_pdon["type"]
                        hbond_pdon_group.create_dataset("protein_atoms", data=np.array(hbond_pdon["protein_atoms"], dtype=np.int32))
                        hbond_pdon_group.create_dataset("ligand_atoms", data=np.array(hbond_pdon["ligand_atoms"], dtype=np.int32))
                        hbond_pdon_group.create_dataset("distance", data=float(hbond_pdon["distance"]))
                
                # Hydrophobic Contacts
                hydrophobic_contacts = interactions["hydrophobic_contacts"]
                hydrophobic_contacts_group = complex_group.create_group("hydrophobic_contacts")
                if hydrophobic_contacts is not None:
                    for i in range(len(hydrophobic_contacts)):
                        hydroph_interaction = hydrophobic_contacts[i]
                        hydroph_interaction_group = hydrophobic_contacts_group.create_group("hydrophobic_contact_" + str(i))
                        hydroph_interaction_group.attrs["type"] = hydroph_interaction["type"]
                        hydroph_interaction_group.create_dataset("protein_atoms", data=np.array(hydroph_interaction["protein_atoms"], dtype=np.int32))
                        hydroph_interaction_group.create_dataset("ligand_atoms", data=np.array(hydroph_interaction["ligand_atoms"], dtype=np.int32))
                        hydroph_interaction_group.create_dataset("distance", data=float(hydroph_interaction["distance"]))
                
                # Pi-Stacking Interactions
                pistacking_interactions = interactions["pistacking"]
                pistacking_interactions_group = complex_group.create_group("pistacking")
                if pistacking_interactions is not None:
                    for i in range(len(pistacking_interactions)):
                        pistacking = pistacking_interactions[i]
                        pistacking_group = pistacking_interactions_group.create_group("pistacking_" + str(i))
                        pistacking_group.attrs["type"] = pistacking["type"]
                        pistacking_group.create_dataset("protein_atoms", data=np.array(pistacking["protein_atoms"], dtype=np.int32))
                        pistacking_group.create_dataset("ligand_atoms", data=np.array(pistacking["ligand_atoms"], dtype=np.int32))
                        pistacking_group.create_dataset("distance", data=float(pistacking["distance"]))
    print()
    print(f"Interactions detected from {complex_id} valid complexes and successfully written to '{inter_file_path}'.")
    print()

def get_complex_names_from_hdf5(inter_file_path):
    complex_names = []
    with h5py.File(inter_file_path, "r") as inter_file:
        complex_names = [complex_name for complex_name, obj in inter_file["complexes"].items() if isinstance(obj, h5py.Group)]
    return complex_names

# Returns the number of atoms that are needed from the protein and ligand for a docking problem
# with the given number of qubits.
def get_dp_config(qubits):
    dp_config = {
        6: (3, 2),
        8: (4, 2),
        12: (4, 3),
        15: (5, 3),
        20: (5,4)
    }
    return dp_config[qubits]

def get_atoms(pdb_file, target_ids):
    target_atoms = []
    for id in target_ids:
        atom = atm.find_atom(pdb_file, id)
        if atom is not None:
            target_atoms.append(atom)
        else:
            print(f"Warning: Atom with ID {id} not found in PDB file '{pdb_file}'.")
    if len(target_atoms) == len(target_ids):
        return target_atoms
    else:
        return None

# Selects protein and ligand atoms from all interaction types, sorted by distance. Returns unique 
# protein atoms and unique ligand atoms, prioritizing the shortest-distance interactions.
def select_interactions(inter_file_path, complex_name, p_atoms, l_atoms):
    all_interactions = []
    with h5py.File(inter_file_path, "r") as inter_file:
        # Hydrogen bonds (ligand donor)
        hbonds_ldon = inter_file[f"/complexes/{complex_name}/hbonds_ldon"]
        if hbonds_ldon is not None:
            for _, hbond in hbonds_ldon.items():
                all_interactions.append([
                    hbond["protein_atoms"][:],
                    hbond["ligand_atoms"][:],
                    hbond["distance"][()],
                    "ha", 
                    "hd",
                ])
        # Hydrogen bonds (protein donor)
        hbonds_pdon = inter_file[f"/complexes/{complex_name}/hbonds_pdon"]
        if hbonds_pdon is not None:
            for _, hbond in hbonds_pdon.items():
                all_interactions.append([
                    hbond["protein_atoms"][:],
                    hbond["ligand_atoms"][:],
                    hbond["distance"][()],
                    "hd", 
                    "ha",  
                ])
        # Hydrophobic contacts
        hydrophobic_contacts = inter_file[f"/complexes/{complex_name}/hydrophobic_contacts"]
        if hydrophobic_contacts is not None:
            for _, contact in hydrophobic_contacts.items():
                all_interactions.append([
                    contact["protein_atoms"][:],
                    contact["ligand_atoms"][:],
                    contact["distance"][()],
                    "hp",
                    "hp",
                ])
        # Pi-stacking — both sides are aromatic (ar)
        pistacking = inter_file[f"/complexes/{complex_name}/pistacking"]
        if pistacking is not None:
            for _, ps in pistacking.items():
                all_interactions.append([
                    ps["protein_atoms"][:],
                    ps["ligand_atoms"][:],
                    ps["distance"][()],
                    "ar",
                    "ar",
                ])

    # Sort all interactions by distance (shortest first)
    all_interactions.sort(key=lambda x: x[2])

    # Select unique protein atoms (by atom ID only)
    selected_protein_atoms = []
    i = 0
    while len(selected_protein_atoms) < p_atoms and i < len(all_interactions):
        interaction = all_interactions[i]
        new_p_atom = interaction[0]
        p_type = interaction[3]
        if not any(np.array_equal(new_p_atom, selected_atom[0]) for selected_atom in selected_protein_atoms):
            selected_protein_atoms.append((new_p_atom, p_type))
        i += 1

    # Select unique ligand atoms (by atom ID only)
    selected_ligand_atoms = []
    i = 0
    while len(selected_ligand_atoms) < l_atoms and i < len(all_interactions):
        interaction = all_interactions[i]
        new_l_atom = interaction[1]
        l_type = interaction[4]
        if not any(np.array_equal(new_l_atom, selected_atom[0]) for selected_atom in selected_ligand_atoms):
            selected_ligand_atoms.append((new_l_atom, l_type))
        i += 1
    return selected_protein_atoms, selected_ligand_atoms

def generate_docking_problem(inter_file_path, pdb_dir_path, complex_name, qubits, c, penalty):
    pdb_file_path = pdb_dir_path + "/" + complex_name + ".pdb"
    p, l = get_dp_config(qubits)

    # Select protein and ligand atoms by shortest distance across all interaction types
    selected_protein, selected_ligand = select_interactions(inter_file_path, complex_name, p, l)

    # Build atom coordinate lists from PDB file
    protein_atoms = []
    protein_types = []
    for atom_ids, ptype in selected_protein:
        protein_atom = get_atoms(pdb_file_path, atom_ids)
        if protein_atom is not None:
            protein_atoms.append(protein_atom)
            protein_types.append(ptype)

    ligand_atoms = []
    ligand_types = []
    for atom_ids, ltype in selected_ligand:
        ligand_atom = get_atoms(pdb_file_path, atom_ids)
        if ligand_atom is not None:
            ligand_atoms.append(ligand_atom)
            ligand_types.append(ltype)

    # Generate docking problem if there are enough pharmacophore points
    if len(protein_atoms) * len(ligand_atoms) == qubits:
        dp_name = "DP" + str(qubits) + "_" + complex_name
        protein_edm = atm.euclidean_distance_matrix(protein_atoms)
        ligand_edm = atm.euclidean_distance_matrix(ligand_atoms)
        docking_problem = DockingProblem(
            name=dp_name,
            protein_atoms=protein_atoms,
            protein_types=protein_types,
            protein_edm=protein_edm,
            ligand_atoms=ligand_atoms,
            ligand_types=ligand_types,
            ligand_edm=ligand_edm,
            c=c,
            penalty=penalty
        )
        return docking_problem
    else:
        return None
    
def write_problems_to_hdf5(inter_file_path, pdb_dir_path, prob_file_path, fig_dir_path, complex_names, qubits, c, penalty):
    print(f"Generating docking problems for {qubits}-qubit problems with C={c} and PENALTY={penalty}...")
    print()
    tpr_fpr_values = []
    total_problems = 0

    with h5py.File(prob_file_path, "w") as prob_file:
        docking_problems_group = prob_file.create_group("docking_problems")
        for complex_name in complex_names:
            # Generate docking problem for the complex
            docking_problem = generate_docking_problem(inter_file_path, pdb_dir_path, complex_name, qubits, c, penalty)
            if docking_problem is not None:
                total_problems += 1
                # print(f"Succesfully generated docking problem for complex {complex_name}.")

                # Create group for the complex
                docking_problem_group = docking_problems_group.create_group(docking_problem.name)

                # Generate BIG for the docking problem
                # print(f"Generating BIG for docking problem {docking_problem.name}...")
                nodes, edges, weights = prob.generate_big(docking_problem)
                # print(f"BIG generated. Generating Quadratic Program for docking problem {docking_problem.name}...")

                # Generate quadratic program for the docking problem
                quadratic_program = prob.generate_qp(docking_problem.name, nodes, edges, weights, docking_problem.penalty)
                # print("Quadratic Program successfully generated:")
                # print()
                # print(quadratic_program)
                # print()

                # Save docking problem data
                docking_problem_group.attrs["name"] = docking_problem.name
                docking_problem_group.attrs["penalty"] = docking_problem.penalty
                docking_problem_group.create_dataset("nodes", data=json.dumps(nodes), dtype=h5py.string_dtype())
                docking_problem_group.create_dataset("edges", data=np.array(edges, dtype=np.int32))
                docking_problem_group.create_dataset("weights", data=np.array(weights, dtype=np.float64))

                # Solve quadratic program with CPLEX and validate results against stored interactions
                # print("Solving Quadratic Program with CPLEX...")
                # print()
                cplex_result = prob.solve_qp_with_cplex(quadratic_program)
                interactions = get_all_interactions(inter_file_path, complex_name)
                _, tpr, fpr = validate_cplex_result(interactions, nodes, cplex_result.x)
                validation_results = {
                            "complex": complex_name,
                            "tpr": tpr,
                            "fpr": fpr
                        }
                tpr_fpr_values.append(validation_results)
    print(f"Docking problems generated for {total_problems} complexes.")
    print()
    print("Plotting TPR vs FPR for CPLEX results of all generated problems...")
    print()

    # Plot TPR vs FPR for all problems
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_chars = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    fig_path = os.path.join(fig_dir_path, f"tpr_vs_fpr_{qubits}QB_c{c}_p{penalty}_{timestamp}_{random_chars}.png")
    title = f"TPR vs FPR for {qubits}-Qubit Problems (C={c}, P={penalty})"
    plot_tpr_vs_fpr(fig_path, title, tpr_fpr_values)

    print(f"Quadratic programs for a total of {total_problems}, {qubits}-qubit problems (c = {c}, penalty = {penalty}) successfully generated and saved to {prob_file_path}.")
    print()

# Returns an array of all interactions in a given complex.
def get_all_interactions(inter_file_path, complex_name):
    interactions = []
    with h5py.File(inter_file_path, "r") as inter_file:
        # Get all atoms in hydrogen bonds (ligand donors)
        hbonds_ldon = inter_file[f"/complexes/{complex_name}/hbonds_ldon"]
        if hbonds_ldon is not None:
            for _, hbond_ldon in hbonds_ldon.items():
                hbond = [
                    hbond_ldon["protein_atoms"][:],
                    hbond_ldon["ligand_atoms"][:]
                ]
                interactions.append(hbond)
        # Get all atoms in hydrogen bonds (protein donors)
        hbonds_pdon = inter_file[f"/complexes/{complex_name}/hbonds_pdon"]
        if hbonds_pdon is not None:
            for _, hbond_pdon in hbonds_pdon.items():
                hbond = [
                    hbond_pdon["protein_atoms"][:],
                    hbond_pdon["ligand_atoms"][:]
                ]
                interactions.append(hbond)
        # Get all atoms in hydrophobic contacts
        hydrophobic_contacts = inter_file[f"/complexes/{complex_name}/hydrophobic_contacts"]
        if hydrophobic_contacts is not None:
            for _, hydroph_interaction in hydrophobic_contacts.items():
                interaction = [
                    hydroph_interaction["protein_atoms"][:],
                    hydroph_interaction["ligand_atoms"][:]
                ]
                interactions.append(interaction)
        # Get all atoms in pistacking interactions
        pistacking_interactions = inter_file[f"/complexes/{complex_name}/pistacking"]
        if pistacking_interactions is not None:
            for _, pistacking in pistacking_interactions.items():
                interaction = [
                    pistacking["protein_atoms"][:],
                    pistacking["ligand_atoms"][:]
                ]
                interactions.append(interaction)
    return interactions

"""
Testing Functions
"""

# Returns the accuracy, true positive rate (TPR), and false positive rate (FPR) of a CPLEX result
# compared to known interactions for a complex. 
def validate_cplex_result(interactions, nodes, result):
    interactions = [[array.tolist() for array in x] for x in interactions]
    total = len(result)
    correct = 0
    tp = 0  # true positives
    fn = 0  # false negatives
    tn = 0  # true negatives
    fp = 0  # false positives

    for i in range(total):
        if result[i] == 1:
            match = False
            for x in interactions:
                if nodes[i] == x:
                    correct += 1
                    tp += 1
                    match = True
                    break
            if not match:
                fp += 1
        else:
            match = False
            for x in interactions:
                if nodes[i] == x:
                    fn += 1
                    match = True
                    break
            if not match:
                correct += 1
                tn += 1
    # Calculate values for accuracy, true positive rate, and false positive rate            
    accuracy = correct / total
    tpr = (tp / (tp + fn)) if (tp + fn) > 0 else 0
    fpr = (fp / (fp + tn)) if (fp + tn) > 0 else 0
    return accuracy, tpr, fpr

# Compute the upper envelope of the ROCCH from (0,0) to (1,1). Returns the ordered upper-boundary 
# points as an Nx2 array, or None if there are fewer than 3 unique points.
def compute_upper_envelope(fpr_values, tpr_values):
    corner_points = np.array([[0, 0], [1, 1]])
    points = np.column_stack((fpr_values, tpr_values))
    all_points = np.vstack([points, corner_points])
    unique_points = np.unique(all_points, axis=0)
    if len(unique_points) < 3:
        return None
    # Check if all points are collinear (ConvexHull requires non-degenerate input)
    ref = unique_points[0]
    vecs = unique_points[1:] - ref
    # 2D cross product: v0.x * vi.y - v0.y * vi.x
    cross = vecs[0, 0] * vecs[:, 1] - vecs[0, 1] * vecs[:, 0]
    if np.all(np.abs(cross) < 1e-10):
        return None
    hull = ConvexHull(unique_points)
    # hull.vertices are CCW-ordered for 2D
    hull_pts = unique_points[hull.vertices]
    n = len(hull_pts)
    # Start at bottom-left corner: min x, then min y among ties
    min_x = hull_pts[:, 0].min()
    start_candidates = np.where(np.isclose(hull_pts[:, 0], min_x))[0]
    start_idx = start_candidates[np.argmin(hull_pts[start_candidates, 1])]
    # End at top-right corner: max x, then max y among ties
    max_x = hull_pts[:, 0].max()
    end_candidates = np.where(np.isclose(hull_pts[:, 0], max_x))[0]
    end_idx = end_candidates[np.argmax(hull_pts[end_candidates, 1])]
    # Walk CW (decrement) from start to end to trace the upper boundary
    upper = []
    i = start_idx
    while True:
        upper.append(hull_pts[i])
        if i == end_idx:
            break
        i = (i - 1) % n
    return np.array(upper)

def save_results_to_csv(c_and_penalty_results, csv_path):
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["c", "penalty", "accuracy", "tpr", "fpr"])
        writer.writeheader()
        writer.writerows(c_and_penalty_results)
    print(f"CSV saved to {csv_path}.")
    print()

def plot_tpr_vs_fpr(fig_path, title, tpr_fpr_results):
    if len(tpr_fpr_results) == 0:
        print(f"No results to plot for {title}.")
        print()
        return

    # Group results by (fpr, tpr) coordinates
    coord_groups = defaultdict(list)
    for r in tpr_fpr_results:
        coord_groups[(r["fpr"], r["tpr"])].append(r)

    # Detect labeling mode based on keys in result dicts
    has_c_and_penalty = "c" in tpr_fpr_results[0] and "penalty" in tpr_fpr_results[0]
    has_complex = "complex" in tpr_fpr_results[0]

    if has_c_and_penalty:
        all_penalties = set(r["penalty"] for r in tpr_fpr_results)
        penalty_varies = len(all_penalties) > 1

    fpr_values = [coord[0] for coord in coord_groups]
    tpr_values = [coord[1] for coord in coord_groups]
    plt.figure()
    plt.scatter(fpr_values, tpr_values, s=10)

    # Label each unique coordinate with the count
    texts = []
    for (fpr, tpr), group in coord_groups.items():
        count = len(group)
        offset_x = 0.01
        offset_y = 0.01
        texts.append(plt.text(fpr + offset_x, tpr + offset_y, str(count), fontsize=8, color='black'))
    # Use adjust_text to avoid overlap
    from adjustText import adjust_text
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    # Label each unique coordinate with c and penalty values
    # texts = []
    # for (fpr, tpr), group in coord_groups.items():
    #     if has_c_and_penalty:
    #         unique_c = sorted(set(r["c"] for r in group))
    #         c_str = ", ".join(str(v) for v in unique_c)
    #         label = f"c={{{c_str}}}"
    #         if penalty_varies:
    #             unique_p = sorted(set(r["penalty"] for r in group))
    #             p_str = ", ".join(str(v) for v in unique_p)
    #             label += f", p={{{p_str}}}"
    #     elif has_complex:
    #         unique_complexes = sorted(set(r["complex"] for r in group))
    #         label = ", ".join(unique_complexes)
    #     texts.append(plt.text(fpr, tpr, label, fontsize=6))
    # adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    # Plot ROCCH upper envelope from (0,0) to (1,1)
    envelope = compute_upper_envelope(fpr_values, tpr_values)
    if envelope is not None:
        plt.plot(envelope[:, 0], envelope[:, 1], 'r-', linewidth=1.5, label="ROCCH Envelope")
        plt.legend()
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Plot saved to {fig_path}.")
    print()

def plot_combined_tpr_vs_fpr(fig_path, title, all_tpr_fpr_results, per_complex_results):
    figures_dir = os.path.join("dev-code", "results", "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # Check if penalty varies across all results
    all_penalties = set(r["penalty"] for r in all_tpr_fpr_results)
    penalty_varies = len(all_penalties) > 1

    plt.figure()

    # Scatter all individual points
    all_fpr = [r["fpr"] for r in all_tpr_fpr_results]
    all_tpr = [r["tpr"] for r in all_tpr_fpr_results]
    plt.scatter(all_fpr, all_tpr, s=10, zorder=3)

    # Draw faint individual per-complex upper envelopes
    for complex_name, results in per_complex_results.items():
        fpr_vals = [r["fpr"] for r in results]
        tpr_vals = [r["tpr"] for r in results]
        envelope = compute_upper_envelope(fpr_vals, tpr_vals)
        if envelope is not None:
            plt.plot(envelope[:, 0], envelope[:, 1],
                     color='gray', linewidth=0.7, alpha=0.4, label=complex_name)

    # Average (FPR, TPR) per (c, penalty) across complexes
    param_groups = defaultdict(list)
    for r in all_tpr_fpr_results:
        param_groups[(r["c"], r["penalty"])].append((r["fpr"], r["tpr"]))

    avg_points = []
    avg_labels = {}  # map (avg_fpr, avg_tpr) -> list of (c, penalty) combos
    for (c, penalty), coords in param_groups.items():
        avg_fpr = np.mean([fp for fp, _ in coords])
        avg_tpr = np.mean([tp for _, tp in coords])
        avg_points.append((avg_fpr, avg_tpr))
        key = (round(avg_fpr, 8), round(avg_tpr, 8))
        if key not in avg_labels:
            avg_labels[key] = []
        avg_labels[key].append((c, penalty))

    # Label averaged points with their c (and penalty) values
    texts = []
    for (fpr, tpr), params in avg_labels.items():
        unique_c = sorted(set(c for c, _ in params))
        c_str = ", ".join(str(v) for v in unique_c)
        label = f"c={{{c_str}}}"
        if penalty_varies:
            unique_p = sorted(set(p for _, p in params))
            p_str = ", ".join(str(v) for v in unique_p)
            label += f", p={{{p_str}}}"
        texts.append(plt.text(fpr, tpr, label, fontsize=6))
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    # Compute and plot the average ROCCH upper envelope
    if avg_points:
        avg_fpr_vals = [p[0] for p in avg_points]
        avg_tpr_vals = [p[1] for p in avg_points]
        envelope = compute_upper_envelope(avg_fpr_vals, avg_tpr_vals)
        if envelope is not None:
            plt.plot(envelope[:, 0], envelope[:, 1],
                     'r-', linewidth=1.5, label="Average ROCCH Envelope")

    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Combined plot saved to {fig_path}.")
    print()

def plot_combined_metrics_vs_c(fig_path, title, tpr_fpr_results):
    # Group accuracy, TPR, and FPR by C value
    c_accuracy = defaultdict(list)
    c_tpr = defaultdict(list)
    c_fpr = defaultdict(list)
    for r in tpr_fpr_results:
        c_accuracy[r["c"]].append(r["accuracy"])
        c_tpr[r["c"]].append(r["tpr"])
        c_fpr[r["c"]].append(r["fpr"])
    sorted_c = sorted(c_accuracy.keys())
    avg_accuracies = [np.mean(c_accuracy[c]) for c in sorted_c]
    avg_tprs = [np.mean(c_tpr[c]) for c in sorted_c]
    avg_fprs = [np.mean(c_fpr[c]) for c in sorted_c]

    plt.figure()
    plt.plot(sorted_c, avg_accuracies, '-o', markersize=4, linewidth=1.5, color='red', label="Avg Accuracy")
    plt.plot(sorted_c, avg_tprs, '-o', markersize=4, linewidth=1.5, color='blue', label="Avg TPR")
    plt.plot(sorted_c, avg_fprs, '-o', markersize=4, linewidth=1.5, color='green', label="Avg FPR")

    # Find and label pairwise intersections between the three average lines
    line_pairs = [
        (avg_accuracies, avg_tprs, "Acc \u2229 TPR"),
        (avg_accuracies, avg_fprs, "Acc \u2229 FPR"),
        (avg_tprs, avg_fprs, "TPR \u2229 FPR"),
    ]
    # Alternate annotation offsets to avoid overlapping labels
    offsets = [(8, 12), (-8, -18), (8, -18)]
    intersection_idx = 0
    for y1_vals, y2_vals, pair_label in line_pairs:
        for k in range(len(sorted_c) - 1):
            # Check if the two lines cross between sorted_c[k] and sorted_c[k+1]
            d1 = y1_vals[k] - y2_vals[k]
            d2 = y1_vals[k + 1] - y2_vals[k + 1]
            if d1 * d2 < 0:  # sign change means a crossing
                # Linear interpolation to find exact crossing point
                t = d1 / (d1 - d2)
                cx = sorted_c[k] + t * (sorted_c[k + 1] - sorted_c[k])
                cy = y1_vals[k] + t * (y1_vals[k + 1] - y1_vals[k])
                offset = offsets[intersection_idx % len(offsets)]
                intersection_idx += 1
                plt.plot(cx, cy, marker='*', markersize=10, color='black', zorder=5)
                plt.annotate(f"{pair_label}\nC\u2248{cx:.3f}, y\u2248{cy:.3f}",
                             (cx, cy), textcoords="offset points", xytext=offset, fontsize=6,
                             bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

    plt.legend(fontsize=8)
    plt.xlabel("C (Threshold)")
    plt.ylabel("Value")
    plt.title(title)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Metrics vs C plot saved to {fig_path}.")
    print()

def plot_combined_metrics_vs_penalty(fig_path, title, tpr_fpr_results):
    # Group accuracy, TPR, and FPR by penalty value
    p_accuracy = defaultdict(list)
    p_tpr = defaultdict(list)
    p_fpr = defaultdict(list)
    for r in tpr_fpr_results:
        p_accuracy[r["penalty"]].append(r["accuracy"])
        p_tpr[r["penalty"]].append(r["tpr"])
        p_fpr[r["penalty"]].append(r["fpr"])
    sorted_p = sorted(p_accuracy.keys())
    avg_accuracies = [np.mean(p_accuracy[p]) for p in sorted_p]
    avg_tprs = [np.mean(p_tpr[p]) for p in sorted_p]
    avg_fprs = [np.mean(p_fpr[p]) for p in sorted_p]

    plt.figure()
    plt.plot(sorted_p, avg_accuracies, '-o', markersize=4, linewidth=1.5, color='red', label="Avg Accuracy")
    plt.plot(sorted_p, avg_tprs, '-o', markersize=4, linewidth=1.5, color='blue', label="Avg TPR")
    plt.plot(sorted_p, avg_fprs, '-o', markersize=4, linewidth=1.5, color='green', label="Avg FPR")

    # Find and label pairwise intersections between the three average lines
    line_pairs = [
        (avg_accuracies, avg_tprs, "Acc \u2229 TPR"),
        (avg_accuracies, avg_fprs, "Acc \u2229 FPR"),
        (avg_tprs, avg_fprs, "TPR \u2229 FPR"),
    ]
    offsets = [(8, 12), (-8, -18), (8, -18)]
    intersection_idx = 0
    for y1_vals, y2_vals, pair_label in line_pairs:
        for k in range(len(sorted_p) - 1):
            d1 = y1_vals[k] - y2_vals[k]
            d2 = y1_vals[k + 1] - y2_vals[k + 1]
            if d1 * d2 < 0:  # sign change means a crossing
                t = d1 / (d1 - d2)
                px = sorted_p[k] + t * (sorted_p[k + 1] - sorted_p[k])
                py = y1_vals[k] + t * (y1_vals[k + 1] - y1_vals[k])
                offset = offsets[intersection_idx % len(offsets)]
                intersection_idx += 1
                plt.plot(px, py, marker='*', markersize=10, color='black', zorder=5)
                plt.annotate(f"{pair_label}\nP\u2248{px:.3f}, y\u2248{py:.3f}",
                             (px, py), textcoords="offset points", xytext=offset, fontsize=6,
                             bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

    plt.legend(fontsize=8)
    plt.xlabel("Penalty")
    plt.ylabel("Value")
    plt.title(title)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Metrics vs Penalty plot saved to {fig_path}.")
    print()

def test_threshold_and_penalty_values(inter_file_path, pdb_dir_path, complexes, qubits, c_min, c_max, penalty_min, penalty_max):
    print("Testing threshold and penalty values...")
    print()
    print(f"Testing threshold and penalty values for C within [{c_min}, {c_max}] and PENALTY within [{penalty_min}, {penalty_max}]...")
    print()
    all_tpr_fpr_results = []
    per_complex_results = {}  # dict mapping complex name to its results
    total_problems = 0

    for complex in complexes:
        tpr_fpr_results = []
        for c in np.arange(c_min, c_max + 0.5, 0.5):
            for penalty in range(penalty_min, penalty_max + 1, 1):
                interactions = get_all_interactions(inter_file_path, complex)
                docking_problem = generate_docking_problem(inter_file_path, pdb_dir_path, complex, qubits, c, penalty)
                if docking_problem is not None:
                    total_problems += 1
                    nodes, edges, weights = prob.generate_big(docking_problem)
                    quadratic_program = prob.generate_qp(docking_problem.name, nodes, edges, weights, docking_problem.penalty)
                    cplex_result = prob.solve_qp_with_cplex(quadratic_program)
                    accuracy, tpr, fpr = validate_cplex_result(interactions, nodes, cplex_result.x)
                    result = {
                        "c": c,
                        "penalty": penalty,
                        "accuracy": accuracy,
                        "tpr": tpr,
                        "fpr": fpr
                    }
                    tpr_fpr_results.append(result)
                    all_tpr_fpr_results.append(result)

        per_complex_results[complex] = tpr_fpr_results

        # Directory names based on parameter ranges
        param_dir_name = f"c{c_min}-{c_max}_p{penalty_min}-{penalty_max}"

        # Save c and penalty values and results to CSV
        results_dir = os.path.join("dev-code", "results", param_dir_name)
        os.makedirs(results_dir, exist_ok=True)
        csv_path = os.path.join(results_dir, f"{complex}_c{c_min}-{c_max}_p{penalty_min}-{penalty_max}.csv")
        save_results_to_csv(tpr_fpr_results, csv_path)
        # Plot TPR vs FPR for the complex
        figures_dir = os.path.join("dev-code", "results", "figures", param_dir_name)
        os.makedirs(figures_dir, exist_ok=True)
        fig_path = os.path.join(figures_dir, f"{complex}_c{c_min}-{c_max}_p{penalty_min}-{penalty_max}.png")
        title = f"{complex} (C=[{c_min},{c_max}], P=[{penalty_min},{penalty_max}])"
        plot_tpr_vs_fpr(fig_path, title, tpr_fpr_results)

    param_dir_name = f"c{c_min}-{c_max}_p{penalty_min}-{penalty_max}"
    figures_dir = os.path.join("dev-code", "results", "figures", param_dir_name)
    os.makedirs(figures_dir, exist_ok=True)

    # Plot combined TPR vs FPR for all complexes (average hull)
    tpr_fpr_fig_path = os.path.join(figures_dir, f"combined_tpr_vs_fpr_c{c_min}-{c_max}_p{penalty_min}-{penalty_max}.png")
    tpr_fpr_title = f"All Complexes (C=[{c_min},{c_max}], P=[{penalty_min},{penalty_max}])"
    plot_combined_tpr_vs_fpr(tpr_fpr_fig_path, tpr_fpr_title, all_tpr_fpr_results, per_complex_results)

    # Plot combined Metrics vs C for all complexes
    metrics_c_fig_path = os.path.join(figures_dir, f"metrics_vs_c_c{c_min}-{c_max}_p{penalty_min}-{penalty_max}.png")
    metrics_c_title = f"All Complexes: Metrics vs C (P=[{penalty_min},{penalty_max}])"
    plot_combined_metrics_vs_c(metrics_c_fig_path, metrics_c_title, all_tpr_fpr_results)

    # Plot combined Metrics vs Penalty for all complexes (only when penalty varies)
    if penalty_min != penalty_max:
        metrics_penalty_fig_path = os.path.join(figures_dir, f"metrics_vs_penalty_c{c_min}-{c_max}_p{penalty_min}-{penalty_max}.png")
        metrics_penalty_title = f"All Complexes: Metrics vs Penalty (C=[{c_min},{c_max}])"
        plot_combined_metrics_vs_penalty(metrics_penalty_fig_path, metrics_penalty_title, all_tpr_fpr_results)

    print(f"Plots and CSV files generated for {total_problems} problems, with C in [{c_min}, {c_max}] and PENALTY in [{penalty_min}, {penalty_max}].")
    print()

"""
Problem Generation
"""


def main():
    print("\n------------- PROTEIN-LIGAND MOLECULAR DOCKING PROBLEM GENERATION -------------\n")
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--interactions", action="store_true", help="Run interaction detection and storage into .h5 file.")
        parser.add_argument("--testing", action="store_true", help="Run testing for threshold and penalty values and log results, including optimal values for C and PENALTY.")
        parser.add_argument("--problems", action="store_true", help="Run docking problem and quadratic program generation, storing results into .h5 file.")

        parser.add_argument("--interfile", type=str, help="Relative path of .h5 file to save/read detected interactions.")
        parser.add_argument("--pdbdir", type=str, help="Relative path of directory containing PDB files for interaction detection.")
        parser.add_argument("--probfile", type=str, help="Relative path of .h5 file to save generated problems.")
        parser.add_argument("--figdir", type=str, help="Relative path of directory to save generated figures (--problems mode only).")

        args = parser.parse_args()

        PDB_DIR_PATH = args.pdbdir   # relative path of existing directory containing the PDB files
        INTER_FILE_PATH = args.interfile    # relative path to store detected complex interactions (must be .h5 and parent directory must exist)
        PROB_FILE_PATH = args.probfile   # relative path to store generated docking problems and results (must be .h5 and parent directory must exist)
        FIG_DIR_PATH = args.figdir   # relative path to existing directory for saving generated figures (only for --problems mode)

        if INTER_FILE_PATH is not None:
            inter_file_dir = os.path.dirname(INTER_FILE_PATH)

        if PROB_FILE_PATH is not None:
            prob_file_dir = os.path.dirname(PROB_FILE_PATH)

        if args.interactions:
            print(">> Mode: Interaction Detection and Storage")
            if PDB_DIR_PATH is None or INTER_FILE_PATH is None:
                raise Exception("Both --pdbdir and --interfile arguments must be provided.")
            elif not os.path.isdir(PDB_DIR_PATH) and PDB_DIR_PATH != "":
                raise Exception("Provided PDB files directory does not exist.")
            elif not os.path.isdir(inter_file_dir) and inter_file_dir != "":
                raise Exception("Provided directory for saving interactions does not exist.")
            elif not INTER_FILE_PATH.endswith('.h5'):
                raise Exception("Only .h5 files are accepted for storing interactions.")
            else:
                print(f"PDB files from the following directory (relative path) will be used: {PDB_DIR_PATH}")
                print(f"Interactions will be saved to the following .h5 file (relative path): {INTER_FILE_PATH}")
                print()

                # Fetch PDB files
                print("Fetching PDB files...")
                print()
                pdb_directory = Path(PDB_DIR_PATH)
                pdb_files = [pdb.name for pdb in pdb_directory.glob("*.pdb") if pdb.is_file()]
                total_pdb_files = len(pdb_files)
                if total_pdb_files == 0:
                    raise Exception("No PDB files found in the provided directory. Only .pdb files are accepted.")
                else:
                    print(f"Total number of PDB files provided: {total_pdb_files}")
                    print()

                    # Write and store interactions in .h5 file
                    write_interactions_to_hdf5(PDB_DIR_PATH, INTER_FILE_PATH, pdb_files)

        if args.testing:
            print(">> Mode: Testing for Threshold and Penalty Values")
            if PDB_DIR_PATH is None or INTER_FILE_PATH is None:
                raise Exception("Both --pdbdir and --interfile arguments must be provided.")
            elif not os.path.isdir(PDB_DIR_PATH) and PDB_DIR_PATH != "":
                raise Exception("Provided PDB files directory does not exist.")
            elif not os.path.isdir(inter_file_dir) and inter_file_dir != "":
                raise Exception("Provided directory for saving interactions does not exist.")
            elif not INTER_FILE_PATH.endswith('.h5'):
                raise Exception("Only .h5 files are accepted for storing interactions.")
            else:
                print(f"Using interactions stored in the following .h5 file (relative path): {INTER_FILE_PATH}")
                print("Retrieving complex names...")
                complex_names = get_complex_names_from_hdf5(INTER_FILE_PATH)
                print(f"Total number of complexes found: {len(complex_names)}")
                test_threshold_and_penalty_values(INTER_FILE_PATH, PDB_DIR_PATH, complex_names, QUBITS, C_MIN, C_MAX, PENALTY_MIN, PENALTY_MAX)

        if args.problems:
            print(">> Mode: Docking Problem and Quadratic Program Generation")
            if PDB_DIR_PATH is None or INTER_FILE_PATH is None or PROB_FILE_PATH is None or FIG_DIR_PATH is None:
                raise Exception("All --pdbdir, --interfile, --probfile, and --figdir arguments must be provided.")
            elif not os.path.isdir(PDB_DIR_PATH) and PDB_DIR_PATH != "":
                raise Exception("Provided PDB files directory does not exist.")
            elif not os.path.isdir(inter_file_dir) and inter_file_dir != "":
                raise Exception("Provided directory for saving interactions does not exist.")
            elif not INTER_FILE_PATH.endswith('.h5'):
                raise Exception("Only .h5 files are accepted for storing interactions.")
            elif not os.path.isdir(prob_file_dir) and prob_file_dir != "":
                raise Exception("Provided directory for saving generated docking problems does not exist.")
            elif not PROB_FILE_PATH.endswith('.h5'):
                raise Exception("Only .h5 files are accepted for saving generated docking problems.")
            elif not os.path.isdir(FIG_DIR_PATH) and FIG_DIR_PATH != "":
                raise Exception("Provided directory for saving generated figures does not exist.")
            else:
                print(f"PDB files from the following directory (relative path) will be used: {PDB_DIR_PATH}")
                print(f"Interactions will be read from the following .h5 file (relative path): {INTER_FILE_PATH}")
                print(f"Generated docking problems and CPLEX results will be saved to the following .h5 file (relative path): {PROB_FILE_PATH}")
                print()

                # Fetch PDB files
                print("Fetching PDB files...")
                print()
                pdb_directory = Path(PDB_DIR_PATH)
                pdb_files = [pdb.name for pdb in pdb_directory.glob("*.pdb") if pdb.is_file()]
                total_pdb_files = len(pdb_files)
                if total_pdb_files == 0:
                    raise Exception("No PDB files found in the provided directory. Only .pdb files are accepted.")
                else:
                    print(f"Total number of PDB files provided: {total_pdb_files}")
                    print()

                    # Get complex names from interactions file
                    print("Retrieving complex names from interactions file...")
                    complex_names = get_complex_names_from_hdf5(INTER_FILE_PATH)
                    print(f"Total number of complexes found: {len(complex_names)}")
                    print()

                    # Write and store generated docking problems and results in .h5 file
                    write_problems_to_hdf5(INTER_FILE_PATH, PDB_DIR_PATH, PROB_FILE_PATH, FIG_DIR_PATH, complex_names, QUBITS, C, PENALTY)

        print("-------------------------------- END OF SCRIPT --------------------------------\n")

    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    main()
