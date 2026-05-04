import traceback
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import CplexOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo

"""
The following module contains the DockingProblem class and functions for generating Binding
Interaction Graphs (BIGs) and quadratic programs (QPs) for docking problems.
"""

class DockingProblem:
    def __init__(self, name, protein_atoms, protein_types, protein_edm, ligand_atoms, ligand_types, ligand_edm, c, penalty):
        self.name = name
        self.protein_atoms = protein_atoms
        self.protein_types = protein_types
        self.protein_edm = protein_edm
        self.ligand_atoms = ligand_atoms
        self.ligand_types = ligand_types
        self.ligand_edm = ligand_edm
        self.c = c
        self.penalty = penalty # Negative penalty scalar


"""
Binding Interaction Graph (BIG) Generation
"""

# Inputs are protein atom and ligand atom interaction types. Returns the corresponding 
# pharmacophore potential associated with the type of protein-ligand interaction.
def find_node_weight(protein_type, ligand_type):
    potentials = {
        "hd-hd": 0.5244,
        "hd-ha": 0.6686,
        "hd-hp": 0.1453,
        "hd-ar": 0.1091,
        "ha-hd": 0.6686,
        "ha-ha": 0.5478,
        "ha-hp": 0.2317,
        "ha-ar": 0.0770,
        "hp-hd": 0.1453,
        "hp-ha": 0.2317,
        "hp-hp": 0.0504,
        "hp-ar": 0.0795,
        "ar-hd": 0.1091,
        "ar-ha": 0.0770,
        "ar-hp": 0.0795,
        "ar-ar": 0.1943,
    }
    interaction = protein_type + "-" + ligand_type
    return potentials[interaction]

# Returns True if two interactions defined by the given parameters are valid and can co-exist 
# within the BIG, and returns False otherwise.
def is_valid_interaction(d1, d2, c):
    if (abs(d1 - d2) - c) >= 0:
        return True
    else:
        return False

# Returns a list of sublists, where each sublist contains four elements that represent indices for 
# protein atom 1, protein atom 2, ligand atom 1, and ligand atom 2, respectively. This sublist 
# indicates that the two interactions between the first protein and ligand atoms and the second 
# protein and ligand atoms can exist simultaneously, and is therefore representative of an edge 
# within the BIG. 
def generate_big_interactions(protein_edm, ligand_edm, c):
    interactions = []
    for p_row in range(len(protein_edm)):
        for p_col in range(len(protein_edm[0])):
            for l_row in range(len(ligand_edm)):
                for l_col in range(len(ligand_edm[0])):
                    d1 = protein_edm[p_row][p_col]
                    d2 = ligand_edm[l_row][l_col]
                    if is_valid_interaction(d1, d2, c):
                        interactions.append([p_row, p_col, l_row, l_col])
    return interactions    

# Generates the BIG of a given docking problem. Returns the nodes, edges, and weights for the BIG.
def generate_big(dp):
    nodes = []
    for p in dp.protein_atoms:
        for l in dp.ligand_atoms:
            p_atoms = []
            for atom in p:
                p_atoms.append(atom.id)
            l_atoms = []
            for atom in l:
                l_atoms.append(atom.id)
            nodes.append([p_atoms, l_atoms])
    interactions = generate_big_interactions(dp.protein_edm, dp.ligand_edm, dp.c)
    edges = []
    for values in interactions:
        p1 = []
        for atom in dp.protein_atoms[values[0]]:
            p1.append(atom.id)
        p2 = []
        for atom in dp.protein_atoms[values[1]]:
            p2.append(atom.id)
        l1 = []
        for atom in dp.ligand_atoms[values[2]]:
            l1.append(atom.id)
        l2 = []
        for atom in dp.ligand_atoms[values[3]]:
            l2.append(atom.id)
        inter1 = [p1, l1]
        inter2 = [p2, l2]
        if inter1 != inter2:
            edges.append([nodes.index(inter1), nodes.index(inter2)])
    BIG = nx.Graph()
    for n in range(len(nodes)):
        BIG.add_node(n)
    for e in edges:
        BIG.add_edge(e[0], e[1])
    # Uncomment the following two lines to visualize BIGs (not recommended for running problems in bulk)
    # nx.draw_spring(BIG, with_labels=True, node_size=800)
    # plt.show()
    weights = []
    for p_type in dp.protein_types:
        for l_type in dp.ligand_types:
            weights.append(find_node_weight(p_type, l_type))
    return nodes, edges, weights

"""
Quadratic Program (QP) Generation
"""

# Given the nodes, edges, and weights of a BIG and the penalty value of the corresponding docking
# problem, generates and returns a quadratic program.
def generate_qp(name, nodes, edges, weights, penalty):
    quadratic_program = QuadraticProgram(f"{name}")
    N = len(nodes)
    for i in range(N):
        quadratic_program.binary_var(name=f"x{i + 1}")
    linear = {f"x{i + 1}": weights[i] for i in range(N)}
    
    # Build an undirected edge set, so (i, j) and (j, i) are treated identically
    edges_set = {tuple(sorted((int(edge[0]), int(edge[1])))) for edge in edges}
    
    quadratic = {}
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if tuple(sorted((i, j))) not in edges_set:
                quadratic[(f"x{i + 1}", f"x{j + 1}")] = -penalty
    quadratic_program.maximize(linear=linear, quadratic=quadratic)
    return quadratic_program

# Prints a given quadratic program.
def print_qp(qp):
    qp.prettyprint()

# Solves a given quadratic program using CPLEX and returns the result.
def solve_qp_with_cplex(qp):
    result = CplexOptimizer().solve(qp)
    return result