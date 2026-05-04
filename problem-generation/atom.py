import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Atom import DisorderedAtom
from Bio.PDB.Residue import DisorderedResidue

class Atom:
    def __init__(self, id, name, element, pos_x, pos_y, pos_z):
        self.id = id
        self.name = name
        self.element = element
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.pos_z = pos_z

"""
Atom Class Functions
"""

# Searches a residue for an atom with a given id, including alternate location children of 
# DisorderedAtom objects.
def search_residue_atoms(file, residue, id):
    for atom in residue:
        # Check the default atom
        if atom.get_serial_number() == id:
            return Atom(
                id=atom.get_serial_number(),
                name=atom.get_name(),
                element=atom.element,
                pos_x=atom.get_coord()[0],
                pos_y=atom.get_coord()[1],
                pos_z=atom.get_coord()[2]
            )
        # Check atoms that have alternate locations
        if isinstance(atom, DisorderedAtom):
            for altloc_key in atom.disordered_get_id_list():
                child = atom.child_dict[altloc_key]
                if child.get_serial_number() == id:
                    target = Atom(
                        id=child.get_serial_number(),
                        name=child.get_name(),
                        element=child.element,
                        pos_x=child.get_coord()[0],
                        pos_y=child.get_coord()[1],
                        pos_z=child.get_coord()[2]
                    )
                    print("Atom with altloc found in file: " + file + ": " + str(target.id) + " (altloc " + altloc_key + ") with element " + target.element + " at position (" + str(target.pos_x) + ", " + str(target.pos_y) + ", " + str(target.pos_z) + ")")
                    return target
    return None


# Takes a file path for a PDB file and a target atom id. Returns an Atom object of the target atom.
def find_atom(file, id):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('complex', file)
    for model in structure:
        for chain in model:
            for residue in chain:
                # Handle disordered residues (different residue types at the same position)
                if isinstance(residue, DisorderedResidue):
                    for child_resname in residue.disordered_get_id_list():
                        child_residue = residue.child_dict[child_resname]
                        result = search_residue_atoms(file, child_residue, id)
                        if result is not None:
                            print("Atom in disordered residue found in file " + file + ": " + str(result.id) + " with element " + result.element + " at position (" + str(result.pos_x) + ", " + str(result.pos_y) + ", " + str(result.pos_z) + ")")
                            return result
                else:
                    result = search_residue_atoms(file, residue, id)
                    if result is not None:
                        return result

# Input is an array of atoms. Returns a 2D array representing the Euclidean distance matrix
def euclidean_distance_matrix(atoms):
    n = len(atoms)  # Total number of atoms
    positions = []
    matrix = []
    for atom in atoms:
        if len(atom) == 1:
            # Regular atom case (hydrogen donors/acceptors, hydrophobic contacts)
            pos_x = float(atom[0].pos_x)
            pos_y = float(atom[0].pos_y)
            pos_z = float(atom[0].pos_z)
            atom_pos = [pos_x, pos_y, pos_z]
            positions.append(atom_pos)
        else:
            # Pseudo-atom case for aromatic rings
            atoms_pos = []
            for a in atom:
                pos_x = float(a.pos_x)
                pos_y = float(a.pos_y)
                pos_z = float(a.pos_z)
                atoms_pos.append([pos_x, pos_y, pos_z])
            atoms_pos = np.array(atoms_pos)
            center_pos = atoms_pos.mean(axis=0)
            positions.append(center_pos)
    positions = np.array(positions)
    for i in range(n):
        row = []
        for j in range(n):
            dist = np.linalg.norm(positions[i] - positions[j])
            row.append(dist)
        matrix.append(row)
    return matrix


# Prints a given 2D euclidean distance matrix.
def display_matrix(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    for i in range(rows):
        row_str = ""
        for j in range(cols):
            row_str += str(matrix[i][j])
            row_str += "   "
        print(row_str)