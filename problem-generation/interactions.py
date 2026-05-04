from plip.structure.preparation import PDBComplex

# Returns the unique binding site identifier of a given ligand.
def get_bsid(ligand):
    hetid = ligand.hetid
    chain = ligand.chain
    position = ligand.position
    bsid = hetid + ":" + chain + ":" + str(position) # Unique binding site identifier
    return bsid

# Returns a list of hydrogen bond (ligand donor) interactions given PLIP hbonds_ldon interactions. 
# Returns None if there are no hbonds_ldon interactions.
def get_hbonds_ldon(interactions):
    if len(interactions) == 0:
        return None
    else:
        hbonds_ldon = []
        for hbond_ldon in interactions:
            interaction = {
                "type": "hbond_ldon",
                "protein_atoms": [hbond_ldon.a_orig_idx],
                "ligand_atoms": [hbond_ldon.d_orig_idx],
                "distance": hbond_ldon.distance_ad
            }
            hbonds_ldon.append(interaction)
        return hbonds_ldon

# Returns a list of hydrogen bond (protein donor) interactions given PLIP hbonds_pdon interactions. 
# Returns None if there are no hbonds_pdon interactions.
def get_hbonds_pdon(interactions):
    if len(interactions) == 0:
        return None
    else:
        hbonds_pdon = []
        for hbond_pdon in interactions:
            interaction = {
                "type": "hbond_pdon",
                "protein_atoms": [hbond_pdon.d_orig_idx],
                "ligand_atoms": [hbond_pdon.a_orig_idx],
                "distance": hbond_pdon.distance_ad
            }
            hbonds_pdon.append(interaction)
        return hbonds_pdon

# Returns a list of hydrophobic contacts given PLIP hydrophobic_contacts interactions. 
# Returns None if there are no hydrophobic_contacts interactions.
def get_hydrophobic_contacts(interactions):
    if len(interactions) == 0:
        return None
    else:
        hydrophobic_contacts = []
        for hydroph_interaction in interactions:
            interaction = {
                "type": "hydrophobic_contact",
                "protein_atoms": [hydroph_interaction.bsatom_orig_idx],
                "ligand_atoms": [hydroph_interaction.ligatom_orig_idx],
                "distance": hydroph_interaction.distance
            }
            hydrophobic_contacts.append(interaction)
        return hydrophobic_contacts

# Returns a list of pi stacking interactions given PLIP pistacking interactions. 
# Returns None if there are no pistacking interactions.
def get_pistacking(interactions):
    if len(interactions) == 0:
        return None
    else:
        pistacking_interactions = []
        for pistacking in interactions:
            interaction = {
                "type": "pi_stacking",
                "protein_atoms": pistacking.proteinring.atoms_orig_idx,    # already an array
                "ligand_atoms": pistacking.ligandring.atoms_orig_idx,  # already an array
                "distance": pistacking.distance
            }
            pistacking_interactions.append(interaction)
        return pistacking_interactions

# Prints the interactions and number of each interaction type from a provided dictionary of
# interactions detected from a PDB file.
def print_interactions(interactions):
    hbonds_ldon = interactions["hbonds_ldon"]
    hbonds_pdon = interactions["hbonds_pdon"]
    hydrophobic_contacts = interactions["hydrophobic_contacts"]
    pistacking_interactions = interactions["pistacking"]
    total = 0
    if hbonds_ldon is not None:
        print("Total number of hbonds_ldon:", len(hbonds_ldon))
        total += len(hbonds_ldon)
        for hbond_ldon in hbonds_ldon:
            print("     " + str(hbond_ldon))
    if hbonds_pdon is not None:
        print("Total number of hbonds_pdon:", len(hbonds_pdon))
        total += len(hbonds_pdon)
        for hbond_pdon in hbonds_pdon:
            print("     " + str(hbond_pdon))
    if hydrophobic_contacts is not None:
        print("Total number of hydrophobic_contacts:", len(hydrophobic_contacts))
        total += len(hydrophobic_contacts)
        for hydroph_interaction in hydrophobic_contacts:
            print("     " + str(hydroph_interaction))
    if pistacking_interactions is not None:
        print("Total number of pi_stacking interactions:", len(pistacking_interactions))
        total += len(pistacking_interactions)
        for pistacking in pistacking_interactions:
            print("     " + str(pistacking))
    print("Total number of interactions:", total)

# Given a PDB file, finds interactions in a given protein-ligand complex. Returns a dictionary
# of four interaction types: hydrogen bonds with ligand donors (hbonds_ldon), hydrogen bonds
# with protein donors (hbonds_pdon), hydrophobic contacts (hydrophobic_contacts), and pi-stacking 
# interactions (pistacking). Returns None if the provided PDB file contains more or less than one
# ligand, or if no interactions are found.
def find_complex_interactions(pdb_file_path):
    # Load the complex from the PDB file
    complex = PDBComplex()
    complex.load_pdb(pdb_file_path)
    ligands = complex.ligands

    # Find interactions in the complex
    if len(ligands) != 1:
        print(f"Expected exactly one ligand in the complex ({pdb_file_path}), but found {len(ligands)}")
        return None
    else:
        ligand = ligands[0]
        bsid = get_bsid(ligand)
        complex.analyze()
        interactions = complex.interaction_sets[bsid]
        if interactions.no_interactions:
            return None
        else:
            all_interactions = {}
            all_interactions["hbonds_ldon"] = get_hbonds_ldon(interactions.hbonds_ldon)
            all_interactions["hbonds_pdon"] = get_hbonds_pdon(interactions.hbonds_pdon)
            all_interactions["hydrophobic_contacts"] = get_hydrophobic_contacts(interactions.hydrophobic_contacts)
            all_interactions["pistacking"] = get_pistacking(interactions.pistacking)
            if (all_interactions["hbonds_ldon"] is None and
                all_interactions["hbonds_pdon"] is None and
                all_interactions["hydrophobic_contacts"] is None and
                all_interactions["pistacking"] is None):
                return None
            else:
                return all_interactions