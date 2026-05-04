import argparse
import json
import os
import traceback
import h5py

# Prints all interactions in a given interaction file.
def print_all_interactions(inter_file_path):
    with h5py.File(inter_file_path, "r") as inter_file:
        complexes_group = inter_file["complexes"]
        print(f"Total complexes: {len(complexes_group)}\n")
        
        for complex_name, complex_group in complexes_group.items():
            print(f"Interactions for complex {complex_name}:")
            
            # Interaction types to iterate through
            interaction_types = ["hbonds_ldon", "hbonds_pdon", "hydrophobic_contacts", "pistacking"]
            
            for interaction_type in interaction_types:
                if interaction_type in complex_group:
                    interactions_group = complex_group[interaction_type]
                    if len(interactions_group) > 0:
                        print(f"\n  {interaction_type.upper()}:")
                        print("  " + "-" * 50)
                        
                        for interaction_name, interaction_data in interactions_group.items():
                            print(f"    {interaction_name}:")
                            print(f"      Type: {interaction_data.attrs['type']}")
                            print(f"      Protein atoms: {interaction_data['protein_atoms'][:]}")
                            print(f"      Ligand atoms: {interaction_data['ligand_atoms'][:]}")
                            print(f"      Distance: {interaction_data['distance'][()]}")
            
            print("\n" + "=" * 60 + "\n")

# Prints interactions for a single complex.
def print_complex_interactions(inter_file_path, complex_name):
    with h5py.File(inter_file_path, "r") as inter_file:
        complexes_group = inter_file["complexes"]
        if complex_name not in complexes_group:
            print(f"Complex '{complex_name}' not found in {inter_file_path}.")
            return
        complex_group = complexes_group[complex_name]
        print(f"Interactions for complex {complex_name}:")

        interaction_types = ["hbonds_ldon", "hbonds_pdon", "hydrophobic_contacts", "pistacking"]

        for interaction_type in interaction_types:
            if interaction_type in complex_group:
                interactions_group = complex_group[interaction_type]
                if len(interactions_group) > 0:
                    print(f"\n  {interaction_type.upper()}:")
                    print("  " + "-" * 50)

                    for interaction_name, interaction_data in interactions_group.items():
                        print(f"    {interaction_name}:")
                        print(f"      Type: {interaction_data.attrs['type']}")
                        print(f"      Protein atoms: {interaction_data['protein_atoms'][:]}")
                        print(f"      Ligand atoms: {interaction_data['ligand_atoms'][:]}")
                        print(f"      Distance: {interaction_data['distance'][()]}")

        print("\n" + "=" * 60 + "\n")

def print_problems(prob_file_path):
    with h5py.File(prob_file_path, "r") as prob_file:
        problems_group = prob_file["docking_problems"]
        total = len(problems_group)
        print(f"Total complexes in file: {total}")
        print()
        for complex_name, complex_group in problems_group.items():
            name = complex_group.attrs["name"]
            penalty = complex_group.attrs["penalty"]
            nodes = json.loads(complex_group["nodes"][()])
            edges = complex_group["edges"][:].tolist()
            weights = complex_group["weights"][:].tolist()

            print(f"Complex: {complex_name}")
            print(f"  Name:    {name}")
            print(f"  Penalty: {penalty}")
            print(f"  Nodes ({len(nodes)}):")
            for i, node in enumerate(nodes):
                print(f"    [{i}] {node}")
            print(f"  Edges ({len(edges)}):")
            for i, edge in enumerate(edges):
                print(f"    [{i}] {edge}")
            print(f"  Weights ({len(weights)}):")
            for i, w in enumerate(weights):
                print(f"    [{i}] {w}")
            print()

def print_complex_names(prob_file_path):
    with h5py.File(prob_file_path, "r") as prob_file:
        problems_group = prob_file["docking_problems"]
        names = list(problems_group.keys())
        print(f"Complex names in {prob_file_path} ({len(names)} total):")
        for i in range(0, len(names), 8):
            row = names[i:i+8]
            print("  " + "  ".join(f"{name:<8}" for name in row))

def main():
    try:
        parser = argparse.ArgumentParser(description="Read and print interactions from an .h5 file.")
        parser.add_argument("--interfile", type=str, help="Path to the .h5 interactions file.")
        parser.add_argument("--probfile", type=str, help="Path to the .h5 problems file.")
        parser.add_argument("--printinteractions", nargs='?', const='all', default=None, metavar="COMPLEX", help="Print interactions. Optionally provide a complex name to print only that complex.")
        parser.add_argument("--printproblems", action="store_true", help="Print all data for each complex stored in a problem .h5 file.")
        parser.add_argument("--printnames", action="store_true", help="Print only the complex names stored in the .h5 file.")
        args = parser.parse_args()

        if args.printinteractions is not None:
            if args.interfile is None:
                raise Exception("The --interfile argument must be provided.")
            if args.printinteractions == 'all':
                print_all_interactions(args.interfile)
            else:
                print(f">> Mode: Print Interactions for Complex '{args.printinteractions}' from Interactions File")
                print_complex_interactions(args.interfile, args.printinteractions)

        elif args.printproblems:
            print(">> Mode: Print Problems from Problems File")
            if args.probfile is None:
                raise Exception("The --probfile argument must be provided.")
            elif not os.path.isfile(args.probfile):
                raise Exception("Provided problem .h5 file does not exist.")
            elif not args.probfile.endswith('.h5'):
                raise Exception("Only .h5 files are accepted.")
            else:
                print()
                print(f"Reading problems from: {args.probfile}")
                print()
                print_problems(args.probfile)

        elif args.printnames:
            print(">> Mode: Print Complex Names from Problems File")
            if args.probfile is None:
                raise Exception("The --probfile argument must be provided.")
            elif not os.path.isfile(args.probfile):
                raise Exception("Provided problem .h5 file does not exist.")
            elif not args.probfile.endswith('.h5'):
                raise Exception("Only .h5 files are accepted.")
            else:
                print()
                print(f"Reading problems from: {args.probfile}")
                print()
                print_complex_names(args.probfile)

    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    main()