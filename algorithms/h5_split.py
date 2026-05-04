import argparse
import os
import random
import string
import sys

import h5py


def read_problem_names(txt_file_path):
	names = []
	seen = set()

	with open(txt_file_path, "r", encoding="utf-8") as txt_file:
		for raw_line in txt_file:
			stripped = raw_line.strip()
			if not stripped or stripped.startswith("#"):
				continue
			if stripped not in seen:
				names.append(stripped)
				seen.add(stripped)

	if not names:
		raise ValueError(f"The txt file '{txt_file_path}' does not contain any valid problem names.")

	return names


def validate_requested_names(source_names, requested_names, txt_file_path):
	source_name_set = set(source_names)
	missing_names = [name for name in requested_names if name not in source_name_set]

	if missing_names:
		missing_display = ", ".join(missing_names)
		raise ValueError(
			f"The following names from '{txt_file_path}' were not found in the source HDF5 file: {missing_display}"
		)


def split_evenly(items, n):
	base_size = len(items) // n
	remainder = len(items) % n

	splits = []
	start = 0
	for index in range(n):
		chunk_size = base_size + (1 if index < remainder else 0)
		end = start + chunk_size
		splits.append(items[start:end])
		start = end

	return splits


def random_suffix(length=8):
	alphabet = string.ascii_letters + string.digits
	return "".join(random.choices(alphabet, k=length))


def copy_root_structure(source_file, destination_file, selected_problem_names=None):
	selected_problem_names = set(selected_problem_names or [])

	for key, value in source_file.attrs.items():
		destination_file.attrs[key] = value

	for top_level_name, top_level_item in source_file.items():
		if top_level_name != "docking_problems":
			source_file.copy(top_level_name, destination_file, name=top_level_name)
			continue

		destination_problems_group = destination_file.create_group("docking_problems")
		for key, value in top_level_item.attrs.items():
			destination_problems_group.attrs[key] = value

		for problem_name in top_level_item.keys():
			if problem_name in selected_problem_names:
				top_level_item.copy(problem_name, destination_problems_group, name=problem_name)


def write_split_file(source_prob_file_path, save_dir, output_basename, split_index, selected_problem_names):
	while True:
		random_tag = random_suffix()
		output_filename = f"{output_basename}_split{split_index + 1}_{random_tag}.h5"
		output_path = os.path.join(save_dir, output_filename)
		if not os.path.exists(output_path):
			break

	with h5py.File(source_prob_file_path, "r") as source_file, h5py.File(output_path, "w") as destination_file:
		if "docking_problems" not in source_file:
			raise ValueError("The source HDF5 file does not contain a 'docking_problems' group.")

		copy_root_structure(source_file, destination_file, selected_problem_names=selected_problem_names)

	return output_path


def main():
	parser = argparse.ArgumentParser(
		description="Split an HDF5 problem file into N smaller files after excluding problem names from a txt file."
	)
	parser.add_argument("--probfile", type=str, required=True, help="Path to the source .h5 problems file.")
	parser.add_argument("--txtfile", type=str, help="Optional txt file with problem names to exclude.")
	parser.add_argument("--save", type=str, required=True, help="Directory to save the split .h5 files.")
	parser.add_argument("--n", type=int, required=True, help="Number of output files to create.")
	args = parser.parse_args()

	source_prob_file_path = args.probfile
	txt_file_path = args.txtfile
	save_dir = args.save
	n = args.n

	if not os.path.isfile(source_prob_file_path):
		raise FileNotFoundError(f"The source HDF5 file '{source_prob_file_path}' does not exist.")
	if not source_prob_file_path.endswith(".h5"):
		raise ValueError("The --probfile argument must point to a .h5 file.")
	if n <= 0:
		raise ValueError("The --n argument must be a positive integer.")

	os.makedirs(save_dir, exist_ok=True)

	if txt_file_path is None:
		requested_names = []
	else:
		if not os.path.isfile(txt_file_path):
			raise FileNotFoundError(f"The txt file '{txt_file_path}' does not exist.")
		requested_names = read_problem_names(txt_file_path)

	with h5py.File(source_prob_file_path, "r") as source_file:
		if "docking_problems" not in source_file:
			raise ValueError("The source HDF5 file does not contain a 'docking_problems' group.")

		source_problems_group = source_file["docking_problems"]
		source_problem_names = list(source_problems_group.keys())

	if txt_file_path is not None:
		validate_requested_names(source_problem_names, requested_names, txt_file_path)

	requested_name_set = set(requested_names)
	remaining_problem_names = [name for name in source_problem_names if name not in requested_name_set]
	problem_splits = split_evenly(remaining_problem_names, n)

	output_name_base = os.path.splitext(os.path.basename(source_prob_file_path))[0]
	output_paths = []

	for split_index, split_problem_names in enumerate(problem_splits):
		output_path = write_split_file(
			source_prob_file_path=source_prob_file_path,
			save_dir=save_dir,
			output_basename=output_name_base,
			split_index=split_index,
			selected_problem_names=split_problem_names,
		)
		output_paths.append(output_path)

	print(f"Created {len(output_paths)} split HDF5 file(s) in '{save_dir}'.")
	for path in output_paths:
		print(path)


if __name__ == "__main__":
	try:
		main()
	except Exception as exc:
		print(str(exc), file=sys.stderr)
		sys.exit(1)
