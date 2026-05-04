import argparse
import csv
import datetime
import os
import uuid
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import traceback

ALGORITHM_LABELS = {
    "qaoa":      "QAOA",
    "qrao":      "QRAO",
    "vqe":       "VQE",
    "ws_qaoa":   "WS-QAOA",
    "ma_qaoa":   "MA-QAOA",
    "cvar_qaoa": "CVaR QAOA",
    "cvar_vqe":  "CVaR VQE",
    "tensorrl":  "TensorRL (trainable TN-init)",
}

def format_alg_label(alg):
    return ALGORITHM_LABELS.get(alg, alg.replace("_", "-").upper())

def load_exclude(exclude_path):
    if not exclude_path:
        return set()
    with open(exclude_path) as fh:
        return {line.strip() for line in fh if line.strip()}

def tabular_success_probabilities(outputdir, rows):
    # Group SP values by (algorithm, reps, alpha)
    groups = defaultdict(list)
    for row in rows:
        key = (row["algorithm"], row["reps"], row["alpha"])
        groups[key].append(row["success_probability"])

    # Compute stats per group
    stats = {}
    for key, sps in groups.items():
        arr = np.array(sps)
        stats[key] = (float(np.mean(arr)), float(np.median(arr)), float(np.std(arr)))

    # Sort: algorithm, then reps (None → inf), then alpha (None → inf)
    def sort_key(k):
        alg, reps, alpha = k
        return (alg, reps if reps is not None else float("inf"), alpha if alpha is not None else float("inf"))

    sorted_keys = sorted(stats.keys(), key=sort_key)

    # Ordered unique algorithm names
    seen_algs = set()
    alg_names = []
    for k in sorted_keys:
        if k[0] not in seen_algs:
            alg_names.append(k[0])
            seen_algs.add(k[0])

    def fmt(v):
        return f"{v:.3g}"

    def latex_escape(s):
        return s.replace("_", r"\_")

    lines = []
    lines.append(r"% Requires \usepackage{booktabs, multirow} in your LaTeX preamble")
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{llllll}")
    lines.append(r"\toprule")
    lines.append(r"Algorithm & Reps & $\alpha$ & Mean SP & Median SP & Std SP \\")
    lines.append(r"\midrule")

    first_alg = True
    for alg in alg_names:
        if not first_alg:
            lines.append(r"\midrule")
        first_alg = False

        alg_keys = [k for k in sorted_keys if k[0] == alg]
        alg_count = len(alg_keys)
        alg_latex = format_alg_label(alg)
        has_alpha = any(k[2] is not None for k in alg_keys)

        if has_alpha:
            # Ordered unique reps values for this algorithm
            seen_reps = set()
            reps_order = []
            for k in alg_keys:
                if k[1] not in seen_reps:
                    reps_order.append(k[1])
                    seen_reps.add(k[1])

            first_reps_group = True
            for reps in reps_order:
                reps_keys = [k for k in alg_keys if k[1] == reps]
                reps_count = len(reps_keys)

                if not first_reps_group:
                    lines.append(r"\cmidrule{2-6}")
                first_reps_group = False

                for i, key in enumerate(reps_keys):
                    mean, median, std = stats[key]
                    alpha = key[2]
                    alg_cell = f"\\multirow{{{alg_count}}}{{*}}{{{alg_latex}}}" if key == alg_keys[0] else ""
                    reps_cell = f"\\multirow{{{reps_count}}}{{*}}{{{reps}}}" if i == 0 else ""
                    alpha_cell = f"{alpha:.2f}"
                    lines.append(f"    {alg_cell} & {reps_cell} & {alpha_cell} & {fmt(mean)} & {fmt(median)} & {fmt(std)} \\\\")
        else:
            for i, key in enumerate(alg_keys):
                reps = key[1]
                mean, median, std = stats[key]
                alg_cell = f"\\multirow{{{alg_count}}}{{*}}{{{alg_latex}}}" if i == 0 else ""
                reps_cell = str(reps) if reps is not None else ""
                lines.append(f"    {alg_cell} & {reps_cell} & & {fmt(mean)} & {fmt(median)} & {fmt(std)} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Success Probabilities by Algorithm}")
    lines.append(r"\label{tab:sp}")
    lines.append(r"\end{table}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_chars = uuid.uuid4().hex[:6]
    filename = f"success_probabilities_table_{timestamp}_{random_chars}.tex"
    filepath = os.path.join(outputdir, filename)
    if os.path.exists(filepath):
        raise FileExistsError(f"Output file already exists: '{filepath}'")
    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved LaTeX table to: {filepath}")

def plot_epa_by_algorithm(outputdir, rows):
    # Group SP values by (algorithm, reps), collapsing across alpha for cvar algorithms
    groups = defaultdict(list)
    for row in rows:
        key = (row["algorithm"], row["reps"])
        groups[key].append(row["success_probability"])

    # Ordered unique algorithms and reps (None last, for algorithms with no reps concept e.g. TensorRL)
    all_algs = sorted(set(k[0] for k in groups))
    reps_set = set(k[1] for k in groups)
    all_reps = sorted(r for r in reps_set if r is not None)
    if None in reps_set:
        all_reps.append(None)

    # Compute mean and std per (algorithm, reps)
    means = {k: float(np.mean(v)) for k, v in groups.items()}
    stds  = {k: float(np.std(v))  for k, v in groups.items()}

    n_algs = len(all_algs)
    bar_width = 0.8 / n_algs
    x = np.arange(len(all_reps))
    colors = plt.cm.tab10(np.linspace(0, 0.9, n_algs))

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, alg in enumerate(all_algs):
        alg_means = [means.get((alg, r), 0.0) for r in all_reps]
        alg_stds  = [stds.get((alg, r),  0.0) for r in all_reps]
        offset = (i - n_algs / 2 + 0.5) * bar_width
        ax.bar(
            x + offset, alg_means, width=bar_width,
            yerr=alg_stds, capsize=3,
            label=format_alg_label(alg),
            color=colors[i], alpha=0.85,
        )
        # Draw a horizontal tick at the top of each bar (center of the error bar)
        existing = [(x[j] + offset, alg_means[j]) for j, r in enumerate(all_reps) if (alg, r) in means]
        if existing:
            xs, ys = zip(*existing)
            ax.plot(xs, ys, linestyle="none", marker="_", color="black",
                    markersize=bar_width * 80, markeredgewidth=1.5, zorder=5)

    ax.set_ylabel("Mean SP")
    ax.set_xticks(x)
    tick_labels = [str(r) if r is not None else "\u2014" for r in all_reps]
    ax.set_xticklabels(tick_labels)
    if any(r is not None for r in all_reps):
        ax.set_xlabel("Reps")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)
    plt.tight_layout()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_chars = uuid.uuid4().hex[:6]
    filepath = os.path.join(outputdir, f"epa_by_algorithm_{timestamp}_{random_chars}.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved algorithm bar chart to: {filepath}")

def plot_cvar_epa_by_alpha(outputdir, rows):
    cvar_algs = ["cvar_qaoa", "cvar_vqe"]

    # Group EPA values by (algorithm, reps, alpha) for cvar algorithms only
    groups = defaultdict(list)
    for row in rows:
        if row["algorithm"] in cvar_algs and row["alpha"] is not None:
            key = (row["algorithm"], row["reps"], row["alpha"])
            groups[key].append(row["success_probability"])

    if not groups:
        print("Warning: No cvar algorithm data with alpha values found; skipping alpha line plot.")
        return

    line_styles = ["-", "--", "-.", ":"]
    line_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax_idx, alg in enumerate(cvar_algs):
        ax = axes[ax_idx]
        label = chr(ord("a") + ax_idx)
        ax.set_title(f"({label}) {format_alg_label(alg)}")
        alg_keys = [k for k in groups if k[0] == alg]
        if not alg_keys:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            continue

        reps_values  = sorted(set(k[1] for k in alg_keys), key=lambda r: r if r is not None else float("inf"))
        alpha_values = sorted(set(k[2] for k in alg_keys))

        for i, reps in enumerate(reps_values):
            x_vals, y_means, y_stds = [], [], []
            for alpha in alpha_values:
                key = (alg, reps, alpha)
                if key in groups:
                    arr = np.array(groups[key])
                    x_vals.append(alpha)
                    y_means.append(float(np.mean(arr)))
                    y_stds.append(float(np.std(arr)))
            x_arr = np.array(x_vals)
            y_arr = np.array(y_means)
            y_std_arr = np.array(y_stds)
            color = line_colors[i % len(line_colors)]
            ls    = line_styles[i % len(line_styles)]
            ax.plot(x_arr, y_arr, linestyle=ls, color=color, marker="o", label=f"reps={reps}")
            ax.fill_between(x_arr, y_arr - y_std_arr, y_arr + y_std_arr, alpha=0.15, color=color)

        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("Mean SP")
        ax.legend(frameon=False)
    plt.tight_layout()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_chars = uuid.uuid4().hex[:6]
    filepath = os.path.join(outputdir, f"epa_cvar_by_alpha_{timestamp}_{random_chars}.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved CVaR alpha line plot to: {filepath}")

def plot_success_probabilities(outputdir, rows):
    plot_epa_by_algorithm(outputdir, rows)
    plot_cvar_epa_by_alpha(outputdir, rows)


def tabular_relative_solution_quality(outputdir, rows):
    # Group RSQ values by (algorithm, reps, alpha)
    groups = defaultdict(list)
    for row in rows:
        key = (row["algorithm"], row["reps"], row["alpha"])
        groups[key].append(row["relative_solution_quality"])

    # Compute stats per group
    stats = {}
    for key, rsqs in groups.items():
        arr = np.array(rsqs)
        stats[key] = (float(np.mean(arr)), float(np.median(arr)), float(np.std(arr)))

    # Sort: algorithm, then reps (None → inf), then alpha (None → inf)
    def sort_key(k):
        alg, reps, alpha = k
        return (alg, reps if reps is not None else float("inf"), alpha if alpha is not None else float("inf"))

    sorted_keys = sorted(stats.keys(), key=sort_key)

    # Ordered unique algorithm names
    seen_algs = set()
    alg_names = []
    for k in sorted_keys:
        if k[0] not in seen_algs:
            alg_names.append(k[0])
            seen_algs.add(k[0])

    def fmt(v):
        return f"{v:.3g}"

    def latex_escape(s):
        return s.replace("_", r"\_")

    lines = []
    lines.append(r"% Requires \usepackage{booktabs, multirow} in your LaTeX preamble")
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{llllll}")
    lines.append(r"\toprule")
    lines.append(r"Algorithm & Reps & $\alpha$ & Mean RSQ (\%) & Median RSQ (\%) & Std RSQ (\%) \\")
    lines.append(r"\midrule")

    first_alg = True
    for alg in alg_names:
        if not first_alg:
            lines.append(r"\midrule")
        first_alg = False

        alg_keys = [k for k in sorted_keys if k[0] == alg]
        alg_count = len(alg_keys)
        alg_latex = format_alg_label(alg)
        has_alpha = any(k[2] is not None for k in alg_keys)

        if has_alpha:
            seen_reps = set()
            reps_order = []
            for k in alg_keys:
                if k[1] not in seen_reps:
                    reps_order.append(k[1])
                    seen_reps.add(k[1])

            first_reps_group = True
            for reps in reps_order:
                reps_keys = [k for k in alg_keys if k[1] == reps]
                reps_count = len(reps_keys)

                if not first_reps_group:
                    lines.append(r"\cmidrule{2-6}")
                first_reps_group = False

                for i, key in enumerate(reps_keys):
                    mean, median, std = stats[key]
                    alpha = key[2]
                    alg_cell = f"\\multirow{{{alg_count}}}{{*}}{{{alg_latex}}}" if key == alg_keys[0] else ""
                    reps_cell = f"\\multirow{{{reps_count}}}{{*}}{{{reps}}}" if i == 0 else ""
                    alpha_cell = f"{alpha:.2f}"
                    lines.append(f"    {alg_cell} & {reps_cell} & {alpha_cell} & {fmt(mean)} & {fmt(median)} & {fmt(std)} \\\\")
        else:
            for i, key in enumerate(alg_keys):
                reps = key[1]
                mean, median, std = stats[key]
                alg_cell = f"\\multirow{{{alg_count}}}{{*}}{{{alg_latex}}}" if i == 0 else ""
                reps_cell = str(reps) if reps is not None else ""
                lines.append(f"    {alg_cell} & {reps_cell} & & {fmt(mean)} & {fmt(median)} & {fmt(std)} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Relative Solution Quality by Algorithm}")
    lines.append(r"\label{tab:rsq}")
    lines.append(r"\end{table}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_chars = uuid.uuid4().hex[:6]
    filename = f"relative_solution_quality_table_{timestamp}_{random_chars}.tex"
    filepath = os.path.join(outputdir, filename)
    if os.path.exists(filepath):
        raise FileExistsError(f"Output file already exists: '{filepath}'")
    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved LaTeX table to: {filepath}")

def plot_rsq_by_algorithm(outputdir, rows):
    # Group RSQ values by (algorithm, reps), collapsing across alpha for cvar algorithms
    groups = defaultdict(list)
    for row in rows:
        key = (row["algorithm"], row["reps"])
        groups[key].append(row["relative_solution_quality"])

    # Ordered unique algorithms and reps
    all_algs = sorted(set(k[0] for k in groups))
    all_reps = sorted(r for r in set(k[1] for k in groups) if r is not None)

    # Compute mean and std per (algorithm, reps)
    means = {k: float(np.mean(v)) for k, v in groups.items()}
    stds  = {k: float(np.std(v))  for k, v in groups.items()}

    n_algs = len(all_algs)
    bar_width = 0.8 / n_algs
    x = np.arange(len(all_reps))
    colors = plt.cm.tab10(np.linspace(0, 0.9, n_algs))

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, alg in enumerate(all_algs):
        alg_means = [means.get((alg, r), 0.0) for r in all_reps]
        alg_stds  = [stds.get((alg, r),  0.0) for r in all_reps]
        offset = (i - n_algs / 2 + 0.5) * bar_width
        ax.bar(
            x + offset, alg_means, width=bar_width,
            yerr=alg_stds, capsize=3,
            label=format_alg_label(alg),
            color=colors[i], alpha=0.85,
        )
        # Draw a horizontal tick at the top of each bar (center of the error bar)
        existing = [(x[j] + offset, alg_means[j]) for j, r in enumerate(all_reps) if (alg, r) in means]
        if existing:
            xs, ys = zip(*existing)
            ax.plot(xs, ys, linestyle="none", marker="_", color="black",
                    markersize=bar_width * 80, markeredgewidth=1.5, zorder=5)

    ax.set_xlabel("Reps")
    ax.set_ylabel("Mean RSQ (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in all_reps])
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)
    plt.tight_layout()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_chars = uuid.uuid4().hex[:6]
    filepath = os.path.join(outputdir, f"rsq_by_algorithm_{timestamp}_{random_chars}.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved RSQ bar chart to: {filepath}")

def plot_cvar_rsq_by_alpha(outputdir, rows):
    cvar_algs = ["cvar_qaoa", "cvar_vqe"]

    # Group RSQ values by (algorithm, reps, alpha) for cvar algorithms only
    groups = defaultdict(list)
    for row in rows:
        if row["algorithm"] in cvar_algs and row["alpha"] is not None:
            key = (row["algorithm"], row["reps"], row["alpha"])
            groups[key].append(row["relative_solution_quality"])

    if not groups:
        print("Warning: No cvar algorithm data with alpha values found; skipping RSQ alpha line plot.")
        return

    line_styles = ["-", "--", "-.", ":"]
    line_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax_idx, alg in enumerate(cvar_algs):
        ax = axes[ax_idx]
        label = chr(ord("a") + ax_idx)
        ax.set_title(f"({label}) {format_alg_label(alg)}")
        alg_keys = [k for k in groups if k[0] == alg]
        if not alg_keys:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            continue

        reps_values  = sorted(set(k[1] for k in alg_keys), key=lambda r: r if r is not None else float("inf"))
        alpha_values = sorted(set(k[2] for k in alg_keys))

        for i, reps in enumerate(reps_values):
            x_vals, y_means, y_stds = [], [], []
            for alpha in alpha_values:
                key = (alg, reps, alpha)
                if key in groups:
                    arr = np.array(groups[key])
                    x_vals.append(alpha)
                    y_means.append(float(np.mean(arr)))
                    y_stds.append(float(np.std(arr)))
            x_arr = np.array(x_vals)
            y_arr = np.array(y_means)
            y_std_arr = np.array(y_stds)
            color = line_colors[i % len(line_colors)]
            ls    = line_styles[i % len(line_styles)]
            ax.plot(x_arr, y_arr, linestyle=ls, color=color, marker="o", label=f"reps={reps}")
            ax.fill_between(x_arr, y_arr - y_std_arr, y_arr + y_std_arr, alpha=0.15, color=color)

        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("Mean RSQ (%)")
        ax.legend(frameon=False)
    plt.tight_layout()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_chars = uuid.uuid4().hex[:6]
    filepath = os.path.join(outputdir, f"rsq_cvar_by_alpha_{timestamp}_{random_chars}.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved CVaR RSQ alpha line plot to: {filepath}")

def plot_relative_solution_quality(outputdir, rows):
    plot_rsq_by_algorithm(outputdir, rows)
    plot_cvar_rsq_by_alpha(outputdir, rows)

def main():
    try:
        parser = argparse.ArgumentParser(description="Generate tabular and plot summaries from pre-computed EPA CSV files.")
        parser.add_argument("--resdir", type=str, required=True, help="Directory containing pre-computed CSV files (success_probabilities_*.csv for --metrics epa, relative_solution_quality_*.csv for --metrics rsq)")
        parser.add_argument("--outputdir", type=str, required=True, help="Directory to save output plots and files")
        parser.add_argument("--metrics", type=str, required=True, choices=["epa", "rsq"], help="Metric to analyze (currently supported: 'epa', 'rsq')")
        parser.add_argument("--exclude", type=str, default=None, help="Path to a .txt file listing problem names to exclude (one per line)")
        args = parser.parse_args()

        for arg_name, dir_path in [("--resdir", args.resdir), ("--outputdir", args.outputdir)]:
            if not os.path.isdir(dir_path):
                parser.error(f"{arg_name}: '{dir_path}' is not an existing directory")
        if args.exclude and not os.path.isfile(args.exclude):
            parser.error(f"--exclude: '{args.exclude}' is not an existing file")

        exclude = load_exclude(args.exclude)

        if args.metrics == "epa":
            csv_files = sorted(
                os.path.join(args.resdir, f)
                for f in os.listdir(args.resdir)
                if f.startswith("success_probabilities_") and f.endswith(".csv")
            )
            if not csv_files:
                raise FileNotFoundError(f"No success_probabilities_*.csv files found in --resdir '{args.resdir}'")
            rows = []
            seen_excluded: set = set()
            excluded_row_count = 0
            kept_problems: set = set()
            for csv_path in csv_files:
                print(f"Loading: {csv_path}")
                with open(csv_path, "r", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        prob = row["problem_name"]
                        if prob in exclude:
                            seen_excluded.add(prob)
                            excluded_row_count += 1
                            continue
                        kept_problems.add(prob)
                        rows.append({
                            "problem_name": prob,
                            "algorithm": row["algorithm"],
                            "reps": int(row["reps"]) if row["reps"] not in ("", "None") else None,
                            "alpha": float(row["alpha"]) if row["alpha"] not in ("", "None") else None,
                            "success_probability": float(row["success_probability"]),
                            "n_ground_states": int(row["n_ground_states"]),
                        })
            if exclude:
                print(f"Excluded {len(seen_excluded)} problems ({excluded_row_count} rows); "
                      f"{len(kept_problems)} problems remaining.")
            tabular_success_probabilities(args.outputdir, rows)
            plot_success_probabilities(args.outputdir, rows)

        elif args.metrics == "rsq":
            csv_files = sorted(
                os.path.join(args.resdir, f)
                for f in os.listdir(args.resdir)
                if f.startswith("relative_solution_quality_") and f.endswith(".csv")
            )
            if not csv_files:
                raise FileNotFoundError(f"No relative_solution_quality_*.csv files found in --resdir '{args.resdir}'")
            rows = []
            seen_excluded: set = set()
            excluded_row_count = 0
            kept_problems: set = set()
            for csv_path in csv_files:
                print(f"Loading: {csv_path}")
                with open(csv_path, "r", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        prob = row["problem_name"]
                        if prob in exclude:
                            seen_excluded.add(prob)
                            excluded_row_count += 1
                            continue
                        kept_problems.add(prob)
                        rows.append({
                            "problem_name": prob,
                            "algorithm": row["algorithm"],
                            "reps": int(row["reps"]) if row["reps"] not in ("", "None") else None,
                            "alpha": float(row["alpha"]) if row["alpha"] not in ("", "None") else None,
                            "best_objective_value": float(row["best_objective_value"]),
                            "cplex_fval": float(row["cplex_fval"]),
                            "relative_solution_quality": float(row["relative_solution_quality"]),
                        })
            if exclude:
                print(f"Excluded {len(seen_excluded)} problems ({excluded_row_count} rows); "
                      f"{len(kept_problems)} problems remaining.")
            tabular_relative_solution_quality(args.outputdir, rows)
            plot_relative_solution_quality(args.outputdir, rows)

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        raise SystemExit(1)

if __name__ == "__main__":
    main()

