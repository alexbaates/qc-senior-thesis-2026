#!/usr/bin/env python3
"""
tally_wins.py – For each docking problem, determine which (algorithm, reps, alpha)
combination performed best on SP (success probability), then tally wins per
combination across all problems.

Tie rule: all combinations tied at the max value for a problem each receive 1 win.
Per-row percentage = wins / n_problems * 100.
"""

import argparse
import csv
import datetime
import os
import sys
import uuid
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

"""
Constants 
"""

ALGORITHMS = ["qaoa", "qrao", "cvar_qaoa", "cvar_vqe", "ma_qaoa", "vqe", "ws_qaoa"]
CVAR_ALGS = {"cvar_qaoa", "cvar_vqe"}
ALGORITHM_LABELS = {
    "qaoa":      "QAOA",
    "qrao":      "QRAO",
    "vqe":       "VQE",
    "ws_qaoa":   "WS-QAOA",
    "ma_qaoa":   "MA-QAOA",
    "cvar_qaoa": "CVaR QAOA",
    "cvar_vqe":  "CVaR VQE",
}
ALGORITHM_ORDER = {alg: i for i, alg in enumerate(ALGORITHMS)}

"""
Sorting helpers
"""

def category_sort_key(cat):
    alg, reps, alpha = cat
    return (
        ALGORITHM_ORDER.get(alg, len(ALGORITHMS)),
        reps  if reps  is not None else float("inf"),
        alpha if alpha is not None else float("inf"),
    )


def format_alg_label(alg):
    return ALGORITHM_LABELS.get(alg, alg.replace("_", "-").upper())


"""
Loading
"""

def load_exclude(exclude_path):
    if not exclude_path:
        return set()
    with open(exclude_path) as fh:
        return {line.strip() for line in fh if line.strip()}


def load_all_csvs(dirpath, metric_col):
    data = defaultdict(dict)   # problem_name -> {(alg, reps, alpha): value}
    first_seen = defaultdict(dict)  # problem_name -> {(alg, reps, alpha): fname}
    for fname in sorted(os.listdir(dirpath)):
        if not fname.endswith(".csv"):
            continue
        fpath = os.path.join(dirpath, fname)
        with open(fpath, newline="") as fh:
            reader = csv.DictReader(fh)
            if metric_col not in (reader.fieldnames or []):
                print(
                    f"  Warning: '{fname}' has no column '{metric_col}'; skipping.",
                    file=sys.stderr,
                )
                continue
            for row in reader:
                alg   = row["algorithm"]
                reps  = int(row["reps"])   if row.get("reps",  "") not in ("", "None") else None
                alpha = float(row["alpha"]) if row.get("alpha", "") not in ("", "None") else None
                value = float(row[metric_col])
                prob  = row["problem_name"]
                key   = (alg, reps, alpha)
                if key in data[prob]:
                    raise ValueError(
                        f"Duplicate entry: problem='{prob}', algorithm='{alg}', "
                        f"reps={reps}, alpha={alpha} — first seen in "
                        f"'{first_seen[prob][key]}', also in '{fname}'"
                    )
                data[prob][key] = value
                first_seen[prob][key] = fname

    # Convert inner dicts to lists of tuples
    return {
        prob: [(alg, reps, alpha, val) for (alg, reps, alpha), val in entries.items()]
        for prob, entries in data.items()
    }


"""
Validation
"""

def derive_and_validate_categories(sp_data, rsqbest_data, rsqavg_data, problems):
    all_cats = set()
    for data in (sp_data, rsqbest_data, rsqavg_data):
        for prob in problems:
            if prob in data:
                for alg, reps, alpha, _ in data[prob]:
                    all_cats.add((alg, reps, alpha))

    for metric_name, data in [
        ("SP",       sp_data),
        ("RSQ best", rsqbest_data),
        ("RSQ avg",  rsqavg_data),
    ]:
        for problem in problems:
            if problem not in data:
                raise ValueError(
                    f"Problem '{problem}' has no data in the {metric_name} directory."
                )
            problem_cats = {(a, r, al) for a, r, al, _ in data[problem]}
            for cat in sorted(all_cats, key=category_sort_key):
                if cat not in problem_cats:
                    alg, reps, alpha = cat
                    raise ValueError(
                        f"[{metric_name}] Problem '{problem}' is missing "
                        f"algorithm='{alg}', reps={reps}, alpha={alpha}."
                    )

    return sorted(all_cats, key=category_sort_key)


"""
Win Tallying
"""

def compute_wins(data, problems):
    wins = defaultdict(int)
    for problem in problems:
        entries = data[problem]
        max_val = max(val for _, _, _, val in entries)
        for alg, reps, alpha, val in entries:
            if val == max_val:
                wins[(alg, reps, alpha)] += 1
    return wins


"""
Save CSV and LaTeX outputs
"""

def save_csv(outputdir, categories, sp_wins, n_problems):
    rows = []
    for cat in categories:
        alg, reps, alpha = cat
        sp_w = sp_wins.get(cat, 0)
        rows.append({
            "algorithm": alg,
            "reps":      reps  if reps  is not None else "",
            "alpha":     alpha if alpha is not None else "",
            "SP_wins":   sp_w,
            "SP_pct":    round(sp_w / n_problems * 100.0, 2),
        })

    timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rand       = uuid.uuid4().hex[:6]
    fname      = f"wins_tally_{timestamp}_{rand}.csv"
    fpath      = os.path.join(outputdir, fname)
    fieldnames = ["algorithm", "reps", "alpha", "SP_wins", "SP_pct"]
    with open(fpath, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved wins tally CSV to: {fpath}")
    return fpath

def save_latex(outputdir, categories, sp_wins, n_problems):
    def fmt_pct(v):
        return f"{v:.1f}"

    def fmt_alpha(v):
        return f"{v:.2f}"

    def fmt_wins(v):
        return str(int(v)) if v == int(v) else f"{v:.1f}"

    def latex_escape(s):
        return s.replace("_", r"\_")

    # Ordered unique algorithm names
    seen_algs, seen_set = [], set()
    for cat in categories:
        if cat[0] not in seen_set:
            seen_algs.append(cat[0])
            seen_set.add(cat[0])

    lines = []
    lines.append(r"% Requires \usepackage{booktabs, multirow} in your LaTeX preamble")
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lll|rr}")
    lines.append(r"\toprule")
    lines.append(
        r"Algorithm & Reps & $\alpha$ & SP & SP\,(\%) \\"
    )
    lines.append(r"\midrule")

    first_alg = True
    for alg in seen_algs:
        if not first_alg:
            lines.append(r"\midrule")
        first_alg = False

        alg_keys    = [k for k in categories if k[0] == alg]
        alg_count   = len(alg_keys)
        alg_display = format_alg_label(alg)
        has_alpha   = any(k[2] is not None for k in alg_keys)

        if has_alpha:
            # Iterate reps groups; within each reps group, iterate alpha variants
            seen_reps, seen_reps_set = [], set()
            for k in alg_keys:
                if k[1] not in seen_reps_set:
                    seen_reps.append(k[1])
                    seen_reps_set.add(k[1])

            first_reps_group = True
            for reps in seen_reps:
                reps_keys  = [k for k in alg_keys if k[1] == reps]
                reps_count = len(reps_keys)

                if not first_reps_group:
                    lines.append(r"\cmidrule{2-5}")
                first_reps_group = False

                for i, key in enumerate(reps_keys):
                    sp_w = sp_wins.get(key, 0)
                    sp_p = sp_w / n_problems * 100.0
                    alpha = key[2]

                    is_first_row = (reps == seen_reps[0] and i == 0)
                    alg_cell  = (
                        f"\\multirow{{{alg_count}}}{{*}}{{{alg_display}}}"
                        if is_first_row else ""
                    )
                    reps_cell = (
                        f"\\multirow{{{reps_count}}}{{*}}{{{reps}}}"
                        if i == 0 else ""
                    )
                    alpha_cell = fmt_alpha(alpha) if alpha is not None else ""

                    lines.append(
                        f"    {alg_cell} & {reps_cell} & {alpha_cell} "
                        f"& {fmt_wins(sp_w)} & {fmt_pct(sp_p)} \\\\"
                    )
        else:
            for i, key in enumerate(alg_keys):
                reps = key[1]
                sp_w = sp_wins.get(key, 0)
                sp_p = sp_w / n_problems * 100.0

                alg_cell  = (
                    f"\\multirow{{{alg_count}}}{{*}}{{{alg_display}}}"
                    if i == 0 else ""
                )
                reps_cell  = str(reps) if reps is not None else ""

                lines.append(
                    f"    {alg_cell} & {reps_cell} & "
                    f"& {fmt_wins(sp_w)} & {fmt_pct(sp_p)} \\\\"
                )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Algorithm win tallies: number of problems (and \%) for which each "
        r"(algorithm, reps, $\alpha$) combination achieved the best metric value.}"
    )
    lines.append(r"\label{tab:wins}")
    lines.append(r"\end{table}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rand      = uuid.uuid4().hex[:6]
    fname     = f"wins_tally_table_{timestamp}_{rand}.tex"
    fpath     = os.path.join(outputdir, fname)
    if os.path.exists(fpath):
        raise FileExistsError(f"Output file already exists: '{fpath}'")
    with open(fpath, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Saved LaTeX table to: {fpath}")
    return fpath


"""
Save Plots
"""


def plot_wins(outputdir, categories, wins_dict, metric_label, filename_tag, n_problems):
    non_cvar_algs = [a for a in ALGORITHMS if a not in CVAR_ALGS]

    # Assign one color per reps value
    all_reps = sorted(
        {reps for (_, reps, _) in categories if reps is not None},
        key=lambda r: r if r is not None else float("inf"),
    )
    colors     = plt.cm.tab10(np.linspace(0, 0.9, max(len(all_reps), 1)))
    reps_color = {r: colors[i] for i, r in enumerate(all_reps)}

    # Build ordered (x_label, bars) pairs
    # bars = list of (reps, wins_value), sorted by reps
    x_labels = []
    x_bars   = []

    for alg in non_cvar_algs:
        alg_cats = [c for c in categories if c[0] == alg]
        if not alg_cats:
            continue
        bars = sorted(
            [(reps, wins_dict.get((alg, reps, alpha), 0))
             for (_, reps, alpha) in alg_cats],
            key=lambda b: b[0] if b[0] is not None else float("inf"),
        )
        x_labels.append(format_alg_label(alg))
        x_bars.append(bars)

    for alg in ["cvar_qaoa", "cvar_vqe"]:
        alg_cats = [c for c in categories if c[0] == alg]
        if not alg_cats:
            continue
        alphas = sorted({alpha for (_, _, alpha) in alg_cats if alpha is not None})
        for alpha in alphas:
            alpha_cats = [c for c in alg_cats if c[2] == alpha]
            bars = sorted(
                [(reps, wins_dict.get((alg, reps, alpha), 0))
                 for (_, reps, _) in alpha_cats],
                key=lambda b: b[0] if b[0] is not None else float("inf"),
            )
            x_labels.append(f"{format_alg_label(alg)}\n$\\alpha$={alpha:.2f}")
            x_bars.append(bars)

    n_groups = len(x_labels)
    if n_groups == 0:
        print(f"No data to plot for {metric_label}; skipping.", file=sys.stderr)
        return

    max_bars = max(len(b) for b in x_bars)
    bar_width = min(0.8 / max(max_bars, 1), 0.25)
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(max(10, n_groups * 1.4), 6))

    reps_legend_added = set()
    for gi, bars in enumerate(x_bars):
        n_bars = len(bars)
        for bi, (reps, wins) in enumerate(bars):
            offset = (bi - n_bars / 2 + 0.5) * bar_width
            color  = reps_color.get(reps, "gray")
            label  = f"reps={reps}" if reps not in reps_legend_added else None
            if reps not in reps_legend_added:
                reps_legend_added.add(reps)
            ax.bar(
                x[gi] + offset, wins, width=bar_width,
                color=color, alpha=0.85, label=label,
                edgecolor="white", linewidth=0.5,
            )
            # Value label on each bar
            if wins > 0:
                ax.text(
                    x[gi] + offset, wins + 0.15, str(wins),
                    ha="center", va="bottom", fontsize=7,
                )

    # Dashed reference line at total number of problems
    ax.axhline(n_problems, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(
        n_groups - 0.5, n_problems + 0.2, f"n = {n_problems}",
        ha="right", va="bottom", color="gray", fontsize=8,
    )

    ax.set_ylabel("Number of Problems")
    ax.set_xlabel("Algorithm")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, ha="center", fontsize=9)
    ax.set_xlim(-0.5, n_groups - 0.5)
    ax.set_ylim(0, n_problems * 1.2)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False, title="Reps")
    plt.tight_layout()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rand      = uuid.uuid4().hex[:6]
    fpath     = os.path.join(outputdir, f"wins_{filename_tag}_{timestamp}_{rand}.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to: {fpath}")


"""
Main script
"""

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Tally per-problem wins for each (algorithm, reps, alpha) combination "
            "based on success probability (SP)."
        )
    )
    parser.add_argument(
        "--rsqavg", default=None,
        help="Optional: directory containing avg_relative_solution_quality_*.csv files (ignored if omitted)",
    )
    parser.add_argument(
        "--rsqbest", default=None,
        help="Optional: directory containing relative_solution_quality_*.csv files (ignored if omitted)",
    )
    parser.add_argument(
        "--sp", required=True,
        help="Directory containing success_probabilities_*.csv files",
    )
    parser.add_argument(
        "--output", required=True,
        help="Directory to write output files",
    )
    parser.add_argument(
        "--exclude", default=None,
        help="Path to a .txt file listing problem names (one per line) to exclude",
    )
    args = parser.parse_args()

    for flag, path in [("--sp", args.sp), ("--output", args.output)]:
        if not os.path.isdir(path):
            parser.error(f"{flag}: '{path}' is not an existing directory")
    for flag, path in [("--rsqavg", args.rsqavg), ("--rsqbest", args.rsqbest)]:
        if path is not None and not os.path.isdir(path):
            parser.error(f"{flag}: '{path}' is not an existing directory")
    if args.exclude and not os.path.isfile(args.exclude):
        parser.error(f"--exclude: '{args.exclude}' is not an existing file")

    os.makedirs(args.output, exist_ok=True)

    exclude = load_exclude(args.exclude)
    if exclude:
        print(f"Excluding {len(exclude)} problems listed in '{args.exclude}'.")

    print("Loading SP data …")
    sp_data = load_all_csvs(args.sp, "success_probability")
    print(f"  {len(sp_data)} problems found.")

    all_problems = sorted(sp_data.keys())
    problems = [p for p in all_problems if p not in exclude]
    print(
        f"{len(problems)} problems to tally "
        f"({len(all_problems)} total, {len(all_problems) - len(problems)} excluded)."
    )

    all_cats = set()
    for prob in problems:
        for alg, reps, alpha, _ in sp_data[prob]:
            all_cats.add((alg, reps, alpha))
    categories = sorted(all_cats, key=category_sort_key)
    print(f"{len(categories)} (algorithm, reps, alpha) categories found:")
    for cat in categories:
        alg, reps, alpha = cat
        alpha_str = f", alpha={alpha}" if alpha is not None else ""
        print(f"  {alg}, reps={reps}{alpha_str}")

    print("Computing wins …")
    sp_wins = compute_wins(sp_data, problems)

    n_problems = len(problems)
    print(f"Saving outputs to '{args.output}' …")
    save_csv(args.output, categories, sp_wins, n_problems)
    save_latex(args.output, categories, sp_wins, n_problems)
    plot_wins(args.output, categories, sp_wins, "SP", "sp", n_problems)
    print("Done.")


if __name__ == "__main__":
    main()
