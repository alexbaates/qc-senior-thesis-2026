"""
Plot mean TensorRL training curve across all problems.

Reads all expectation_value_per_episode_*.csv files from --inputdir,
computes the per-episode mean of a chosen energy column across all problems,
and saves a PNG line plot.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_exclude(exclude_path):
    if not exclude_path:
        return set()
    with open(exclude_path) as fh:
        return {line.strip() for line in fh if line.strip()}


def main():
    parser = argparse.ArgumentParser(
        description="Plot mean TensorRL training curve across all problems."
    )
    parser.add_argument("--inputdir", required=True, help="Directory containing expectation_value_per_episode_*.csv files")
    parser.add_argument("--outputdir", required=True, help="Directory to write the output PNG")
    parser.add_argument(
        "--column",
        default="final_energy",
        choices=["final_energy", "min_energy"],
        help="Energy column to plot (default: final_energy)",
    )
    parser.add_argument(
        "--exclude",
        default=None,
        help="Path to a .txt file listing problem names to exclude (one per line)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.inputdir):
        parser.error(f"--inputdir: '{args.inputdir}' is not a directory")
    if args.exclude and not os.path.isfile(args.exclude):
        parser.error(f"--exclude: '{args.exclude}' is not an existing file")

    exclude = load_exclude(args.exclude)

    csv_files = sorted(
        f for f in os.listdir(args.inputdir)
        if f.startswith("expectation_value_per_episode_") and f.endswith(".csv")
    )
    if exclude:
        before = len(csv_files)
        csv_files = [f for f in csv_files if not any(name in f for name in exclude)]
        print(f"Excluded {before - len(csv_files)} files matching {len(exclude)} problem names.")
    if not csv_files:
        parser.error(f"No expectation_value_per_episode_*.csv files found in '{args.inputdir}'")

    print(f"Found {len(csv_files)} CSV files.")

    # Load all files; keep only rows where the episode column is an integer
    frames = []
    skipped = 0
    for fname in csv_files:
        path = os.path.join(args.inputdir, fname)
        try:
            df = pd.read_csv(path)
            if "episode" not in df.columns or args.column not in df.columns:
                print(f"  Warning: skipping '{fname}' — missing required columns.")
                skipped += 1
                continue
            frames.append(df[["episode", args.column]].copy())
        except Exception as e:
            print(f"  Warning: skipping '{fname}': {e}")
            skipped += 1

    if not frames:
        raise RuntimeError("No usable CSV files could be loaded.")

    if skipped:
        print(f"Skipped {skipped} files.")

    combined = pd.concat(frames, ignore_index=True)

    # Average across all problems for each episode
    grouped = combined.groupby("episode")[args.column]
    mean_curve = grouped.mean()
    std_curve = grouped.std()
    episodes = mean_curve.index.to_numpy()
    means = mean_curve.to_numpy()
    stds = std_curve.to_numpy()

    print(f"Episodes: {episodes[0]}–{episodes[-1]}  ({len(episodes)} points, {len(frames)} problems)")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(episodes, means, linewidth=1.5, color="steelblue", label="Mean")
    ax.fill_between(episodes, means - stds, means + stds, alpha=0.2, color="steelblue", label="±1 std")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Expectation Value")
    ax.legend()
    fig.tight_layout()

    os.makedirs(args.outputdir, exist_ok=True)
    out_path = os.path.join(args.outputdir, "tensorrl_training_curve.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
