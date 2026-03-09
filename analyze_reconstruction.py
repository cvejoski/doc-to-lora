#!/usr/bin/env python3
"""
Memorization analysis for the Doc-to-LoRA context reconstruction dataset.

Loads JetBrains-Research/d2l-reconstruction (or a local file) and produces
a 2×2 figure:

  1. Distribution of best word-overlap per sample (histogram + KDE)
  4. Which prompt wins most often (bar chart)
  7. Response word-count vs word-overlap (scatter)
  S. Longest memorized span distribution (histogram + KDE)

Span memorization
-----------------
For each (context, response) pair we find the longest contiguous sequence of
words that appears verbatim in both strings (case-insensitive), using
difflib.SequenceMatcher.  This is then expressed as a fraction of the context
word count and as a raw word count.

Usage
-----
    uv run python analyze_reconstruction.py \\
        [--hub_dataset JetBrains-Research/d2l-reconstruction] \\
        [--config squad] \\
        [--local_file results.parquet] \\
        [--out_dir plots/] \\
        [--dpi 150]
"""

import argparse
import difflib
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
]


# ──────────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_data(hub_dataset: str | None, config: str | None, local_file: str | None) -> pd.DataFrame:
    if local_file:
        path = Path(local_file)
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix in (".arrow", ".feather"):
            df = pd.read_feather(path)
        elif path.suffix == ".csv":
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported local file format: {path.suffix}")
        print(f"Loaded {len(df):,} rows from {local_file}")
        return df

    from datasets import load_dataset
    kwargs = {"name": config} if config else {}
    print(f"Downloading {hub_dataset}  config={config or 'all'} …")
    ds = load_dataset(hub_dataset, **kwargs)

    dfs = []
    if hasattr(ds, "items"):
        for split_name, split_ds in ds.items():
            tmp = split_ds.to_pandas()
            tmp["split"] = split_name
            dfs.append(tmp)
    else:
        dfs.append(ds.to_pandas())

    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df):,} rows from Hub")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Span memorization metric
# ──────────────────────────────────────────────────────────────────────────────

def longest_common_span(context: str, response: str) -> int:
    """
    Length (in words) of the longest contiguous word sequence that appears
    verbatim (case-insensitive) in both context and response.
    Uses difflib.SequenceMatcher — O(n*m) but fast enough for typical lengths.
    """
    wa = context.lower().split()
    wb = response.lower().split()
    matcher = difflib.SequenceMatcher(None, wa, wb, autojunk=False)
    block = matcher.find_longest_match(0, len(wa), 0, len(wb))
    return block.size


# ──────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ──────────────────────────────────────────────────────────────────────────────

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ctx_word_count"] = df["context"].str.split().str.len()
    df["resp_word_count"] = df["response"].str.split().str.len()

    print("Computing longest common spans … (may take a moment)")
    df["span_words"] = [
        longest_common_span(ctx, resp)
        for ctx, resp in zip(df["context"], df["response"])
    ]
    # fraction of context words covered by the best span
    df["span_frac"] = df["span_words"] / df["ctx_word_count"].clip(lower=1)

    group_keys = ["source_dataset", "checkpoint", "sample_idx"]

    # best word-overlap per sample
    best_ov = (
        df.groupby(group_keys)["word_overlap"]
        .max().reset_index()
        .rename(columns={"word_overlap": "best_overlap"})
    )
    df = df.merge(best_ov, on=group_keys, how="left")

    # which prompt achieved best overlap
    idx_of_best = (
        df.loc[df.groupby(group_keys)["word_overlap"].idxmax(),
               group_keys + ["prompt_idx", "prompt"]]
        .rename(columns={"prompt_idx": "best_prompt_idx", "prompt": "best_prompt"})
    )
    df = df.merge(idx_of_best, on=group_keys, how="left")

    # best span per sample (over all prompts for that sample)
    best_span = (
        df.groupby(group_keys)["span_words"]
        .max().reset_index()
        .rename(columns={"span_words": "best_span_words"})
    )
    df = df.merge(best_span, on=group_keys, how="left")

    return df


def short_label(prompt: str, max_chars: int = 38) -> str:
    return textwrap.shorten(prompt.strip().split("\n")[0], width=max_chars, placeholder="…")


# ──────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ──────────────────────────────────────────────────────────────────────────────

def _style(ax, xlabel="", ylabel="", title=""):
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def _hist_kde(ax, vals, color, xlabel, title):
    vals = vals[np.isfinite(vals)]
    ax.hist(vals, bins=30, color=color, alpha=0.65, edgecolor="white", density=True)
    if len(vals) > 3:
        try:
            xs = np.linspace(vals.min(), vals.max(), 300)
            ax.plot(xs, gaussian_kde(vals, bw_method="scott")(xs), color=color, lw=2)
        except Exception:
            pass
    med = np.median(vals)
    ax.axvline(med, color="black", lw=1.2, ls="--", label=f"Median {med:.2f}")
    ax.legend(fontsize=8)
    _style(ax, xlabel=xlabel, ylabel="Density", title=title)


# ──────────────────────────────────────────────────────────────────────────────
# The four panels
# ──────────────────────────────────────────────────────────────────────────────

def plot_best_overlap_hist(ax, sample_df: pd.DataFrame):
    _hist_kde(ax, sample_df["best_overlap"].values, PALETTE[0],
              xlabel="Best word-overlap (fraction)",
              title="1. Best word-overlap per sample")
    ax.set_xlim(0, 1)


def plot_best_prompt_frequency(ax, sample_df: pd.DataFrame, prompt_labels: dict):
    counts = sample_df["best_prompt_idx"].value_counts().sort_index()
    labels = [prompt_labels.get(i, str(i)) for i in counts.index]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(counts))]
    bars = ax.bar(range(len(counts)), counts.values, color=colors, edgecolor="white")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=7)
    ax.bar_label(bars, padding=2, fontsize=8)
    _style(ax, ylabel="# samples", title="4. Which prompt wins most often?")


def plot_response_length_vs_overlap(ax, df: pd.DataFrame):
    ax.scatter(df["resp_word_count"], df["word_overlap"],
               alpha=0.25, s=8, color=PALETTE[4], edgecolors="none")
    _style(ax, xlabel="Response word count", ylabel="Word overlap",
           title="7. Response verbosity vs word overlap")
    ax.set_ylim(0, 1)


def plot_span_memorization(ax, sample_df: pd.DataFrame):
    """
    Two sub-views in one panel:
      • Main axis: histogram + KDE of best span length (words) per sample
      • Inset: fraction of context covered by best span
    """
    vals = sample_df["best_span_words"].values.astype(float)
    _hist_kde(ax, vals, PALETTE[2],
              xlabel="Longest memorized span (words)",
              title="S. Longest verbatim span per sample")

    # small inset: span as fraction of context
    inset = ax.inset_axes([0.52, 0.45, 0.44, 0.48])
    frac_vals = (sample_df["best_span_words"] / sample_df["ctx_word_count"].clip(lower=1)).values
    frac_vals = frac_vals[np.isfinite(frac_vals)]
    inset.hist(frac_vals, bins=20, color=PALETTE[3], alpha=0.7, edgecolor="white", density=True)
    inset.set_xlabel("Span / ctx length", fontsize=7)
    inset.set_ylabel("Density", fontsize=7)
    inset.tick_params(labelsize=6)
    inset.spines[["top", "right"]].set_visible(False)
    inset.set_xlim(0, 1)
    med_frac = np.median(frac_vals)
    inset.axvline(med_frac, color="black", lw=1, ls="--")
    inset.set_title(f"Span fraction (med={med_frac:.2f})", fontsize=7)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Memorization analysis plots for d2l-reconstruction dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--hub_dataset", default="JetBrains-Research/d2l-reconstruction")
    parser.add_argument("--config", default=None,
                        help="HF dataset config name (e.g. 'squad', 'codeqa')")
    parser.add_argument("--local_file", default=None,
                        help="Load from a local parquet/arrow/csv instead of the Hub.")
    parser.add_argument("--out_dir", default="plots")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.hub_dataset, args.config, args.local_file)
    df = enrich(df)

    sample_df = (
        df.drop_duplicates(subset=["source_dataset", "checkpoint", "sample_idx"])
        .reset_index(drop=True)
    )

    prompt_labels: dict[int, str] = (
        df.drop_duplicates("prompt_idx").sort_values("prompt_idx")
        .set_index("prompt_idx")["prompt"].apply(short_label).to_dict()
    )

    configs = df["source_dataset"].unique().tolist() if "source_dataset" in df.columns else ["?"]
    print(f"\nDataset summary")
    print(f"  configs        : {configs}")
    print(f"  samples        : {df['sample_idx'].nunique()}")
    print(f"  prompts        : {df['prompt_idx'].nunique()}")
    print(f"  records        : {len(df):,}")
    print(f"  best overlap   : mean={sample_df['best_overlap'].mean():.3f}  "
          f"median={sample_df['best_overlap'].median():.3f}")
    print(f"  best span(wds) : mean={sample_df['best_span_words'].mean():.1f}  "
          f"median={sample_df['best_span_words'].median():.1f}  "
          f"max={sample_df['best_span_words'].max():.0f}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.subplots_adjust(hspace=0.45, wspace=0.35)

    plot_best_overlap_hist(axes[0, 0], sample_df)
    plot_best_prompt_frequency(axes[0, 1], sample_df, prompt_labels)
    plot_response_length_vs_overlap(axes[1, 0], df)
    plot_span_memorization(axes[1, 1], sample_df)

    title = "Doc-to-LoRA  ·  Context Memorization Analysis"
    if configs:
        title += f"  [{', '.join(str(c) for c in configs)}]"
    fig.suptitle(title, fontsize=13, fontweight="bold")

    out_path = out_dir / "memorization_overview.png"
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out_path}")

    # print top-memorized spans for inspection
    top = (
        df.sort_values("span_words", ascending=False)
        .head(10)[["sample_idx", "prompt_idx", "span_words", "word_overlap", "response"]]
    )
    print("\n── Top 10 longest memorized spans ──")
    for _, row in top.iterrows():
        snippet = " ".join(row["response"].split()[:20])
        print(f"  sample={row['sample_idx']}  prompt={row['prompt_idx']}  "
              f"span={row['span_words']}w  overlap={row['word_overlap']:.2f}  "
              f"response: {snippet}…")

    print("\nDone.")


if __name__ == "__main__":
    main()
