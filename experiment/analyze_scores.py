"""Generate CSV recap and visualizations of scores.jsonl grouped by method."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

METRICS = ["detection_accuracy", "fuzz_accuracy", "embedding_gap"]
METHOD_RENAME = {"delphi": "eleuther_acts_top20"}


def load(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        s = r["scores"]
        rows.append({
            "index": r["index"],
            "method": r["method"],
            "explanation_model": r["explanation_model"],
            "detection_accuracy": s["detection"]["accuracy"],
            "fuzz_accuracy": s["fuzz"]["accuracy"],
            "embedding_gap": s["embedding"]["gap"],
            "embedding_mean_pos": s["embedding"]["mean_pos"],
            "embedding_mean_neg": s["embedding"]["mean_neg"],
        })
    df = pd.DataFrame(rows)
    df["method"] = df["method"].replace(METHOD_RENAME)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("method")[METRICS].agg(["mean", "std", "median", "min", "max", "count"])
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    return agg.reset_index()


def plot_tables(df: pd.DataFrame, out: Path) -> None:
    stats = ["mean", "std", "median", "min", "max"]
    for metric in METRICS:
        t = df.groupby("method")[metric].agg(stats).round(4).reset_index()
        fig, ax = plt.subplots(figsize=(1.4 * (len(stats) + 1) + 1, 0.5 * len(t) + 1.2))
        ax.axis("off")
        tbl = ax.table(cellText=t.values, colLabels=t.columns, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.5)
        for j in range(len(t.columns)):
            tbl[(0, j)].set_facecolor("#40466e")
            tbl[(0, j)].set_text_props(color="white", weight="bold")
        ax.set_title(f"{metric} by method", pad=12)
        fig.tight_layout()
        fig.savefig(out / f"table_{metric}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot(df: pd.DataFrame, out: Path) -> None:
    sns.set_theme(style="whitegrid")
    order = sorted(df["method"].unique())

    fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 5))
    for ax, metric in zip(axes, METRICS):
        sns.boxplot(data=df, x="method", y=metric, order=order, ax=ax, showmeans=True,
                    meanprops={"marker": "D", "markerfacecolor": "red", "markeredgecolor": "red"})
        sns.stripplot(data=df, x="method", y=metric, order=order, ax=ax, color="black", alpha=0.3, size=3)
        ax.set_title(metric)
        ax.set_xlabel("")
    fig.suptitle("Score distribution by label generation method", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "scores_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    means = df.groupby("method")[METRICS].mean().loc[order].T
    fig, ax = plt.subplots(figsize=(8, 5))
    means.plot(kind="bar", ax=ax, rot=0)
    ax.set_title("Mean score by scorer metric")
    ax.set_xlabel("")
    ax.set_ylabel("value")
    ax.legend(title="method", loc="best")
    fig.tight_layout()
    fig.savefig(out / "scores_mean_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exp-dir", default="data/experiments/exp_3")
    args = p.parse_args()
    exp = Path(args.exp_dir)

    df = load(exp / "scores.jsonl")
    df.to_csv(exp / "scores_flat.csv", index=False)

    summary = summarize(df)
    summary.to_csv(exp / "scores_summary_by_method.csv", index=False)

    plot(df, exp)
    plot_tables(df, exp)

    print(summary.to_string(index=False))
    print(f"\nWrote: {exp/'scores_flat.csv'}, {exp/'scores_summary_by_method.csv'}, "
          f"{exp/'scores_boxplot.png'}, {exp/'scores_mean_bar.png'}, "
          + ", ".join(f"table_{m}.png" for m in METRICS))


if __name__ == "__main__":
    main()
