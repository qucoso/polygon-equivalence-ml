"""Plots for E1 (train-task x eval-task matrix), E2/E3 (bars) and the MLP permutation importance.

With --paper, writes the manuscript figures to results/plots/:
    e1_<metric>_paper.pdf    Figure 11 (cross-task generalisation, MLP + Perceiver)
    pfi_mlp_paper.pdf        Figure 12 (permutation feature importance of the MLP)
    e3_<metric>_paper.pdf    Figure 13 (robustness to negative-pair distribution shifts)
Input: results/mlp_metrics.csv, results/e2e_metrics.csv, results/mlp_pfi.csv
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from experiments.common import FEATURE_NAMES, TASKS, results_dir  # noqa: E402

INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MODEL_ORDER = list(TASKS) + ["ref", "pretrained_full"]
MODEL_COLOR = dict(zip(MODEL_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                                     "#008300", "#6b6a66"]))
SEQUENTIAL = LinearSegmentedColormap.from_list("seq", ["#eef4fc", "#9cc0ec", "#2a78d6", "#123f78"])
LABEL = {"same_center_dif_shape": "same_center", "pretrained_full": "published (full data)"}
ARCH_NAME = {"mlp": "MLP", "e2e": "Perceiver"}


def label(name):
    return LABEL.get(name, name)


def aggregate(df, metric):
    g = df.groupby(["model", "eval_set"])[metric].agg(["mean", "std"]).reset_index()
    return g.fillna({"std": 0.0})


def annotate(ax, mean, std, vmin, vmax, fmt):
    for i in range(mean.shape[0]):
        for j in range(mean.shape[1]):
            frac = (mean[i, j] - vmin) / max(vmax - vmin, 1e-12)
            ink = "#ffffff" if frac > 0.55 else INK
            text = fmt.format(mean[i, j]) if std is None else f"{fmt.format(mean[i, j])}\n±{std[i, j]:.3f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=ink)


def heatmap(ax, mean, std, rows, cols, title, fmt="{:.3f}", cmap=SEQUENTIAL, vmin=None, vmax=None):
    vmin = np.nanmin(mean) if vmin is None else vmin
    vmax = np.nanmax(mean) if vmax is None else vmax
    image = ax.imshow(mean, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)), [label(c) for c in cols], rotation=35, ha="right", color=INK_MUTED)
    ax.set_yticks(range(len(rows)), [label(r) for r in rows], color=INK_MUTED)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    annotate(ax, mean, std, vmin, vmax, fmt)
    return image


def pivot(agg, rows, cols):
    mean = agg.pivot(index="model", columns="eval_set", values="mean").reindex(index=rows, columns=cols)
    std = agg.pivot(index="model", columns="eval_set", values="std").reindex(index=rows, columns=cols)
    return mean.to_numpy(), std.to_numpy()


def plot_e1(df, arch, metric, out):
    agg = aggregate(df, metric)
    rows = [m for m in TASKS if m in set(agg.model)]
    mean, std = pivot(agg, rows, list(TASKS))
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    image = heatmap(ax, mean, std, rows, list(TASKS),
                    f"E1 · {ARCH_NAME[arch]} · {metric} (rows: training task, columns: evaluation task)")
    ax.set_xlabel("evaluation set", color=INK_MUTED)
    ax.set_ylabel("training task", color=INK_MUTED)
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(out / f"e1_{arch}_{metric}.png", dpi=200)
    fig.savefig(out / f"e1_{arch}_{metric}.pdf")
    plt.close(fig)


PAPER_LABEL = {"same_center_dif_shape": "same center", "modified": "modified", "random": "random",
               "intersection": "intersection", "cluster": "cluster"}


def plot_e1_paper(metric, out):
    """Both architectures side by side on one shared colour scale, without a title (caption in LaTeX)."""
    res = results_dir()
    matrices = {}
    for arch in ("mlp", "e2e"):
        agg = aggregate(pd.read_csv(res / f"{arch}_metrics.csv"), metric)
        matrices[arch] = pivot(agg, list(TASKS), list(TASKS))
    low = min(np.nanmin(m[0]) for m in matrices.values())
    vmin, vmax = np.floor(low * 20) / 20, 1.0

    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 8})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), sharey=True)
    names = [PAPER_LABEL[t] for t in TASKS]
    for ax, arch, panel in zip(axes, ("mlp", "e2e"), ("(a) MLP", "(b) Perceiver")):
        mean, std = matrices[arch]
        image = ax.imshow(mean, cmap=SEQUENTIAL, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(TASKS)), names, rotation=35, ha="right", color=INK_MUTED)
        ax.set_yticks(range(len(TASKS)), names, color=INK_MUTED)
        ax.set_title(panel, loc="left", color=INK, fontsize=9)
        ax.set_xlabel("evaluation set", color=INK_MUTED)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i in range(len(TASKS)):
            for j in range(len(TASKS)):
                frac = (mean[i, j] - vmin) / (vmax - vmin)
                ax.text(j, i - 0.12, f"{mean[i, j]:.3f}", ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if frac > 0.55 else INK)
                ax.text(j, i + 0.22, f"±{std[i, j]:.3f}", ha="center", va="center", fontsize=6,
                        color="#ffffff" if frac > 0.55 else INK_MUTED)
            ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=1.1))
    axes[0].set_ylabel("training task", color=INK_MUTED)
    colorbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    colorbar.set_label(metric.upper() if metric != "f1" else "F1 score", color=INK_MUTED)
    fig.savefig(out / f"e1_{metric}_paper.pdf", bbox_inches="tight")
    fig.savefig(out / f"e1_{metric}_paper.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)


PAPER_FEATURES = ["area", "length", "width", "height", "area-to-length ratio", "centroid x",
                  "centroid y", "elongation", "sine angle", "cosine angle", "degree of convexity",
                  "circularity", "number of nodes", "polygon roughness"]


def plot_pfi_paper(out, relative=False):
    """Permutation importance of the five single-type models and the full-data model on the normal mix.

    relative=True divides every drop by the largest possible drop (baseline accuracy - 0.5).
    """
    pfi = pd.read_csv(results_dir() / "mlp_pfi.csv")
    models = list(TASKS) + ["pretrained_full"]
    pfi = pfi[(pfi.eval_set == "ref") & pfi.model.isin(models)]
    pfi = pfi.assign(value=pfi.delta_accuracy / (pfi.baseline_accuracy - 0.5) if relative
                     else pfi.delta_accuracy)
    agg = pfi.groupby(["feature", "model"])["value"].agg(["mean", "std"]).reset_index()
    mean = agg.pivot(index="feature", columns="model", values="mean").reindex(index=FEATURE_NAMES, columns=models)
    std = agg.pivot(index="feature", columns="model", values="std").reindex(index=FEATURE_NAMES, columns=models)
    mean, std = mean.to_numpy(), std.to_numpy()
    base = (pfi.drop_duplicates(["model", "seed"]).groupby("model")["baseline_accuracy"]
            .agg(["mean", "std"]).reindex(models))
    vmax = 1.0 if relative else np.ceil(np.nanmax(mean) * 20) / 20

    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 8})
    fig, (ax, ax_base) = plt.subplots(2, 1, figsize=(5.6, 6.4), sharex=True,
                                      gridspec_kw={"height_ratios": [len(PAPER_FEATURES), 1.4],
                                                   "hspace": 0.08})
    image = ax.imshow(np.clip(mean, 0, None), cmap=SEQUENTIAL, vmin=0.0, vmax=vmax, aspect="auto")
    columns = [PAPER_LABEL[m] for m in TASKS] + ["full dataset"]
    ax.set_yticks(range(len(PAPER_FEATURES)), PAPER_FEATURES, color=INK_MUTED)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(len(PAPER_FEATURES)):
        for j in range(len(models)):
            value = max(mean[i, j], 0.0)
            ink = "#ffffff" if value / vmax > 0.55 else INK
            single = models[j] == "pretrained_full"
            ax.text(j, i - (0.0 if single else 0.14), f"{value:.3f}", ha="center", va="center",
                    fontsize=7.5, color=ink)
            if not single:
                ax.text(j, i + 0.24, f"±{std[i, j]:.3f}", ha="center", va="center", fontsize=5.5,
                        color="#ffffff" if value / vmax > 0.55 else INK_MUTED)

    ax_base.set_facecolor("#f1f0ec")
    ax_base.set_xlim(-0.5, len(models) - 0.5)
    ax_base.set_ylim(0, 1)
    ax_base.set_yticks([0.5], ["accuracy before\npermutation"], color=INK_MUTED)
    ax_base.tick_params(length=0)
    for spine in ax_base.spines.values():
        spine.set_visible(False)
    for j, model in enumerate(models):
        single = model == "pretrained_full"
        ax_base.text(j, 0.5 if single else 0.62, f"{base['mean'][model]:.3f}", ha="center", va="center",
                     fontsize=7.5, color=INK)
        if not single:
            ax_base.text(j, 0.22, f"±{base['std'][model]:.3f}", ha="center", va="center", fontsize=5.5,
                         color=INK_MUTED)
    ax_base.set_xticks(range(len(models)), columns, rotation=35, ha="right", color=INK_MUTED)
    ax_base.set_xlabel("training data of the model", color=INK_MUTED)
    for axis in (ax, ax_base):
        axis.axvline(len(TASKS) - 0.5, color="#ffffff", linewidth=4)

    cax = ax.inset_axes([1.04, 0.0, 0.035, 1.0])
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label("fraction of the largest possible drop" if relative
                       else "drop in accuracy after permutation", color=INK_MUTED)
    stem = "pfi_mlp_paper_relative" if relative else "pfi_mlp_paper"
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)


E3_DISTRIBUTIONS = [("hard", "Distribution 1"), ("easy", "Distribution 2"), ("uniform", "Uniform")]
E3_COLOR = {"hard": "#2a78d6", "easy": "#eb6834", "uniform": "#1baf7a"}
E3_MODELS = list(TASKS) + ["ref", "pretrained_full"]
E3_MODEL_LABEL = {**PAPER_LABEL, "same_center_dif_shape": "same\ncenter", "ref": "published\n(reduced data)", "pretrained_full": "published\n(full data)"}
METRIC_LABEL = {"f1": "F1 score", "auc": "AUC", "accuracy": "accuracy"}


def plot_e3_paper(metric, out):
    """All training models on one shared x-axis, MLP above Perceiver, bars per test distribution.

    No title (caption in LaTeX). `ref` is the published setup trained on the reduced data size.
    """
    res = results_dir()
    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 8})
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True, gridspec_kw={"hspace": 0.18})
    n_dist = len(E3_DISTRIBUTIONS)
    width = 0.8 / n_dist
    x = np.arange(len(E3_MODELS))
    for ax, arch, panel in zip(axes, ("mlp", "e2e"), ("(a) MLP", "(b) Perceiver")):
        agg = aggregate(pd.read_csv(res / f"{arch}_metrics.csv"), metric)
        for k, (eval_set, name) in enumerate(E3_DISTRIBUTIONS):
            sub = agg[agg.eval_set == eval_set].set_index("model").reindex(E3_MODELS)
            ax.bar(x + (k - (n_dist - 1) / 2) * width, sub["mean"], width * 0.86, yerr=sub["std"],
                   color=E3_COLOR[eval_set], label=name,
                   error_kw={"elinewidth": 0.8, "ecolor": INK, "capsize": 1.5, "capthick": 0.8})
        ax.set_ylim(0.5, 1.0)
        ax.set_yticks(np.arange(0.5, 1.01, 0.1))
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0, colors=INK_MUTED)
        ax.tick_params(axis="x", length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(INK_MUTED)
        ax.axvline(len(TASKS) - 0.5, color=INK_MUTED, linewidth=0.6, linestyle=(0, (3, 3)))
        ax.set_title(panel, loc="left", color=INK, fontsize=9)
        ax.set_ylabel(METRIC_LABEL.get(metric, metric), color=INK_MUTED)
    axes[-1].set_xticks(x, [E3_MODEL_LABEL[m] for m in E3_MODELS], color=INK_MUTED)
    axes[-1].set_xlabel("training data of the model", color=INK_MUTED)
    axes[0].legend(frameon=False, ncol=n_dist, loc="lower right", bbox_to_anchor=(1.0, 1.0),
                   handlelength=1.2, columnspacing=1.2)
    fig.savefig(out / f"e3_{metric}_paper.pdf", bbox_inches="tight")
    fig.savefig(out / f"e3_{metric}_paper.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)


def plot_e3_paper_dots(metric, out):
    """Dot version of the E3 figure: per model the three distributions as connected points.

    Above every model the spread (max - min of the three distribution means) is printed.
    Error bars are the std over seeds. The y-axis does not start at chance level because
    dots, unlike bars, do not encode the value by length.
    """
    res = results_dir()
    aggs = {arch: aggregate(pd.read_csv(res / f"{arch}_metrics.csv"), metric) for arch in ("mlp", "e2e")}
    lows = [(a["mean"] - a["std"])[a.eval_set.isin(E3_COLOR)].min() for a in aggs.values()]
    ymin = np.floor(min(lows) * 20) / 20

    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 8})
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.8), sharex=True, gridspec_kw={"hspace": 0.32})
    offsets = np.array([-0.18, 0.0, 0.18])
    x = np.arange(len(E3_MODELS))
    for ax, arch, panel in zip(axes, ("mlp", "e2e"), ("(a) MLP", "(b) Perceiver")):
        agg = aggs[arch]
        mean = agg.pivot(index="model", columns="eval_set", values="mean").reindex(
            index=E3_MODELS, columns=[d for d, _ in E3_DISTRIBUTIONS])
        std = agg.pivot(index="model", columns="eval_set", values="std").reindex(
            index=E3_MODELS, columns=[d for d, _ in E3_DISTRIBUTIONS])
        for i in range(len(E3_MODELS)):
            ax.plot(x[i] + offsets, mean.iloc[i], color=INK_MUTED, linewidth=0.9, zorder=2)
            spread = mean.iloc[i].max() - mean.iloc[i].min()
            ax.text(x[i], 1.04, f"Δ {spread:.3f}", transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=7, color=INK)
        for k, (eval_set, name) in enumerate(E3_DISTRIBUTIONS):
            ax.errorbar(x + offsets[k], mean[eval_set], yerr=std[eval_set], fmt="none",
                        ecolor=E3_COLOR[eval_set], elinewidth=0.9, zorder=3)
            ax.plot(x + offsets[k], mean[eval_set], "o", markersize=5.5, color=E3_COLOR[eval_set],
                    markeredgecolor="#ffffff", markeredgewidth=0.8, label=name, zorder=4)
        ax.set_ylim(ymin, 1.005)
        ax.set_xlim(-0.5, len(E3_MODELS) - 0.5)
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0, colors=INK_MUTED)
        ax.tick_params(axis="x", length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(INK_MUTED)
        ax.axvline(len(TASKS) - 0.5, color=INK_MUTED, linewidth=0.6, linestyle=(0, (3, 3)))
        ax.set_title(panel, loc="left", color=INK, fontsize=9, pad=16)
        ax.set_ylabel(METRIC_LABEL.get(metric, metric), color=INK_MUTED)
    axes[-1].set_xticks(x, [E3_MODEL_LABEL[m] for m in E3_MODELS], color=INK_MUTED)
    axes[-1].set_xlabel("training data of the model", color=INK_MUTED)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, ncol=len(names), loc="lower right",
               bbox_to_anchor=(0.9, 0.9), handletextpad=0.3, columnspacing=1.2)
    fig.savefig(out / f"e3_{metric}_paper_dots.pdf", bbox_inches="tight")
    fig.savefig(out / f"e3_{metric}_paper_dots.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)


def plot_bars(df, arch, metric, eval_sets, name, out):
    agg = aggregate(df, metric)
    models = [m for m in MODEL_ORDER if m in set(agg.model)]
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for k, model in enumerate(models):
        sub = agg[agg.model == model].set_index("eval_set").reindex(eval_sets)
        x = np.arange(len(eval_sets)) + (k - (len(models) - 1) / 2) * width
        ax.bar(x, sub["mean"], width * 0.88, yerr=sub["std"], color=MODEL_COLOR[model],
               label=label(model), error_kw={"elinewidth": 1, "ecolor": INK_MUTED, "capsize": 2})
    ax.set_xticks(range(len(eval_sets)), eval_sets, color=INK_MUTED)
    ax.set_ylabel(f"{metric} (chance level = 0.5)", color=INK_MUTED)
    ax.set_ylim(0.5, 1.0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_title(f"{name} · {ARCH_NAME[arch]} · {metric} by training model", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              title="training data", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"{name.lower()}_{arch}_{metric}.png", dpi=200)
    fig.savefig(out / f"{name.lower()}_{arch}_{metric}.pdf")
    plt.close(fig)


def plot_pfi(pfi, eval_choice, column, out):
    if eval_choice == "own":
        sub = pfi[pfi.eval_set == pfi.model.where(pfi.model != "pretrained_full", "ref")]
    else:
        sub = pfi[pfi.eval_set == eval_choice]
    agg = sub.groupby(["feature", "model"])[column].agg(["mean", "std"]).reset_index().fillna({"std": 0.0})
    models = [m for m in MODEL_ORDER if m in set(agg.model)]
    mean = agg.pivot(index="feature", columns="model", values="mean").reindex(index=FEATURE_NAMES, columns=models)
    std = agg.pivot(index="feature", columns="model", values="std").reindex(index=FEATURE_NAMES, columns=models)
    fig, ax = plt.subplots(figsize=(max(6.0, 2.5 + 1.1 * len(models)), 6.4))
    image = heatmap(ax, mean.to_numpy(), std.to_numpy(), FEATURE_NAMES, models,
                    f"MLP permutation importance ({column})\nevaluated on: {eval_choice}", vmin=0.0)
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(out / f"pfi_mlp_{column}_{eval_choice}.png", dpi=200)
    fig.savefig(out / f"pfi_mlp_{column}_{eval_choice}.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", nargs="+", default=["auc", "accuracy", "f1"])
    parser.add_argument("--paper", action="store_true", help="only the combined E1 figures for the manuscript")
    args = parser.parse_args()

    res = results_dir()
    out = res / "plots"
    out.mkdir(parents=True, exist_ok=True)
    if args.paper:
        for metric in args.metrics:
            plot_e1_paper(metric, out)
            plot_e3_paper(metric, out)
            plot_e3_paper_dots(metric, out)
        plot_pfi_paper(out)
        plot_pfi_paper(out, relative=True)
        print("paper figures in", out)
        return
    for arch in ("mlp", "e2e"):
        path = res / f"{arch}_metrics.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for metric in args.metrics:
            plot_e1(df, arch, metric, out)
            plot_bars(df, arch, metric, ["ref"], "E2", out)
            plot_bars(df, arch, metric, ["hard", "easy", "uniform"], "E3", out)

    pfi_path = res / "mlp_pfi.csv"
    if pfi_path.exists():
        pfi = pd.read_csv(pfi_path)
        for choice in ["own", "ref", "hard", "easy", "uniform"]:
            plot_pfi(pfi, choice, "delta_accuracy", out)
    print("plots in", out)


if __name__ == "__main__":
    main()
