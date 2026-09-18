"""Semi-supervised EM: fit a mixture with a few anchored (labeled) points.

Run with ``python examples/mixture_anchored.py`` (add ``--show`` to open a
window). The figure is written next to this script.

Two unit normals sit 1.5 apart, close enough that no point is certain. A small
random fraction of the points carries its true label; those are the anchors,
passed to ``train(labels=)``. Their responsibilities are pinned to the known
component through the whole fit, which does three things the unlabeled data
cannot: it decides which index is which (no relabeling afterwards), it makes
the fit converge in far fewer iterations, and where the unlabeled likelihood
has several near-equal optima it picks the one that agrees with the labels.

Panels, clockwise from top left:
    1. the data with the anchors as a rug, the true components, and the fit;
    2. how much label information buys: accuracy on the *unlabeled* points
       against the fraction anchored, against the Bayes rate the true
       parameters achieve;
    3. the semi-supervised log-likelihood climbing to convergence;
    4. the fitted P(group 1 | x) against the true posterior.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from _style import HAIRLINE, INK, INK_SOFT, SERIES, SURFACE, style_axes
from em import EM

TRUE_COMPONENTS = [(0.5, 0.0, 1.0), (0.5, 1.5, 1.0)]
N_SAMPLES = 10_000
ANCHOR_FRACTION = 0.02
FRACTIONS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
N_DRAWS = 5


def make_data(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Draw from the mixture and keep the generating component of each point."""
    rng = np.random.default_rng(seed)
    weights = np.array([w for w, _, _ in TRUE_COMPONENTS])
    truth = rng.choice(len(TRUE_COMPONENTS), size=N_SAMPLES, p=weights)
    x = np.empty(N_SAMPLES)
    for k, (_, mu, sigma) in enumerate(TRUE_COMPONENTS):
        mask = truth == k
        x[mask] = rng.normal(mu, sigma, int(mask.sum()))
    return x, truth


def anchor(truth: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Reveal a random fraction of the labels; -1 marks the rest as unknown."""
    labels = np.full(truth.shape, -1)
    picked = rng.choice(truth.size, size=int(round(fraction * truth.size)), replace=False)
    labels[picked] = truth[picked]
    return labels


def true_posterior(grid: np.ndarray) -> np.ndarray:
    """P(group 1 | x) under the generating parameters."""
    joint = np.column_stack([w * norm.pdf(grid, mu, sigma) for w, mu, sigma in TRUE_COMPONENTS])
    return joint[:, 1] / joint.sum(axis=1)


def accuracy_on_unlabeled(model: EM, truth: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of the *unlabeled* points assigned to their generating component.

    With anchors the indices are fixed, so no relabeling is needed. Without,
    the indices are arbitrary and the better of the two orderings is reported.
    """
    unlabeled = labels < 0
    hits = (model.classify() == truth)[unlabeled].mean()
    if np.any(labels >= 0):
        return float(hits)
    return float(max(hits, 1.0 - hits))


def sweep(x: np.ndarray, truth: np.ndarray, seed: int) -> dict[float, float]:
    """Mean accuracy on the unlabeled points, per anchor fraction, over draws."""
    rng = np.random.default_rng(seed)
    scores = {}
    for fraction in FRACTIONS:
        draws = []
        for _ in range(N_DRAWS if fraction > 0 else 1):
            labels = anchor(truth, fraction, rng)
            model = EM("normal", seed=0).train(
                x, n_groups=2, n_init=4, labels=labels if fraction > 0 else None
            )
            draws.append(accuracy_on_unlabeled(model, truth, labels))
        scores[fraction] = float(np.mean(draws))
    return scores


def plot_fit(ax, x: np.ndarray, labels: np.ndarray, model: EM) -> None:
    """Histogram, the true components, the anchored fit, and the anchors."""
    grid = np.linspace(x.min(), x.max(), 800)
    ax.hist(x, bins=70, density=True, color="#d9d8d4", edgecolor=SURFACE, linewidth=0.4)

    components = np.exp(model.loglike(grid, with_priors=False)) * model.weights_
    top = 0.0
    for k, (w, mu, sigma) in enumerate(TRUE_COMPONENTS):
        ax.plot(grid, w * norm.pdf(grid, mu, sigma), color=INK_SOFT, linewidth=1.2,
                linestyle=(0, (4, 3)), label="true component" if k == 0 else None)
    for k in range(model.weights_.size):
        curve = components[:, k]
        ax.plot(grid, curve, color=SERIES[k], linewidth=2.0, label=f"group {k}, fitted")
        top = max(top, curve.max())
        ax.annotate(
            f"group {k}\nmu={model.params_[k, 0]:.2f}",
            xy=(grid[np.argmax(curve)], curve.max()), xytext=(0, 8),
            textcoords="offset points", ha="center", fontsize=8, color=INK_SOFT,
        )

    # The anchors: a rug in the colour of the component they are pinned to.
    for k in range(model.weights_.size):
        anchored = x[labels == k]
        ax.plot(anchored, np.full(anchored.size, -0.012 - 0.012 * k), "|",
                color=SERIES[k], markersize=6, markeredgewidth=1.0)
    ax.annotate(
        f"{int((labels >= 0).sum())} anchors", xy=(x.max(), -0.018), ha="right",
        va="center", fontsize=8, color=INK_SOFT,
    )
    ax.set_ylim(-0.035, top * 1.45)
    style_axes(ax, "Data, anchors and the fitted mixture", "x", "density")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SOFT, loc="upper right")


def plot_sweep(ax, scores: dict[float, float], bayes: float) -> None:
    """Accuracy on the unlabeled points, as the anchored fraction grows."""
    fractions = sorted(scores)
    # The fractions span two decades, so they sit at even positions and the
    # tick labels carry the values; otherwise 0.5%, 1% and 2% collide.
    positions = np.arange(len(fractions))
    ax.axhline(bayes, color=HAIRLINE, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.annotate(
        f"Bayes rate with the true parameters: {bayes:.1%}",
        xy=(positions[-1], bayes), xytext=(0, 5), textcoords="offset points",
        ha="right", fontsize=8, color=INK_SOFT,
    )
    ax.plot(positions, [scores[f] for f in fractions], color=SERIES[0], linewidth=2.0,
            marker="o", markersize=6)
    ax.annotate(
        "0% is plain EM,\nrelabeled to match", xy=(0, scores[0.0]), xytext=(10, -22),
        textcoords="offset points", fontsize=8, color=INK_SOFT,
    )
    low = min(scores.values())
    ax.set_ylim(low - 0.02, bayes + 0.02)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{f:.1%}".replace(".0%", "%") for f in fractions])
    style_axes(ax, "Accuracy on the unlabeled points", "fraction of points anchored",
               "fraction assigned to the generating group")


def plot_convergence(ax, model: EM, plain: EM) -> None:
    """The semi-supervised objective per observation, iteration by iteration."""
    history = np.asarray(model.loglike_history_) / model.X_.size
    ax.plot(np.arange(1, history.size + 1), history, color=SERIES[0], linewidth=2.0)
    ax.annotate(
        f"converged in {model.n_iter_} iterations\n(plain EM took {plain.n_iter_})",
        xy=(history.size, history[-1]), xytext=(-8, -20), textcoords="offset points",
        ha="right", fontsize=8, color=INK_SOFT,
    )
    style_axes(ax, "Semi-supervised log-likelihood per iteration", "EM iteration",
               "mean log-likelihood")


def plot_posterior(ax, x: np.ndarray, model: EM) -> None:
    """P(group 1 | x): the fit against the truth. No relabeling was needed."""
    grid = np.linspace(x.min(), x.max(), 800)
    ax.plot(grid, true_posterior(grid), color=INK_SOFT, linewidth=1.2,
            linestyle=(0, (4, 3)), label="true posterior")
    ax.plot(grid, model.predict_proba(grid)[:, 1], color=SERIES[1], linewidth=2.0,
            label="anchored fit")
    ax.set_ylim(-0.05, 1.18)
    style_axes(ax, "P(group 1 | x)", "x", "posterior probability")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SOFT, loc="upper left")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="open the figure in a window")
    parser.add_argument("--seed", type=int, default=20240905)
    args = parser.parse_args()

    x, truth = make_data(args.seed)
    labels = anchor(truth, ANCHOR_FRACTION, np.random.default_rng(args.seed + 1))

    plain = EM("normal", seed=0).train(x, n_groups=2, n_init=4)
    model = EM("normal", seed=0).train(x, n_groups=2, n_init=4, labels=labels)
    print(model.summary())
    print(f"\ntrue components: {TRUE_COMPONENTS}")
    print(f"accuracy on the unlabeled points: {accuracy_on_unlabeled(model, truth, labels):.3f}")

    bayes = float((np.where(true_posterior(x) > 0.5, 1, 0) == truth).mean())
    scores = sweep(x, truth, args.seed + 2)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor=SURFACE)
    for ax in axes.ravel():
        ax.set_facecolor(SURFACE)
    plot_fit(axes[0, 0], x, labels, model)
    plot_sweep(axes[0, 1], scores, bayes)
    plot_convergence(axes[1, 0], model, plain)
    plot_posterior(axes[1, 1], x, model)
    fig.suptitle(
        f"Semi-supervised EM: two overlapping normals, {ANCHOR_FRACTION:.0%} of "
        f"{N_SAMPLES:,} points anchored",
        fontsize=13, color=INK, x=0.02, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = Path(__file__).with_suffix(".png")
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    print(f"\nwrote {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
