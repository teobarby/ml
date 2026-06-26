"""
Genera le figure usate nelle slide di classificazione (WDBC):
  - wdbc_confusion_matrix.png  matrice di confusione sul test set
  - wdbc_gaussians.png         gaussiane per classe sulle feature piu' discriminanti
                               (la "campana" che NB impara per ciascuna classe)
  - wdbc_cv_metrics.png        metriche in 5-fold CV (media +/- deviazione standard)

Tutta l'ANALISI (training, valutazione, confronto con sklearn, cross-validation)
vive nel notebook `notebooks/classificazione_wdbc.ipynb`; qui restano solo i grafici
statici che servono alle slide.

Eseguire dalla root:
    ./venv/bin/python figures/classification_figures.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.naive_bayes import GaussianNaiveBayes
from src.metrics import confusion_matrix, accuracy, precision, recall, f1_score
from src.model_selection import kfold_indices

RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

# colori coerenti per le due classi
C_MAL, C_BEN = "tab:red", "tab:blue"
POS = 0   # classe positiva = maligno (target 0 in WDBC)


def load_data():
    df = pd.read_csv(os.path.join(ROOT, "data", "raw", "wdbc.csv"))
    feat = [c for c in df.columns if c not in ("target", "diagnosis")]
    return df[feat].values, df["target"].values, np.array(feat)


def split(X, y, seed=42, test_frac=0.3):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(test_frac * len(X))
    return idx[n_test:], idx[:n_test]   # train_idx, test_idx


def _gaussian(x, mu, var):
    return np.exp(-(x - mu) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)


# ------------------------------------------------------------------
def fig_concept_gaussians():
    """Figura CONCETTUALE (dati sintetici, non WDBC): una gaussiana per classe e
    il confine di decisione. Serve a illustrare il modello generativo nelle slide."""
    muA, sA = -1.3, 1.0
    muB, sB = 1.4, 1.15
    x = np.linspace(-5, 5.5, 500)
    pA = _gaussian(x, muA, sA ** 2)
    pB = _gaussian(x, muB, sB ** 2)

    # confine di decisione (priori uguali): dove le due densita' si incrociano
    diff = pB - pA
    cross = np.where(np.diff(np.sign(diff)) != 0)[0]
    cross = [c for c in cross if muA < x[c] < muB]
    bnd = x[cross[0]] if cross else 0.0

    plt.figure(figsize=(7.5, 4.4))
    plt.axvspan(x[0], bnd, color=C_MAL, alpha=0.07)
    plt.axvspan(bnd, x[-1], color=C_BEN, alpha=0.07)
    plt.plot(x, pA, color=C_MAL, lw=2.4, label="classe A: $\\mathcal{N}(\\mu_A,\\sigma_A^2)$")
    plt.plot(x, pB, color=C_BEN, lw=2.4, label="classe B: $\\mathcal{N}(\\mu_B,\\sigma_B^2)$")
    plt.axvline(bnd, color="0.3", ls="--", lw=1.5)
    ymax = max(pA.max(), pB.max())
    plt.text(bnd, ymax * 1.02, "confine di decisione", ha="center", fontsize=10, color="0.3")
    plt.text(muA, _gaussian(muA, muA, sA ** 2) * 0.5, "predici A", ha="center", color=C_MAL)
    plt.text(muB, _gaussian(muB, muB, sB ** 2) * 0.5, "predici B", ha="center", color=C_BEN)
    plt.xlabel("una feature $x_j$"); plt.ylabel("densità  $P(x_j\\mid c)$")
    plt.title("Naïve Bayes gaussiano: una campana per classe,\ndecisione = densità (× prior) più alta")
    plt.legend(loc="upper right"); plt.yticks([]); plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    _save("nb_concept_gaussians.png")


# ------------------------------------------------------------------
def fig_confusion_matrix(X, y):
    """Matrice di confusione del Naive Bayes sul test set."""
    tr, te = split(X, y)
    nb = GaussianNaiveBayes().fit(X[tr], y[tr])
    yp = nb.predict(X[te])
    C = confusion_matrix(y[te], yp, labels=[0, 1])   # [[TP,FN],[FP,TN]] con pos=maligno

    labels = ["maligno", "benigno"]
    plt.figure(figsize=(5.2, 4.6))
    plt.imshow(C, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(C[i, j]), ha="center", va="center",
                     fontsize=18,
                     color="white" if C[i, j] > C.max() / 2 else "black")
    plt.xticks([0, 1], labels); plt.yticks([0, 1], labels)
    plt.xlabel("Predetto"); plt.ylabel("Reale")
    acc = accuracy(y[te], yp)
    rec = recall(y[te], yp, POS)
    plt.title(f"Matrice di confusione (test)\naccuracy={acc:.3f}, recall maligno={rec:.3f}")
    plt.tight_layout()
    _save("wdbc_confusion_matrix.png")


def fig_gaussians(X, y, feat_names, n_show=2):
    """Per le feature piu' discriminanti: istogramma per classe + gaussiana appresa."""
    nb = GaussianNaiveBayes().fit(X, y)        # fit su tutto: serve solo per illustrare
    i_mal = list(nb.classes_).index(0)
    i_ben = list(nb.classes_).index(1)

    # separazione = |mu_mal - mu_ben| / sqrt(media delle varianze)
    sep = np.abs(nb.theta_[i_mal] - nb.theta_[i_ben]) / np.sqrt(
        0.5 * (nb.var_[i_mal] + nb.var_[i_ben]))
    top = np.argsort(sep)[::-1][:n_show]

    fig, axes = plt.subplots(1, n_show, figsize=(5.2 * n_show, 4.4))
    axes = np.atleast_1d(axes)
    for ax, j in zip(axes, top):
        xj = X[:, j]
        grid = np.linspace(xj.min(), xj.max(), 300)
        for cls, idx, col, name in [(0, i_mal, C_MAL, "maligno"),
                                    (1, i_ben, C_BEN, "benigno")]:
            ax.hist(xj[y == cls], bins=25, density=True, alpha=0.35, color=col)
            ax.plot(grid, _gaussian(grid, nb.theta_[idx, j], nb.var_[idx, j]),
                    color=col, lw=2.2, label=name)
        ax.set_title(feat_names[j]); ax.set_xlabel("valore"); ax.set_ylabel("densita'")
        ax.legend(); ax.grid(True, alpha=0.3)
    fig.suptitle("Gaussiane apprese per classe (modello generativo di Naïve Bayes)")
    plt.tight_layout()
    _save("wdbc_gaussians.png")


def fig_cv_metrics(X, y):
    """Accuracy, precision, recall, F1 in 5-fold CV: media +/- deviazione standard."""
    scorers = {
        "accuracy":  lambda yt, yp: accuracy(yt, yp),
        "precision": lambda yt, yp: precision(yt, yp, POS),
        "recall":    lambda yt, yp: recall(yt, yp, POS),
        "F1":        lambda yt, yp: f1_score(yt, yp, POS),
    }
    scores = {k: [] for k in scorers}
    for tr, va in kfold_indices(len(X), k=5, random_state=42):
        nb = GaussianNaiveBayes().fit(X[tr], y[tr])
        yp = nb.predict(X[va])
        for k, fn in scorers.items():
            scores[k].append(fn(y[va], yp))

    names = list(scorers)
    means = [np.mean(scores[k]) for k in names]
    stds = [np.std(scores[k]) for k in names]

    plt.figure(figsize=(6.5, 4.2))
    bars = plt.bar(names, means, yerr=stds, capsize=6,
                   color=["tab:blue", "tab:green", "tab:red", "tab:purple"], alpha=0.85)
    for b, mu in zip(bars, means):
        plt.text(b.get_x() + b.get_width() / 2, mu + 0.01, f"{mu:.3f}",
                 ha="center", fontsize=10)
    plt.ylim(0.8, 1.02)
    plt.ylabel("punteggio")
    plt.title("Naïve Bayes su WDBC --- 5-fold CV (media $\\pm$ dev. std)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    _save("wdbc_cv_metrics.png")


def _save(name):
    out = os.path.join(RESULTS, name)
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  salvato: {out}")


def main():
    X, y, feat_names = load_data()
    print("Genero le figure di classificazione:")
    # figura CONCETTUALE per le slide (teoria, dati sintetici)
    fig_concept_gaussians()
    # figure dell'ESPERIMENTO WDBC (usate nel notebook)
    fig_confusion_matrix(X, y)
    fig_gaussians(X, y, feat_names)
    fig_cv_metrics(X, y)


if __name__ == "__main__":
    main()
