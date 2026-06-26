"""
Genera le figure CONCETTUALI per le slide di clustering (dati sintetici, non WDBC):
  - kmeans_concept.png      cluster trovati da k-means + centroidi
  - elbow_concept.png       metodo del gomito (distorsione vs K)
  - silhouette_concept.png  silhouette plot (validazione interna)

L'esperimento specifico su WDBC vive nel notebook `notebooks/clustering_wdbc.ipynb`.

Eseguire dalla root:
    ./venv/bin/python figures/clustering_figures.py
"""

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.kmeans import KMeans

RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

COLORS = np.array(["tab:blue", "tab:green", "tab:red", "tab:purple"])


def _blobs(seed=1):
    """Tre 'blob' gaussiani sintetici in 2D."""
    rng = np.random.default_rng(seed)
    centers = [(-2.2, 0.0), (2.0, 1.6), (0.2, -2.6)]
    return np.vstack([rng.normal(c, 0.62, (60, 2)) for c in centers])


def fig_kmeans_concept():
    X = _blobs()
    km = KMeans(n_clusters=3, random_state=0).fit(X)
    plt.figure(figsize=(6.2, 4.6))
    plt.scatter(X[:, 0], X[:, 1], c=COLORS[km.labels_], s=18, alpha=0.7)
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                marker="X", s=240, c="black", edgecolors="white", linewidths=1.5,
                label="centroidi $\\mu_k$")
    plt.title("K-means: punti assegnati al centroide piu' vicino")
    plt.xlabel("$x_1$"); plt.ylabel("$x_2$")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _save("kmeans_concept.png")


def fig_elbow_concept():
    X = _blobs()
    Ks = range(1, 9)
    J = [KMeans(n_clusters=k, n_init=5, random_state=0).fit(X).inertia_ for k in Ks]
    plt.figure(figsize=(6.2, 4.4))
    plt.plot(list(Ks), J, "-o", color="tab:blue")
    plt.scatter([3], [J[2]], s=160, facecolors="none", edgecolors="tab:red", linewidths=2)
    plt.annotate("gomito (K=3)", xy=(3, J[2]), xytext=(4.2, J[2] + 0.15 * (J[0] - J[-1])),
                 arrowprops=dict(arrowstyle="->", color="tab:red"), color="tab:red")
    plt.xlabel("numero di cluster K"); plt.ylabel(r"distorsione media $J$")
    plt.title("Metodo del gomito: scegliere K")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _save("elbow_concept.png")


def _silhouette_samples(X, labels):
    """Silhouette s(i) per ogni punto (per il silhouette plot)."""
    D = np.sqrt(np.maximum(
        (X ** 2).sum(1, keepdims=True) - 2 * X @ X.T + (X ** 2).sum(1), 0.0))
    uniq = np.unique(labels)
    s = np.zeros(len(X))
    for i in range(len(X)):
        same = labels == labels[i]; same[i] = False
        if same.sum() == 0:
            continue
        a = D[i, same].mean()
        b = np.min([D[i, labels == c].mean() for c in uniq if c != labels[i]])
        s[i] = (b - a) / max(a, b)
    return s


def fig_silhouette_concept():
    X = _blobs()
    km = KMeans(n_clusters=3, random_state=0).fit(X)
    s = _silhouette_samples(X, km.labels_)

    plt.figure(figsize=(6.2, 4.6))
    y0 = 0
    for k in np.unique(km.labels_):
        vals = np.sort(s[km.labels_ == k])
        plt.barh(np.arange(y0, y0 + len(vals)), vals, color=COLORS[k], alpha=0.8)
        y0 += len(vals) + 8
    plt.axvline(s.mean(), color="tab:red", ls="--", lw=1.5,
                label=f"media = {s.mean():.2f}")
    plt.xlabel("coefficiente di silhouette $s(i)$"); plt.ylabel("punti (per cluster)")
    plt.title("Silhouette plot: cluster compatti e separati")
    plt.yticks([]); plt.legend(loc="lower right"); plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    _save("silhouette_concept.png")


def _save(name):
    out = os.path.join(RESULTS, name)
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  salvato: {out}")


def main():
    print("Figure CONCETTUALI per le slide di clustering (dati sintetici):")
    fig_kmeans_concept()
    fig_elbow_concept()
    fig_silhouette_concept()


if __name__ == "__main__":
    main()
