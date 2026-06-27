"""
K-means -- implementazione da zero (solo numpy).

Notazione:
  m            = numero di esempi
  n            = numero di feature
  K            = numero di cluster
  mu_k         = centroide del cluster k
  c^(i)        = indice del cluster a cui e' assegnato l'esempio i

Algoritmo, si ripete fino a convergenza:
  1) Assegnazione: ogni punto va al centroide piu' vicino
       c^(i) := argmin_k || x^(i) - mu_k ||^2
  2) Aggiornamento: ogni centroide diventa la media dei suoi punti
       mu_k := media dei punti assegnati al cluster k

Obiettivo (distorsione):
  J = (1/m) sum_i || x^(i) - mu_{c^(i)} ||^2
K-means minimizza J; ogni passo non puo' farlo aumentare.

Inizializzazione: K-means converge a un minimo *locale*, quindi:
  - si fanno piu' restart casuali e si tiene quello con J minore;
  - in alternativa k-means++ sceglie centroidi iniziali ben distanziati.
"""

import numpy as np


def _sq_dist(X, centers):
    """Distanze euclidee al quadrato: matrice (m, K) fra ogni punto e ogni centroide."""
    # ||x - mu||^2 = ||x||^2 - 2 x.mu + ||mu||^2
    d2 = (
        (X ** 2).sum(axis=1, keepdims=True)
        - 2 * X @ centers.T
        + (centers ** 2).sum(axis=1)
    )
    return np.maximum(d2, 0.0)   # evita piccoli negativi da errori di arrotondamento


class KMeans:
    """K-means con multi-restart e inizializzazione random o k-means++.

    Parametri
    ---------
    n_clusters : int        -- K, numero di cluster.
    init : 'k-means++' | 'random'
    n_init : int            -- numero di restart; si tiene quello a distorsione minima.
    max_iter : int          -- iterazioni massime per restart.
    tol : float             -- soglia di convergenza sullo spostamento dei centroidi.
    random_state : int|None -- seme per la riproducibilita'.

    Attributi appresi (in fit)
    --------------------------
    cluster_centers_ : ndarray (K, n)   -- centroidi finali (del miglior restart)
    labels_          : ndarray (m,)     -- cluster assegnato a ogni esempio
    inertia_         : float            -- distorsione J del miglior restart
    n_iter_          : int              -- iterazioni del miglior restart
    """

    def __init__(self, n_clusters, init="k-means++", n_init=10,
                 max_iter=300, tol=1e-6, random_state=None):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None

    # ---- inizializzazione dei centroidi ----
    def _init_centers(self, X, rng):
        m = X.shape[0]
        if self.init == "random":
            # K esempi distinti scelti a caso
            idx = rng.choice(m, size=self.n_clusters, replace=False)
            return X[idx].copy()

        # k-means++: centroidi iniziali ben distanziati
        centers = [X[rng.integers(m)]]
        for _ in range(1, self.n_clusters):
            d2 = _sq_dist(X, np.array(centers)).min(axis=1)   # dist^2 al centro piu' vicino
            probs = d2 / d2.sum()                              # prob proporzionale a D(x)^2
            centers.append(X[rng.choice(m, p=probs)])
        return np.array(centers)

    # ---- un singolo run di k-means ----
    def _single_run(self, X, rng):
        centers = self._init_centers(X, rng)
        for it in range(self.max_iter):
            # 1) assegnazione al centroide piu' vicino
            labels = np.argmin(_sq_dist(X, centers), axis=1)

            # 2) aggiornamento dei centroidi
            new_centers = centers.copy()
            for k in range(self.n_clusters):
                pts = X[labels == k]
                if len(pts) > 0:
                    new_centers[k] = pts.mean(axis=0)
                else:
                    # cluster vuoto: re-inizializza sul punto piu' lontano
                    far = np.argmax(_sq_dist(X, centers).min(axis=1))
                    new_centers[k] = X[far]

            shift = np.sqrt(((new_centers - centers) ** 2).sum())
            centers = new_centers
            if shift < self.tol:
                break

        labels = np.argmin(_sq_dist(X, centers), axis=1)
        inertia = _sq_dist(X, centers)[np.arange(len(X)), labels].mean()
        return centers, labels, inertia, it + 1

    def fit(self, X):
        """Esegue n_init restart e tiene quello con distorsione J minima."""
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)

        best = None
        for _ in range(self.n_init):
            centers, labels, inertia, n_iter = self._single_run(X, rng)
            if best is None or inertia < best[2]:
                best = (centers, labels, inertia, n_iter)

        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = best
        return self

    def predict(self, X):
        """Assegna ogni punto al centroide piu' vicino."""
        X = np.asarray(X, dtype=float)
        return np.argmin(_sq_dist(X, self.cluster_centers_), axis=1)
