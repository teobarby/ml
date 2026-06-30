"""Preprocessing da zero (solo numpy)."""

import numpy as np


class StandardScaler:
    """Standardizzazione z-score: (x - media) / deviazione_standard.

    La media e la deviazione standard si calcolano SOLO sul train (fit), e poi
    si applicano a train e test (transform). Cosi' si evita il "data leakage":
    nessuna informazione del test entra nel preprocessing.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):   # impara mu e sigma dal train set, in modo da evitare data leakage
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        # evita la divisione per zero su feature costanti
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        return self

    def transform(self, X):    # applica mu e sigma imparati precedentemente
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):    # semplicemente le applica assieme
        return self.fit(X).transform(X)
