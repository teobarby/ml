"""Strumenti di validazione da zero (solo numpy)."""

import numpy as np


def kfold_indices(n, k=5, shuffle=True, random_state=None):
    """Genera gli indici di train/validation per una k-fold cross-validation.

    Restituisce una lista di k tuple (train_idx, val_idx). Ogni campione finisce
    in validation esattamente una volta.
    """
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(idx)
    folds = np.array_split(idx, k)
    splits = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        splits.append((train_idx, val_idx))
    return splits


def cross_validate(model_factory, X, y, scorer, preprocess=None,
                   k=5, random_state=None):
    """Esegue una k-fold CV e restituisce la lista dei punteggi per fold.

    Parametri
    ---------
    model_factory : callable senza argomenti che restituisce un modello "fresco"
        (con metodi fit/predict). Serve un modello nuovo a ogni fold.
    X, y : dati.
    scorer : callable (y_true, y_pred) -> float (es. una metrica).
    preprocess : callable opzionale (X_train, X_val) -> (X_train_t, X_val_t)
        per fittare lo scaler SOLO sul train di ogni fold (niente leakage).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    scores = []
    for train_idx, val_idx in kfold_indices(len(X), k=k, random_state=random_state):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        if preprocess is not None:
            X_tr, X_val = preprocess(X_tr, X_val)
        model = model_factory()
        model.fit(X_tr, y_tr)
        scores.append(scorer(y_val, model.predict(X_val)))
    return np.array(scores)
