"""Metriche di valutazione da zero (solo numpy)."""

import numpy as np


def mse(y_true, y_pred):
    """Mean Squared Error."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return np.mean((y_pred - y_true) ** 2)


def rmse(y_true, y_pred):
    """Root Mean Squared Error (stessa unita' di misura del target)."""
    return np.sqrt(mse(y_true, y_pred))


def r2_score(y_true, y_pred):
    """Coefficiente di determinazione R^2 (slide Multivar 21-22).

    R^2 = 1 - SS_res / SS_tot
    dove SS_res = sum (y - y_hat)^2  e  SS_tot = sum (y - media_y)^2.
    E' la proporzione di varianza totale dei dati spiegata dal modello:
    vale 1 per una predizione perfetta, 0 se predice sempre la media.
    Le slide lo chiamano anche "correlation/determination coefficient".
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot


def regression_report(y_true, y_pred):
    """Restituisce un dict con le metriche principali di regressione."""
    return {
        "MSE": mse(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


# ======================================================================
# Metriche di classificazione (slide "8.ML_Evaluation")
# ======================================================================

def confusion_matrix(y_true, y_pred, labels=None):
    """Matrice di confusione (slide 32).

    Restituisce la matrice C dove C[i, j] = numero di esempi di classe reale
    labels[i] predetti come labels[j]. Per un problema binario con labels=[pos, neg]
    la matrice e':  [[TP, FN], [FP, TN]].
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = np.asarray(labels)
    idx = {lab: i for i, lab in enumerate(labels)}
    C = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        C[idx[t], idx[p]] += 1
    return C


def accuracy(y_true, y_pred):
    """Frazione di classificazioni corrette: (TP + TN) / m (slide 33)."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return np.mean(y_true == y_pred)


def _binary_counts(y_true, y_pred, pos_label):
    """Conta TP, FP, FN, TN rispetto alla classe positiva pos_label."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
    fp = np.sum((y_pred == pos_label) & (y_true != pos_label))
    fn = np.sum((y_pred != pos_label) & (y_true == pos_label))
    tn = np.sum((y_pred != pos_label) & (y_true != pos_label))
    return tp, fp, fn, tn


def precision(y_true, y_pred, pos_label=1):
    """Precision = TP / (TP + FP): dei predetti positivi, quanti lo sono davvero (slide 41)."""
    tp, fp, _, _ = _binary_counts(y_true, y_pred, pos_label)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(y_true, y_pred, pos_label=1):
    """Recall = TP / (TP + FN): dei positivi reali, quanti ne troviamo (slide 41).

    Nel contesto medico e' la sensibilita': la quota di malati effettivamente
    individuati (importante non mancarne -- slide 44).
    """
    tp, _, fn, _ = _binary_counts(y_true, y_pred, pos_label)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(y_true, y_pred, pos_label=1):
    """F1 = media armonica di precision e recall (slide 47)."""
    p = precision(y_true, y_pred, pos_label)
    r = recall(y_true, y_pred, pos_label)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def classification_report(y_true, y_pred, pos_label=1):
    """Dict con le metriche principali di classificazione rispetto a pos_label."""
    return {
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred, pos_label),
        "recall": recall(y_true, y_pred, pos_label),
        "f1": f1_score(y_true, y_pred, pos_label),
    }


# ======================================================================
# Metriche di clustering (slide "14.Clustering")
# ======================================================================

def purity(y_true, labels):
    """Purezza (slide 69): validazione ESTERNA contro le classi vere.

        P = (1/N) sum_k  max_d |cluster_k AND classe_d|

    Per ogni cluster si prende la classe di maggioranza; si sommano questi conteggi
    e si divide per N. Vale 1 se ogni cluster contiene una sola classe.
    """
    y_true = np.asarray(y_true).ravel()
    labels = np.asarray(labels).ravel()
    N = len(y_true)
    total = 0
    for k in np.unique(labels):
        classes_in_k = y_true[labels == k]
        # conteggio della classe di maggioranza nel cluster k
        total += np.max(np.bincount(
            np.searchsorted(np.unique(y_true), classes_in_k)))
    return total / N


def silhouette(X, labels, return_samples=False):
    """Coefficiente di silhouette (slide 73-76): validazione INTERNA.

    Per ogni punto i:
        a(i) = distanza media dai punti dello STESSO cluster
        b(i) = minima, fra gli altri cluster, della distanza media a quel cluster
        s(i) = (b - a) / max(a, b)
    Per default restituisce la media di s(i): vicino a 1 = cluster compatti e
    ben separati. Con return_samples=True restituisce l'array dei coefficienti
    per punto (utile per il diagramma a barre della silhouette).
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels).ravel()
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return np.zeros(len(X)) if return_samples else 0.0

    # matrice delle distanze euclidee (m, m)
    D = np.sqrt(np.maximum(
        (X ** 2).sum(1, keepdims=True) - 2 * X @ X.T + (X ** 2).sum(1), 0.0))

    s = np.zeros(len(X))
    for i in range(len(X)):
        same = labels == labels[i]
        same[i] = False
        n_same = same.sum()
        if n_same == 0:                  # cluster con un solo punto: s = 0
            s[i] = 0.0
            continue
        a = D[i, same].mean()
        b = np.min([D[i, labels == c].mean() for c in uniq if c != labels[i]])
        s[i] = (b - a) / max(a, b)
    return s if return_samples else s.mean()
