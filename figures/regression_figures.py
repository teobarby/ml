"""
Genera le figure usate nelle slide di regressione (CCPP):
  - ccpp_learning_rate.png   effetto del learning rate alpha sulla convergenza
  - ccpp_learning_curve.png  learning curve (diagnosi bias/variance)
  - ccpp_function_AT.png      funzione appresa lungo AT: retta vs + AT^2

Tutta l'ANALISI (training, valutazione, confronto con sklearn, cross-validation,
feature engineering) vive nel notebook `notebooks/regressione_ccpp.ipynb`; qui restano
solo i grafici statici che servono alle slide.

Eseguire dalla root:
    ./venv/bin/python figures/regression_figures.py
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

from src.linear_regression_gd import LinearRegressionGD, normal_equation
from src.preprocessing import StandardScaler
from src.metrics import rmse

RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)


def load_data():
    df = pd.read_csv(os.path.join(ROOT, "data", "raw", "ccpp.csv"))
    return df[["AT", "V", "AP", "RH"]].values, df["PE"].values


def split(X, y, seed=42, test_frac=0.2):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(test_frac * len(X))
    return idx[n_test:], idx[:n_test]   # train_idx, test_idx


# ------------------------------------------------------------------
def fig_learning_rate(X, y):
    """Effetto di alpha: convergenza lenta / buona / divergente (scala log)."""
    tr, te = split(X, y, seed=0)
    sc = StandardScaler().fit(X[tr])
    X_tr, y_tr = sc.transform(X[tr]), y[tr]

    n_iters, ceiling = 150, 1e6
    configs = [
        (0.01, "tab:blue",   r"$\alpha=0.01$ - troppo piccolo (lento)"),
        (0.1,  "tab:green",  r"$\alpha=0.1$ - buono"),
        (0.3,  "tab:orange", r"$\alpha=0.3$ - ottimo"),
        (1.0,  "tab:red",    r"$\alpha=1.0$ - troppo grande (diverge)"),
    ]
    plt.figure(figsize=(8, 5))
    for alpha, color, label in configs:
        m = LinearRegressionGD(alpha=alpha, n_iters=n_iters, epsilon=0.0).fit(X_tr, y_tr)
        hist = np.array(m.cost_history_, dtype=float)
        disp = np.clip(np.nan_to_num(hist, nan=ceiling, posinf=ceiling, neginf=ceiling), 1e-1, ceiling)
        plt.plot(disp, color=color, linewidth=1.8, label=label)
    plt.yscale("log")
    plt.ylim(5, ceiling)
    plt.xlabel("Iterazione")
    plt.ylabel(r"$J(\theta)$ (train, scala log)")
    plt.title(r"Effetto del learning rate $\alpha$ sulla convergenza - CCPP")
    plt.legend(loc="center right")
    plt.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    _save("ccpp_learning_rate.png")


def fig_learning_curve(X, y):
    """Errore train/validation al crescere di m (diagnosi bias/variance)."""
    tr, te = split(X, y, seed=42)
    X_pool, y_pool = X[tr], y[tr]
    X_val, y_val = X[te], y[te]
    sizes = sorted(set(int(s) for s in np.geomspace(10, len(X_pool), 25)))

    tr_err, va_err = [], []
    for m in sizes:
        a, b = [], []
        for r in range(5):
            sub = np.random.default_rng(1000 * m + r).choice(len(X_pool), size=m, replace=False)
            sc = StandardScaler().fit(X_pool[sub])
            mod = LinearRegressionGD(alpha=0.1, n_iters=2000, epsilon=1e-9).fit(sc.transform(X_pool[sub]), y_pool[sub])
            a.append(rmse(y_pool[sub], mod.predict(sc.transform(X_pool[sub]))))
            b.append(rmse(y_val, mod.predict(sc.transform(X_val))))
        tr_err.append(np.mean(a)); va_err.append(np.mean(b))

    plt.figure(figsize=(7, 4.5))
    plt.plot(sizes, tr_err, "-o", ms=3, color="tab:blue", label="Errore di training")
    plt.plot(sizes, va_err, "-o", ms=3, color="tab:orange", label="Errore di validation (CV)")
    plt.xlim(0, 750)
    plt.xlabel("Dimensione del training set $m$")
    plt.ylabel("RMSE (MW)")
    plt.title("Learning curve - regressione CCPP")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout()
    _save("ccpp_learning_curve.png")


def fig_function_at(X, y):
    """Funzione appresa lungo AT (altre feature alla media): retta vs + AT^2."""
    def fit(Xf):
        sc = StandardScaler().fit(Xf)
        theta = normal_equation(sc.transform(Xf), y)
        return lambda Z: theta[0] + sc.transform(Z) @ theta[1:]

    add_at2 = lambda Z: np.hstack([Z, (Z[:, [0]]) ** 2])
    pred_lin = fit(X)
    pred_eng = fit(add_at2(X))

    means = X.mean(axis=0)
    at = np.linspace(X[:, 0].min(), X[:, 0].max(), 200)
    grid = np.tile(means, (200, 1)); grid[:, 0] = at

    plt.figure(figsize=(7.5, 5))
    plt.scatter(X[:, 0], y, s=6, alpha=0.15, color="gray", label="Dati reali")
    plt.plot(at, pred_lin(grid), color="tab:blue", linewidth=2.2, label="Retta (4 feature)")
    plt.plot(at, pred_eng(add_at2(grid)), color="tab:red", linewidth=2.2, label=r"+ $\mathrm{AT}^2$")
    plt.xlabel("AT - temperatura ambiente (°C)")
    plt.ylabel("PE - potenza erogata (MW)")
    plt.title("Funzione di regressione lungo AT\n(V, AP, RH fissate alla media)")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _save("ccpp_function_AT.png")


# ==================================================================
# Figure CONCETTUALI (dati sintetici, non CCPP): servono alle slide teoriche
# ==================================================================
def fig_concept_learning_rate():
    """Convergenza di J(theta) per alpha piccolo / giusto / troppo grande."""
    it = np.arange(0, 60)
    good = 3 + 47 * np.exp(-0.20 * it)
    slow = 3 + 47 * np.exp(-0.04 * it)
    div = 3 + 0.8 * (1.18 ** it)        # passo troppo grande: il costo esplode

    plt.figure(figsize=(6.8, 4.4))
    plt.plot(slow, color="tab:blue", lw=2.2, label=r"$\alpha$ piccolo - lento")
    plt.plot(good, color="tab:green", lw=2.2, label=r"$\alpha$ giusto - rapido")
    plt.plot(div, color="tab:red", lw=2.2, label=r"$\alpha$ troppo grande - diverge")
    plt.yscale("log")
    plt.ylim(1, 1e4)
    plt.xlabel("Iterazione"); plt.ylabel(r"$J(\theta)$ (scala log)")
    plt.title(r"Effetto del learning rate $\alpha$ sulla convergenza")
    plt.legend(); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    _save("concept_learning_rate.png")


def fig_concept_learning_curve():
    """Learning curve generica: errore train/validation al crescere di m."""
    m = np.linspace(5, 300, 40)
    asym = 3.0
    train = asym * (1 - np.exp(-m / 35))      # parte basso, sale verso l'asintoto
    val = asym + 5.0 * np.exp(-m / 35)        # parte alto, scende verso l'asintoto

    plt.figure(figsize=(6.6, 4.4))
    plt.plot(m, train, "-o", ms=3, color="tab:blue", label="errore di training")
    plt.plot(m, val, "-o", ms=3, color="tab:orange", label="errore di validation")
    plt.axhline(asym, ls="--", color="0.6", lw=1)
    plt.annotate("il gap si chiude\n(low variance)", xy=(210, asym + 0.4),
                 fontsize=9, color="0.3")
    plt.xlabel("Dimensione del training set $m$"); plt.ylabel("errore")
    plt.title("Learning curve: come si legge bias/variance")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _save("concept_learning_curve.png")


def fig_concept_feature_engineering():
    """Relazione curva: una retta fa underfit, una feature quadratica la cattura."""
    rng = np.random.default_rng(0)
    x = np.linspace(-3, 3, 70)
    y = 5 - 1.4 * x + 0.55 * x ** 2 + rng.normal(0, 0.7, x.size)
    grid = np.linspace(-3, 3, 200)
    lin = np.polyval(np.polyfit(x, y, 1), grid)
    quad = np.polyval(np.polyfit(x, y, 2), grid)

    plt.figure(figsize=(6.6, 4.4))
    plt.scatter(x, y, s=14, alpha=0.5, color="gray", label="dati")
    plt.plot(grid, lin, color="tab:blue", lw=2.2, label="retta (underfit)")
    plt.plot(grid, quad, color="tab:red", lw=2.2, label=r"+ termine quadratico")
    plt.xlabel("feature $x$"); plt.ylabel("target $y$")
    plt.title("Feature engineering: una relazione curva\nrichiede una feature non lineare")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _save("concept_feature_engineering.png")


def _save(name):
    out = os.path.join(RESULTS, name)
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  salvato: {out}")


def main():
    print("Figure CONCETTUALI per le slide (dati sintetici):")
    fig_concept_learning_rate()
    fig_concept_learning_curve()
    fig_concept_feature_engineering()

    X, y = load_data()
    print("Figure dell'ESPERIMENTO CCPP (usate nel notebook):")
    fig_learning_rate(X, y)
    fig_learning_curve(X, y)
    fig_function_at(X, y)


if __name__ == "__main__":
    main()
