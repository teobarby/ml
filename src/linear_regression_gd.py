"""
Linear Regression con Batch Gradient Descent -- implementazione da zero (numpy).

Notazione:
  m            = numero di esempi di training
  n            = numero di feature
  x0 = 1       = feature fittizia per l'intercetta
  theta        = vettore dei parametri, theta in R^(n+1); theta[0] = intercetta (theta_0)
  h_theta(x)   = theta^T x

Funzione di costo:
  J(theta) = (1 / (2m)) * sum_{i=1}^m ( h_theta(x^(i)) - y^(i) )^2
  (il fattore 1/2 semplifica la derivata)

Batch Gradient Descent, update simultaneo:
  theta := theta - alpha * (1/m) * X^T (X theta - y)

Test di convergenza automatico:
  dichiara convergenza se J(theta) diminuisce di meno di epsilon in un'iterazione.

La standardizzazione delle feature va fatta PRIMA, e NON si applica a x0:
x0=1 viene aggiunta dopo lo scaling, nel metodo fit/predict.
"""

import numpy as np


def _add_bias(X):
    """Aggiunge la colonna x0 = 1 (intercetta theta_0)."""
    X = np.asarray(X, dtype=float)
    return np.hstack([np.ones((X.shape[0], 1)), X])


class LinearRegressionGD:
    """Regressione lineare addestrata con Batch Gradient Descent.

    Parametri
    ---------
    alpha : float
        Learning rate. Ampiezza del passo di discesa.
    n_iters : int
        Numero massimo di iterazioni di gradient descent.
    epsilon : float
        Soglia del test di convergenza automatico:
        ci si ferma se |J_prec - J| < epsilon.
    verbose : bool
        Se True stampa J(theta) ogni 100 iterazioni.
    """

    def __init__(self, alpha=0.1, n_iters=2000, epsilon=1e-9, verbose=False):
        self.alpha = alpha
        self.n_iters = n_iters
        self.epsilon = epsilon
        self.verbose = verbose
        self.theta = None
        self.cost_history_ = []
        self.n_iters_run_ = 0

    def fit(self, X, y):
        """Addestra con Batch Gradient Descent.

        X : ndarray (m, n)   -- feature gia' standardizzate, SENZA colonna x0
        y : ndarray (m,)
        """
        X = _add_bias(X)
        y = np.asarray(y, dtype=float).ravel()
        m = X.shape[0]

        self.theta = np.zeros(X.shape[1])
        self.cost_history_ = []

        prev_J = np.inf
        for it in range(self.n_iters):
            # 1) errore: h_theta(x) - y
            error = X @ self.theta - y          # shape (m,)

            # 2) funzione di costo J(theta)
            J = (1.0 / (2 * m)) * np.sum(error ** 2)
            self.cost_history_.append(J)

            # 3) gradiente: (1/m) X^T (X theta - y)
            grad = (1.0 / m) * (X.T @ error)    # shape (n+1,)

            # 4) update simultaneo di tutti i theta_j
            self.theta = self.theta - self.alpha * grad

            if self.verbose and it % 100 == 0:
                print(f"  iter {it:5d} | J(theta) = {J:.6f}")

            # 5) test di convergenza automatico
            if abs(prev_J - J) < self.epsilon:
                self.n_iters_run_ = it + 1
                break
            prev_J = J
        else:
            self.n_iters_run_ = self.n_iters

        return self

    def predict(self, X):
        return _add_bias(X) @ self.theta

    @property
    def intercept_(self):
        """theta_0 (intercetta)."""
        return self.theta[0]

    @property
    def coef_(self):
        """theta_1 ... theta_n (pesi delle feature)."""
        return self.theta[1:]


def normal_equation(X, y):
    """Soluzione in forma chiusa della regressione lineare:

        theta = (X^T X)^(-1) X^T y

    con X aumentata della colonna x0 = 1. Usa pinv per stabilita' numerica.
    Restituisce il vettore theta di shape (n+1,), con theta[0] = intercetta.
    """
    Xb = _add_bias(X)
    y = np.asarray(y, dtype=float).ravel()
    return np.linalg.pinv(Xb) @ y
