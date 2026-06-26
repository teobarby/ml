"""
Gaussian Naive Bayes -- implementazione da zero (solo numpy).

Notazione del corso (slide "11.BayesianLearning"):
  m              = numero di esempi di training
  n              = numero di feature
  k              = numero di classi
  c_i            = classe i-esima

Modello generativo + regola MAP (slide 4, 7):
  P(c | x) = P(c) P(x | c) / P(x)        (Bayes)
  predici  argmax_c  P(c) P(x | c)        (l'evidenza P(x) e' costante tra le classi)

Assunzione "naive" di indipendenza condizionata (slide 8):
  P(x | c) = prod_{j=1}^{n} P(x_j | c)

Feature continue -> verosimiglianza gaussiana (slide 13, 15, 16):
  P(x_j | c) = N(x_j; mu_{jc}, sigma^2_{jc})
             = 1 / sqrt(2 pi sigma^2) * exp( -(x_j - mu)^2 / (2 sigma^2) )
  con stima a massima verosimiglianza dei parametri (slide 15):
  mu_{jc}     = (1/m_c) sum_{i: y=c} x_j^(i)
  sigma^2_{jc} = (1/m_c) sum_{i: y=c} (x_j^(i) - mu_{jc})^2      (varianza con 1/m)

In pratica si lavora in spazio logaritmico: il prodotto di tante densita' piccole
darebbe underflow numerico, quindi si somma il logaritmo (monotono -> stesso argmax):
  log P(c | x) = log P(c) + sum_j log N(x_j; mu_{jc}, sigma^2_{jc}) + cost.
"""

import numpy as np


class GaussianNaiveBayes:
    """Naive Bayes con verosimiglianze gaussiane (feature continue).

    Attributi appresi (impostati in fit)
    ------------------------------------
    classes_ : ndarray (k,)        -- etichette delle classi, ordinate
    priors_  : ndarray (k,)        -- P(c_i), stimati come frequenza relativa
    theta_   : ndarray (k, n)      -- media mu_{jc} per classe e feature
    var_     : ndarray (k, n)      -- varianza sigma^2_{jc} per classe e feature
    """

    def __init__(self):
        self.classes_ = None
        self.priors_ = None
        self.theta_ = None
        self.var_ = None

    def fit(self, X, y):
        """Apprende priori, medie e varianze per ogni classe (slide 9, 15-16).

        Per ogni classe stima:
          - il priore P(c) come frazione di esempi di quella classe;
          - per ogni feature, media e varianza degli esempi di quella classe.
        Niente discesa del gradiente: i parametri si ottengono in forma chiusa
        (massima verosimiglianza), in una sola passata sui dati.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        m, n = X.shape

        self.classes_ = np.unique(y)
        k = len(self.classes_)

        self.priors_ = np.zeros(k)
        self.theta_ = np.zeros((k, n))
        self.var_ = np.zeros((k, n))

        for idx, c in enumerate(self.classes_):
            Xc = X[y == c]                      # solo gli esempi della classe c
            self.priors_[idx] = Xc.shape[0] / m  # P(c) = #c / m   (slide 9)
            self.theta_[idx] = Xc.mean(axis=0)   # mu_{jc}          (slide 15)
            self.var_[idx] = Xc.var(axis=0)      # sigma^2_{jc} con 1/m (ddof=0)

        return self

    def _joint_log_likelihood(self, X):
        """Calcola log P(c) + sum_j log N(x_j; mu_{jc}, sigma^2_{jc}) per ogni classe.

        Restituisce una matrice (m, k): in posizione (i, c) c'e' il log-posterior
        (a meno della costante log P(x)) dell'esempio i per la classe c.
        """
        X = np.asarray(X, dtype=float)
        m = X.shape[0]
        k = len(self.classes_)
        jll = np.zeros((m, k))

        for idx in range(k):
            # log della densita' gaussiana, sommato sulle feature (assunzione naive)
            log_prior = np.log(self.priors_[idx])
            var = self.var_[idx]
            # log N = -1/2 [ log(2 pi sigma^2) + (x - mu)^2 / sigma^2 ]
            log_likelihood = -0.5 * np.sum(
                np.log(2.0 * np.pi * var) + (X - self.theta_[idx]) ** 2 / var,
                axis=1,
            )
            jll[:, idx] = log_prior + log_likelihood

        return jll

    def predict(self, X):
        """Regola MAP (slide 7): assegna la classe col log-posterior piu' alto."""
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]

    def predict_proba(self, X):
        """Posterior normalizzati P(c | x), via softmax stabile sul log-posterior.

        NB: estensione oltre le slide. La classificazione (predict) usa la sola
        regola MAP -- l'argmax del log-posterior -- e non richiede di normalizzare.
        Qui normalizziamo solo per ottenere probabilita' leggibili P(c | x),
        usate ad es. per scegliere un esempio "netto" nell'EDA del notebook.
        """
        jll = self._joint_log_likelihood(X)
        jll -= jll.max(axis=1, keepdims=True)        # stabilita' numerica
        probs = np.exp(jll)
        return probs / probs.sum(axis=1, keepdims=True)
