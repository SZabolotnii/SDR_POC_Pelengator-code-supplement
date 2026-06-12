"""Basis function library for DSGE feature generation.

All bases are sign-preserving where applicable: phi(x) = sign(x) * |x|^p,
which keeps information about signal polarity (important for sensor and
embedding data alike).

The PATP (Parametrically-Adaptive Transition Polynomial) basis unifies
fractional/linear/integer-power forms via a single transition parameter
alpha in [0, 1]:
    p_i(alpha) = 1/i + (4 - i - 3/i) * alpha + (2*i - 4 + 2/i) * alpha^2

Special cases:
    alpha = 0   -> p_i = 1/i  (fractional, "soft amplifier" for small signals)
    alpha = 0.5 -> p_i = 1    (all linear after first term)
    alpha = 1   -> p_i = i    (classical integer polynomial)
"""

from __future__ import annotations

import itertools

import numpy as np


def _sign_pow(x: np.ndarray, p: float) -> np.ndarray:
    """Sign-preserving power: sign(x) * |x|^p. Robust to negative values."""
    return np.sign(x) * np.abs(x) ** p


def polynomial_basis(x: np.ndarray, n: int = 3) -> np.ndarray:
    """Integer-power basis: x^2, x^3, ..., x^(n+1).

    Best when data are roughly Gaussian-like or have been pre-normalised
    to the [-1, 1] range. Sensitive to outliers because high powers amplify
    extreme values.
    """
    return np.stack([_sign_pow(x, i) for i in range(2, n + 2)], axis=-1)


def fractional_basis(x: np.ndarray, n: int = 3) -> np.ndarray:
    """Fractional basis: |x|^(1/2), |x|^(1/3), ..., |x|^(1/(n+1)) (sign-preserving).

    Best for heavy-tailed / non-Gaussian data — the "soft amplifier" property
    means small signals get boosted while large ones are damped. Empirically
    wins on raw embeddings and sensor data with high kurtosis.
    """
    return np.stack([_sign_pow(x, 1.0 / i) for i in range(2, n + 2)], axis=-1)


def trigonometric_basis(x: np.ndarray, n: int = 3) -> np.ndarray:
    """Trigonometric basis: sin(x), sin(2x), ..., sin(n*x).

    Useful when signals have periodic structure. Note: requires x to be
    pre-scaled (e.g., normalised to roughly [-pi, pi]) — otherwise sin
    saturates and basis loses discriminative power.
    """
    return np.stack([np.sin((i + 1) * x) for i in range(n)], axis=-1)


def robust_basis(x: np.ndarray, n: int = 3) -> np.ndarray:
    """Robust basis: tanh, sigmoid, atan (cycled if n > 3).

    Bounded outputs make this basis robust to extreme outliers. Good first
    choice when you don't know the data distribution and don't want to
    pre-normalise carefully.
    """
    funcs = [np.tanh, lambda v: 1.0 / (1.0 + np.exp(-np.clip(v, -50, 50))), np.arctan]
    return np.stack([funcs[i % 3](x) for i in range(n)], axis=-1)


def log_basis(x: np.ndarray, n: int = 3) -> np.ndarray:
    """Log basis: sign(x) * log(1 + |x|^i), i = 1..n.

    Heavy compression of dynamic range. Useful for signals spanning many
    orders of magnitude (e.g., spectral amplitudes, financial returns).
    """
    return np.stack([np.sign(x) * np.log1p(np.abs(x) ** (i + 1)) for i in range(n)], axis=-1)


def patp_power(i: int, alpha: float) -> float:
    """Compute p_i(alpha) for the i-th PATP basis function.

    p_i(alpha) = 1/i + (4 - i - 3/i) * alpha + (2*i - 4 + 2/i) * alpha^2
    """
    A = 1.0 / i
    B = 4 - i - 3.0 / i
    C = 2 * i - 4 + 2.0 / i
    return A + B * alpha + C * alpha**2


def patp_basis(x: np.ndarray, n: int = 3, alpha: float = 0.5) -> np.ndarray:
    """PATP basis: smooth interpolation between fractional (alpha=0),
    linear (alpha=0.5) and integer (alpha=1) bases.

    Use grid search or cross-validation to find optimal alpha. Recommended
    starting point: alpha = 0.5. Boundary values (alpha = 0 or 1) suggest
    the corresponding discrete basis is sufficient.
    """
    return np.stack(
        [_sign_pow(x, patp_power(i, alpha)) for i in range(2, n + 2)],
        axis=-1,
    )


# Public registry for easy basis selection
BASES = {
    "polynomial": polynomial_basis,
    "fractional": fractional_basis,
    "trigonometric": trigonometric_basis,
    "robust": robust_basis,
    "log": log_basis,
}


def get_basis(name: str, n: int = 3, alpha: float | None = None):
    """Return a callable phi(x) -> array of basis values.

    For PATP, pass name='patp' and alpha in [0, 1]. For other bases, alpha
    is ignored.
    """
    if name == "patp":
        if alpha is None:
            raise ValueError("PATP basis requires alpha in [0, 1].")
        return lambda x: patp_basis(x, n=n, alpha=alpha)
    if name not in BASES:
        raise ValueError(f"Unknown basis '{name}'. Choose from {list(BASES) + ['patp']}.")
    return lambda x: BASES[name](x, n=n)


class TensorProductBasis:
    """Joint basis for multivariate (bivariate / d-variate) DSGE.

    The scalar bases above act on one coordinate at a time, so a per-class
    reconstructor built on them is blind to how coordinates co-vary. This class
    builds the *tensor product* of a 1D base across `d` coordinates, giving a
    joint basis whose Gram matrix carries the cross-coordinate moments
    `E[phi_i(x_a) phi_j(x_b)]` — i.e. the joint-distribution structure.

    Construction. Each coordinate `x_k` gets an augmented column block
    `[1, phi_1(x_k), ..., phi_n(x_k)]` (the leading 1 is the constant term,
    index 0). A joint basis function is a product picking one index per axis:

        Phi_m(x) = prod_k  aug_k[m_k],   m in {0,1,...,n}^d

    The all-zero multi-index (the pure constant) is dropped — it is absorbed
    into the reconstruction intercept. Index 0 means "skip this axis", so the
    result naturally contains marginal-only terms (one axis active) as well as
    cross terms (>= 2 axes active). Because the underlying 1D base starts at
    power 2 (the identity is never in the basis), no joint term trivially
    equals a coordinate, so reconstructing any coordinate stays non-degenerate.

    Output dimension `D = (n + 1)**d - 1`  (d=2, n=3 -> 15). This grows fast in
    `d`; keep `d` small (2-3) and watch `cond(F)`.
    """

    def __init__(self, basis: str = "fractional", n: int = 3, d: int = 2,
                 alpha: float | None = None):
        if d < 2:
            raise ValueError(f"TensorProductBasis needs d >= 2 (got d={d}); "
                             "for d=1 use the scalar bases directly.")
        self.basis_name = basis
        self.n = n
        self.d = d
        self.alpha = alpha
        self.phi1 = get_basis(basis, n=n, alpha=alpha)
        # Multi-indices over {0,...,n}^d, excluding the all-constant term.
        self.multi_indices = [m for m in itertools.product(range(n + 1), repeat=d)
                              if any(m)]

    @property
    def dim(self) -> int:
        """Number of joint basis functions, D = (n+1)**d - 1."""
        return len(self.multi_indices)

    def is_cross_term(self, idx: int) -> bool:
        """True if joint basis function `idx` mixes >= 2 coordinates."""
        return sum(1 for j in self.multi_indices[idx] if j != 0) >= 2

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Map points to the joint basis: (N, d) -> (N, D)."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if X.shape[1] != self.d:
            raise ValueError(f"Expected points with {self.d} coordinates, "
                             f"got shape {X.shape}.")
        N = X.shape[0]
        # Per-axis augmented blocks: [1, phi_1, ..., phi_n], each (N, n+1).
        aug = [np.concatenate([np.ones((N, 1)), self.phi1(X[:, k])], axis=1)
               for k in range(self.d)]
        out = np.empty((N, self.dim), dtype=np.float64)
        for c, m in enumerate(self.multi_indices):
            col = np.ones(N, dtype=np.float64)
            for k in range(self.d):
                col = col * aug[k][:, m[k]]
            out[:, c] = col
        return out
