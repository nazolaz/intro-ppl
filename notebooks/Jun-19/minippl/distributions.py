"""Distribution objects for FOPPL/HOPPL (numpy-only, no scipy).

Every distribution supports:
    d.sample(rng)        draw a value using a numpy Generator
    d.log_prob(x)        log density / log mass at x (float; -inf outside support)

Distributions used as *variational guides* (BBVI, ch. 4.4) additionally support
an unconstrained parameter vector:
    d.params()           np.ndarray of unconstrained parameters
    d.with_params(theta) new instance from an unconstrained vector
    d.grad_log_prob(x)   gradient of log_prob(x) w.r.t. the unconstrained params

Constructor names match the book: normal, beta, gamma (shape/rate), dirichlet,
discrete, bernoulli, flip, uniform-continuous, uniform, exponential, poisson,
uniform-discrete, log-normal.
"""

from __future__ import annotations

import math

import numpy as np

LOG2PI = math.log(2.0 * math.pi)


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def _softmax(v):
    v = np.asarray(v, dtype=float)
    m = v.max()
    e = np.exp(v - m)
    return e / e.sum()


class Distribution:
    name = "distribution"

    def sample(self, rng: np.random.Generator):
        raise NotImplementedError

    def log_prob(self, x) -> float:
        raise NotImplementedError

    # --- optimizable-parameter interface (only some families) ---
    def params(self) -> np.ndarray:
        raise NotImplementedError(f"{self.name} is not an optimizable guide")

    def with_params(self, theta: np.ndarray) -> "Distribution":
        raise NotImplementedError(f"{self.name} is not an optimizable guide")

    def grad_log_prob(self, x) -> np.ndarray:
        raise NotImplementedError(f"{self.name} is not an optimizable guide")

    def __repr__(self):
        return f"({self.name} {' '.join(repr(p) for p in self._repr_params())})"

    def _repr_params(self):
        return []


class Normal(Distribution):
    name = "normal"

    def __init__(self, mu, sigma):
        self.mu = float(mu)
        self.sigma = float(sigma)
        if self.sigma <= 0:
            raise ValueError("normal: sigma must be > 0")

    def sample(self, rng):
        return float(rng.normal(self.mu, self.sigma))

    def log_prob(self, x):
        z = (float(x) - self.mu) / self.sigma
        return -0.5 * (LOG2PI + z * z) - math.log(self.sigma)

    # unconstrained params: (mu, log sigma)
    def params(self):
        return np.array([self.mu, math.log(self.sigma)])

    def with_params(self, theta):
        return Normal(theta[0], math.exp(theta[1]))

    def grad_log_prob(self, x):
        z = (float(x) - self.mu) / self.sigma
        return np.array([z / self.sigma, z * z - 1.0])

    def _repr_params(self):
        return [self.mu, self.sigma]


class LogNormal(Distribution):
    name = "log-normal"

    def __init__(self, mu, sigma):
        self.mu = float(mu)
        self.sigma = float(sigma)
        if self.sigma <= 0:
            raise ValueError("log-normal: sigma must be > 0")

    def sample(self, rng):
        return float(math.exp(rng.normal(self.mu, self.sigma)))

    def log_prob(self, x):
        x = float(x)
        if x <= 0:
            return -math.inf
        z = (math.log(x) - self.mu) / self.sigma
        return -0.5 * (LOG2PI + z * z) - math.log(self.sigma) - math.log(x)

    def params(self):
        return np.array([self.mu, math.log(self.sigma)])

    def with_params(self, theta):
        return LogNormal(theta[0], math.exp(theta[1]))

    def grad_log_prob(self, x):
        z = (math.log(float(x)) - self.mu) / self.sigma
        return np.array([z / self.sigma, z * z - 1.0])

    def _repr_params(self):
        return [self.mu, self.sigma]


class Uniform(Distribution):
    name = "uniform-continuous"

    def __init__(self, a, b):
        self.a, self.b = float(a), float(b)
        if not self.b > self.a:
            raise ValueError("uniform-continuous: requires b > a")

    def sample(self, rng):
        return float(rng.uniform(self.a, self.b))

    def log_prob(self, x):
        if self.a <= float(x) <= self.b:
            return -math.log(self.b - self.a)
        return -math.inf

    def _repr_params(self):
        return [self.a, self.b]


class Exponential(Distribution):
    name = "exponential"

    def __init__(self, rate):
        self.rate = float(rate)
        if self.rate <= 0:
            raise ValueError("exponential: rate must be > 0")

    def sample(self, rng):
        return float(rng.exponential(1.0 / self.rate))

    def log_prob(self, x):
        x = float(x)
        if x < 0:
            return -math.inf
        return math.log(self.rate) - self.rate * x

    def _repr_params(self):
        return [self.rate]


class Beta(Distribution):
    name = "beta"

    def __init__(self, alpha, beta):
        self.alpha, self.beta = float(alpha), float(beta)
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError("beta: parameters must be > 0")

    def sample(self, rng):
        return float(rng.beta(self.alpha, self.beta))

    def log_prob(self, x):
        x = float(x)
        if not 0.0 < x < 1.0:
            return -math.inf
        a, b = self.alpha, self.beta
        logB = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        return (a - 1) * math.log(x) + (b - 1) * math.log1p(-x) - logB

    def _repr_params(self):
        return [self.alpha, self.beta]


class Gamma(Distribution):
    """Gamma with *shape/rate* parameterization (as in the book / Anglican)."""

    name = "gamma"

    def __init__(self, shape, rate):
        self.shape, self.rate = float(shape), float(rate)
        if self.shape <= 0 or self.rate <= 0:
            raise ValueError("gamma: parameters must be > 0")

    def sample(self, rng):
        return float(rng.gamma(self.shape, 1.0 / self.rate))

    def log_prob(self, x):
        x = float(x)
        if x <= 0:
            return -math.inf
        a, r = self.shape, self.rate
        return a * math.log(r) - math.lgamma(a) + (a - 1) * math.log(x) - r * x

    def _repr_params(self):
        return [self.shape, self.rate]


class Poisson(Distribution):
    name = "poisson"

    def __init__(self, lam):
        self.lam = float(lam)
        if self.lam <= 0:
            raise ValueError("poisson: rate must be > 0")

    def sample(self, rng):
        return int(rng.poisson(self.lam))

    def log_prob(self, k):
        k = int(k)
        if k < 0:
            return -math.inf
        return k * math.log(self.lam) - self.lam - math.lgamma(k + 1)

    def _repr_params(self):
        return [self.lam]


class Bernoulli(Distribution):
    """Returns booleans (true/false), as `flip` does in the book."""

    name = "flip"

    def __init__(self, p):
        self.p = float(p)
        if not 0.0 <= self.p <= 1.0:
            raise ValueError("flip: p must be in [0,1]")

    def sample(self, rng):
        return bool(rng.random() < self.p)

    def log_prob(self, x):
        x = bool(x)
        p = self.p
        if x:
            return math.log(p) if p > 0 else -math.inf
        return math.log1p(-p) if p < 1 else -math.inf

    # unconstrained param: logit
    def params(self):
        p = min(max(self.p, 1e-12), 1 - 1e-12)
        return np.array([math.log(p / (1 - p))])

    def with_params(self, theta):
        return Bernoulli(_sigmoid(float(theta[0])))

    def grad_log_prob(self, x):
        return np.array([(1.0 if bool(x) else 0.0) - self.p])

    def _repr_params(self):
        return [self.p]


class Discrete(Distribution):
    """Categorical over {0, 1, ..., K-1} with probability vector p."""

    name = "discrete"

    def __init__(self, probs):
        p = np.asarray(list(probs), dtype=float).ravel()
        if (p < 0).any() or p.sum() <= 0:
            raise ValueError("discrete: invalid probability vector")
        self.probs = p / p.sum()

    def sample(self, rng):
        return int(rng.choice(len(self.probs), p=self.probs))

    def log_prob(self, k):
        k = int(k)
        if 0 <= k < len(self.probs) and self.probs[k] > 0:
            return math.log(self.probs[k])
        return -math.inf

    # unconstrained params: logits
    def params(self):
        return np.log(np.clip(self.probs, 1e-12, None))

    def with_params(self, theta):
        return Discrete(_softmax(theta))

    def grad_log_prob(self, k):
        onehot = np.zeros(len(self.probs))
        onehot[int(k)] = 1.0
        return onehot - self.probs

    def _repr_params(self):
        return [list(self.probs)]


class UniformDiscrete(Distribution):
    """Uniform over integers {lo, ..., hi-1} (book: uniform-discrete lo hi)."""

    name = "uniform-discrete"

    def __init__(self, lo, hi):
        self.lo, self.hi = int(lo), int(hi)
        if self.hi <= self.lo:
            raise ValueError("uniform-discrete: requires hi > lo")

    def sample(self, rng):
        return int(rng.integers(self.lo, self.hi))

    def log_prob(self, k):
        if self.lo <= int(k) < self.hi:
            return -math.log(self.hi - self.lo)
        return -math.inf

    def _repr_params(self):
        return [self.lo, self.hi]


class Dirichlet(Distribution):
    name = "dirichlet"

    def __init__(self, alphas):
        a = np.asarray(list(alphas), dtype=float).ravel()
        if (a <= 0).any():
            raise ValueError("dirichlet: alphas must be > 0")
        self.alphas = a

    def sample(self, rng):
        return [float(v) for v in rng.dirichlet(self.alphas)]

    def log_prob(self, x):
        x = np.asarray(list(x), dtype=float).ravel()
        if len(x) != len(self.alphas) or (x <= 0).any():
            return -math.inf
        a = self.alphas
        logB = float(np.sum([math.lgamma(v) for v in a]) - math.lgamma(a.sum()))
        return float(np.sum((a - 1) * np.log(x)) - logB)

    def _repr_params(self):
        return [list(self.alphas)]


# ---------------------------------------------------------------------------
# constructor table (primitive names visible inside programs)
# ---------------------------------------------------------------------------

def _discrete_ctor(*args):
    # accept (discrete [p1 p2 ...]) or (discrete p1 p2 ...)
    if len(args) == 1 and isinstance(args[0], (list, np.ndarray)):
        return Discrete(args[0])
    return Discrete(args)


def _dirichlet_ctor(*args):
    if len(args) == 1 and isinstance(args[0], (list, np.ndarray)):
        return Dirichlet(args[0])
    return Dirichlet(args)


DISTRIBUTIONS = {
    "normal": lambda mu, sigma: Normal(mu, sigma),
    "log-normal": lambda mu, sigma: LogNormal(mu, sigma),
    "beta": lambda a, b: Beta(a, b),
    "gamma": lambda shape, rate: Gamma(shape, rate),
    "exponential": lambda rate: Exponential(rate),
    "uniform-continuous": lambda a, b: Uniform(a, b),
    "uniform": lambda a, b: Uniform(a, b),
    "poisson": lambda lam: Poisson(lam),
    "bernoulli": lambda p: Bernoulli(p),
    "flip": lambda p: Bernoulli(p),
    "discrete": _discrete_ctor,
    "categorical": _discrete_ctor,
    "uniform-discrete": lambda lo, hi: UniformDiscrete(lo, hi),
    "dirichlet": _dirichlet_ctor,
}


def make_guide(d: Distribution) -> Distribution:
    """An optimizable variational guide with the same support as d (BBVI)."""
    if isinstance(d, Normal):
        return Normal(d.mu, d.sigma)
    if isinstance(d, LogNormal):
        return LogNormal(d.mu, d.sigma)
    if isinstance(d, (Gamma, Exponential, Beta)):
        # positive support -> log-normal moment-ish initialization
        return LogNormal(0.0, 1.0)
    if isinstance(d, Bernoulli):
        return Bernoulli(d.p)
    if isinstance(d, Discrete):
        return Discrete(d.probs.copy())
    raise NotImplementedError(
        f"no optimizable guide family for distribution {d.name}"
    )
