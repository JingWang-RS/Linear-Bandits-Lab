"""Define reward noise distributions, including Gaussian, Student-t, Pareto, Lomax, and Fisk."""

import math
from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.special import betaln, gammaln


@lru_cache(maxsize=128)
def noise_moment_root(distribution, exponent, *, df=None, shape=None, centered=True):
    """Return the absolute noise moment root, subtracting its mean when centered."""
    if not np.isfinite(exponent) or not 0 < exponent <= 1:
        raise ValueError("moment_exponent must be in (0, 1]")
    p = 1 + exponent
    if distribution == "none":
        return 0.0
    if distribution == "gaussian":
        return float(np.exp((p / 2 * np.log(2) + gammaln((p + 1) / 2) - 0.5 * np.log(np.pi)) / p))
    if distribution == "student-t":
        if df is None or not np.isfinite(df) or df <= p:
            raise ValueError("Student degrees_of_freedom must exceed the requested moment order")
        log_moment = p / 2 * np.log(df) + gammaln((p + 1) / 2) + gammaln((df - p) / 2)
        log_moment -= 0.5 * np.log(np.pi) + gammaln(df / 2)
        return float(np.exp(log_moment / p))
    if distribution not in {"pareto", "lomax", "fisk"}:
        raise ValueError(f"unsupported noise distribution {distribution!r}")
    if shape is None or not np.isfinite(shape) or shape <= p:
        raise ValueError("noise shape must exceed the requested moment order")
    if not centered:
        if distribution == "pareto":
            log_moment = np.log(shape) - np.log(shape - p)
        elif distribution == "lomax":
            log_moment = gammaln(1 + p) + gammaln(shape - p) - gammaln(shape)
        else:
            log_moment = gammaln(1 + p / shape) + gammaln(1 - p / shape)
        return float(np.exp(log_moment / p))
    if distribution in {"pareto", "lomax"}:
        # Lomax+1 is Pareto; centering gives the same distribution at equal shape and scale.
        mean = shape / (shape - 1)
        negative, error = quad(
            lambda u: shape * (mean - 1) ** (p + 1) / (1 + (mean - 1) * u) ** (shape + 1),
            0,
            1,
            weight="alg",
            wvar=(0, p),
            epsabs=1e-10,
            epsrel=1e-10,
        )
        positive = np.exp(np.log(shape) + (p - shape) * np.log(mean) + betaln(p + 1, shape - p))
        moment = negative + positive
    else:
        mean = math.pi / shape / math.sin(math.pi / shape)
        constant = shape * mean ** (p + shape)
        negative, error_negative = quad(
            lambda u: constant / (1 + (mean * u) ** shape) ** 2,
            0,
            1,
            weight="alg",
            wvar=(shape - 1, p),
            epsabs=1e-10,
            epsrel=1e-10,
        )
        positive, error_positive = quad(
            lambda u: constant / (u**shape + mean**shape) ** 2,
            0,
            1,
            weight="alg",
            wvar=(shape - p - 1, p),
            epsabs=1e-10,
            epsrel=1e-10,
        )
        moment = negative + positive
        error = error_negative + error_positive
    if not np.isfinite(moment) or moment <= 0 or error > 1e-8 * moment:
        raise ArithmeticError("centered noise moment integration failed its error check")
    return float(moment ** (1 / p))


def noise_mean(distribution, *, shape=None):
    """Mean of the uncentered distribution at unit scale."""
    if distribution == "pareto":
        return shape / (shape - 1)
    if distribution == "lomax":
        return 1 / (shape - 1)
    if distribution == "fisk":
        return math.pi / shape / math.sin(math.pi / shape)
    return 0.0


def sample_centered_noise(rng, distribution, *, df=None, shape=None, centered=True):
    """Sample reward noise; centered=False preserves the ICML appendix's positive noise."""
    if distribution == "none":
        return 0.0
    if distribution == "gaussian":
        return float(rng.normal())
    if distribution == "student-t":
        return float(rng.standard_t(df))
    if distribution == "pareto":
        return float(rng.pareto(shape) + 1 - (shape / (shape - 1) if centered else 0))
    if distribution == "lomax":
        return float(rng.pareto(shape) - (1 / (shape - 1) if centered else 0))
    if distribution == "fisk":
        u = float(rng.uniform())
        if not 0 < u < 1:
            raise ArithmeticError("Fisk inverse-CDF sampling requires an interior uniform draw")
        mean = noise_mean(distribution, shape=shape) if centered else 0.0
        return float((u / (1 - u)) ** (1 / shape) - mean)
    raise ValueError(f"unsupported noise distribution {distribution!r}")
