from typing import Any

import array_api_extra as xpx
import numba
import numpy as np
from array_api._2024_12 import Array
from array_api_compat import array_namespace
from numba import complex64, complex128, float32, float64
from numba.cuda import as_cuda_array
from numba.cuda.cudadrv.error import CudaSupportError

from ._lgamma import binom, lgamma


def _jacobi(
    x: Any,
    alpha: Any,
    beta: Any,
    out: np.ndarray[tuple[int], np.dtype[np.floating[Any]]],
    _: Any,
    /,
) -> None:
    # Compute the first two polynomials
    # https://en.wikipedia.org/wiki/Jacobi_polynomials#Special_cases
    n_end = out.shape[0]
    if n_end >= 0:
        out[0] = 1.0
    if n_end >= 1:
        out[1] = (alpha + 1) + (alpha + beta + 2) * (x - 1) / 2

    # Use recurrence relation to compute the rest
    # https://en.wikipedia.org/wiki/Jacobi_polynomials#Recurrence_relations
    for n in range(2, n_end):
        a = n + alpha
        b = n + beta
        c = a + b
        d = (c - 1) * (c * (c - 2) * x + (a - b) * (c - 2 * n)) * out[n - 1] - 2 * (a - 1) * (
            b - 1
        ) * c * out[n - 2]
        out[n] = d / (2 * n * (c - n) * (c - 2))


_numba_args = (
    [
        (float32, float32, float32, float32[:], float32),
        (float64, float64, float64, float64[:], float64),
        (complex64, float32, float32, complex64[:], complex64),
        (complex128, float64, float64, complex128[:], complex128),
        (float32, complex64, complex64, complex64[:], complex64),
        (float64, complex128, complex128, complex128[:], complex128),
        (complex64, complex64, complex64, complex64[:], complex64),
        (complex128, complex128, complex128, complex128[:], complex128),
    ],
    "(),(),(),(n)->()",
)
_jacobi_parallel = numba.guvectorize(*_numba_args, target="parallel", fastmath=True, cache=True)(
    _jacobi
)

try:
    _jacobi_cuda = numba.guvectorize(*_numba_args, target="cuda")(_jacobi)
except CudaSupportError:
    # warnings.warn(
    #     "Numba CUDA support is not available. "
    #     "Falling back to array API implementation.",
    #     UserWarning,
    #     stacklevel=2,
    # )
    _jacobi_cuda = _jacobi


def jacobi_all(
    x: Array,
    *,
    alpha: Array,
    beta: Array,
    n_end: int,
) -> Array:
    """
    Computes the Jacobi polynomials of order {0, ..., n_end - 1} at the points x.

    (...) -> (..., n_end)

    Parameters
    ----------
    x : Array
        X
    alpha : Array
        Alpha
    beta : Array
        Beta
    n_end : int
        The maximum order of the polynomials.

    Returns
    -------
    Array
        The values of the Jacobi polynomials at the points x of order {0,...,n_end-1}.

    """
    xp = array_namespace(x, alpha, beta)
    shape = xpx.broadcast_shapes(x.shape, alpha.shape, beta.shape)
    out = xp.empty((*shape, n_end), dtype=x.dtype, device=x.device)
    if "cuda" in str(x.device):
        x = as_cuda_array(x)
        alpha = as_cuda_array(alpha)
        beta = as_cuda_array(beta)
        out = as_cuda_array(out)
        _jacobi_cuda(x, alpha, beta, out)
    else:
        _jacobi_parallel(x, alpha, beta, out)
    out = xp.asarray(out)
    return out


def log_jacobi_normalization_constant(*, alpha: Array, beta: Array, n: Array) -> Array:
    """
    Computes the log of normalization constant of the Jacobi polynomials of order n.

    Parameters
    ----------
    alpha : Array
        Alpha
    beta : Array
        Beta
    n : Array
        The order of the Jacobi polynomial.

    Returns
    -------
    Array
        The log of the normalization constant.

    """
    xp = array_namespace(alpha, beta, n)
    logupper = xp.log(2 * n + alpha + beta + 1) + lgamma(n + alpha + beta + 1.0) + lgamma(n + 1.0)
    loglower = np.log(2) * (alpha + beta + 1) + lgamma(n + alpha + 1.0) + lgamma(n + beta + 1.0)
    return 0.5 * (logupper - loglower)


def jacobi_normalization_constant(*, alpha: Array, beta: Array, n: Array) -> Array:
    """
    Computes the normalization constant of the Jacobi polynomials of order n.

    Parameters
    ----------
    alpha : Array
        Alpha
    beta : Array
        Beta
    n : Array
        The order of the Jacobi polynomial.

    Returns
    -------
    Array
        The normalization constant.

    """
    xp = array_namespace(alpha, beta, n)
    return xp.exp(log_jacobi_normalization_constant(alpha=alpha, beta=beta, n=n))


def gegenbauer_all(x: Array, *, alpha: Array, n_end: int) -> Array:
    """
    Computes the Gegenbauer polynomials of order {0, ..., n_end - 1} at the points x.

    (...) -> (..., n_end)

    Parameters
    ----------
    x : Array
        X
    alpha : Array
        Alpha
    n_end : int
        The maximum order of the polynomials.

    Returns
    -------
    Array
        The values of the Gegenbauer polynomials at the points x of order {0,...,n_end-1}.

    """
    xp = array_namespace(x, alpha)
    x, alpha = xp.broadcast_arrays(x, alpha)
    n = xp.arange(0, n_end, dtype=x.dtype, device=x.device)[(None,) * x.ndim + (slice(None),)]
    alpha = xp.asarray(alpha - 1 / 2, dtype=x.dtype, device=x.device)
    alpha_ = alpha[..., None]
    log_coef = xp.astype(
        lgamma(2.0 * alpha_ + 1.0 + n)
        - lgamma(2.0 * alpha_ + 1.0)
        - (lgamma(alpha_ + 1.0 + n) - lgamma(alpha_ + 1.0)),
        x.dtype,
    )
    return xp.exp(log_coef) * jacobi_all(x, alpha=alpha, beta=alpha, n_end=n_end)


def legendre_all(x: Array, *, ndim: Array, n_end: int) -> Array:
    """
    Computes the generalized Legendre polynomials of order {0, ..., n_end - 1} at the points x.

    The shape of x should be broadcastable to a common shape.

    (...) -> (..., n_end)

    Parameters
    ----------
    x : Array
        X
    ndim : int
        The dimension of the space.
    n_end : int
        The maximum order of the polynomials.

    Returns
    -------
    Array
        The values of the generalized Legendre polynomials at the points x of order {0,...,n_end-1}.

    """
    # return jacobi(x, alpha=xp.asarray((ndim-3)/2),
    # beta=xp.asarray((ndim-3)/2), n_end=n_end)
    xp = array_namespace(x, ndim)
    x, ndim = xp.broadcast_arrays(x, xp.asarray(ndim))
    n = xp.arange(0, n_end, dtype=x.dtype, device=x.device)[(None,) * x.ndim + (slice(None),)]
    return xp.where(
        ndim[..., None] == 2,
        # Chebyshev polynomials of the first kind
        xp.cos(n * xp.acos(x)[..., None]),
        gegenbauer_all(x, alpha=(ndim - 2) / 2, n_end=n_end)
        / binom(n + ndim[..., None] - 3, ndim[..., None] - 3),
    )
