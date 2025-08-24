from numbers import Number
from typing import overload

from array_api._2024_12 import Array
from array_api_compat import (
    array_namespace,
    is_jax_array,
    is_numpy_array,
    is_torch_array,
)


@overload
def lgamma(x: Array) -> Array: ...
@overload
def lgamma(x: float) -> float: ...  # type: ignore


def lgamma(x: Array | float) -> Array | float:
    """
    Compute the logarithm of the absolute value of the gamma function.

    Parameters
    ----------
    x : Array
        The input array.

    Returns
    -------
    Array
        The logarithm of the absolute value of the gamma function.

    """
    if is_jax_array(x):
        from jax.lax import lgamma as lgamma_jax

        return lgamma_jax(x)
    elif is_torch_array(x):
        from torch import lgamma as lgamma_torch

        return lgamma_torch(x)
    elif is_numpy_array(x) or isinstance(x, Number):
        from scipy.special import gammaln as lgamma_scipy

        return lgamma_scipy(x)
    else:
        xp = array_namespace(x)
        if hasattr(xp, "lgamma"):
            return xp.lgamma(x)
        elif hasattr(xp, "gammaln"):
            return xp.gammaln(x)
        else:
            raise ValueError(
                "The input array must be a JAX, NumPy, or PyTorch array, "
                "or an array with a lgamma or gammaln method."
            )


@overload
def binom(x: Array, y: Array) -> Array: ...
@overload
def binom(x: float, y: float) -> float: ...  # type: ignore


def binom(x: Array | float, y: Array | float) -> Array | float:
    """
    Compute the binomial coefficient.

    Parameters
    ----------
    x : Array
        The first argument.
    y : Array
        The second argument.

    Returns
    -------
    Array
        The binomial coefficient.

    """
    inner = lgamma(x + 1.0) - lgamma(y + 1.0) - lgamma(x - y + 1.0)
    xp = array_namespace(x, y, inner)
    return xp.exp(inner)
