from typing import Literal

import array_api_extra as xpx
import pytest
from array_api._2024_12 import ArrayNamespaceFull
from array_api_compat import to_device
from scipy.special import eval_gegenbauer, eval_jacobi

from jacobi_poly import binom, gegenbauer_all, jacobi_all, legendre_all


@pytest.mark.parametrize(
    "shape",
    [
        (1,),
        (2, 3),
        (3, 3, 4),
    ],
)
@pytest.mark.parametrize("n_end", [8])
def test_jacobi(shape: tuple[int, ...], n_end: int, xp: ArrayNamespaceFull) -> None:
    alpha = xp.random.random_uniform(low=0, high=5, shape=shape)
    beta = xp.random.random_uniform(low=0, high=5, shape=shape)
    x = xp.random.random_uniform(low=0, high=1, shape=shape) + 1j * xp.random.random_uniform(
        low=0, high=1, shape=shape
    )
    n = xp.arange(n_end)[(None,) * x.ndim + (slice(None),)]
    expected = eval_jacobi(
        to_device(n, "cpu"),
        to_device(alpha[..., None], "cpu"),
        to_device(beta[..., None], "cpu"),
        to_device(x[..., None], "cpu"),
    )
    expected = xp.astype(expected, x.dtype, device=x.device)
    actual = jacobi_all(x, alpha=alpha, beta=beta, n_end=n_end)
    assert xp.all(xpx.isclose(expected, actual, rtol=1e-3, atol=1e-3))


@pytest.mark.parametrize(
    "shape",
    [
        (1,),
        (2, 3),
        (3, 3, 4),
    ],
)
@pytest.mark.parametrize("n_end", [8])
def test_gegenbauer(shape: tuple[int, ...], n_end: int, xp: ArrayNamespaceFull) -> None:
    alpha = xp.random.random_uniform(low=0, high=5, shape=shape)
    x = xp.random.random_uniform(low=0, high=1, shape=shape)
    n = xp.arange(n_end)[(None,) * x.ndim + (slice(None),)]
    expected = eval_gegenbauer(
        to_device(n, "cpu"), to_device(alpha[..., None], "cpu"), to_device(x[..., None], "cpu")
    )
    expected = xp.astype(expected, x.dtype, device=x.device)
    actual = gegenbauer_all(x, alpha=alpha, n_end=n_end)
    assert xp.all(xpx.isclose(expected, actual, rtol=1e-3, atol=1e-3))


@pytest.mark.parametrize(
    "shape",
    [
        (1,),
        (2, 3),
        (3, 3, 4),
    ],
)
@pytest.mark.parametrize("n_end", [8])
@pytest.mark.parametrize("type", ["gegenbauer", "jacobi"])
def test_legendre(
    shape: tuple[int, ...],
    n_end: int,
    type: Literal["gegenbauer", "jacobi"],
    xp: ArrayNamespaceFull,
) -> None:
    d = xp.random.integers(3, 8, shape=shape)
    alpha = (d - 3) / 2
    x = xp.random.random_uniform(low=0, high=1, shape=shape)
    n = xp.arange(n_end)[(None,) * x.ndim + (slice(None),)]
    if type == "jacobi":
        expected = jacobi_all(x, alpha=alpha, beta=alpha, n_end=n_end) / binom(
            n + alpha[..., None], n
        )
    elif type == "gegenbauer":
        expected = gegenbauer_all(x, alpha=alpha + 1 / 2, n_end=n_end) / binom(
            n + 2 * alpha[..., None],
            n,
            # n + d[..., None] - 3, d[..., None] - 3
        )
    else:
        raise ValueError(f"Invalid type {type}")
    expected = xp.astype(expected, x.dtype, device=x.device)
    actual = legendre_all(x, ndim=xp.asarray(d), n_end=n_end)
    assert xp.all(xpx.isclose(expected, actual, rtol=1e-3, atol=1e-3))
