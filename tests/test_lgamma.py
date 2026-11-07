import array_api_extra as xpx
import numpy as np
import pytest
from array_api._2024_12 import ArrayNamespaceFull

from jacobi_poly import binom, lgamma


def test_lgamma_python() -> None:
    assert lgamma(1) == 0
    assert lgamma(2) == 0
    assert lgamma(3) == pytest.approx(np.log(2))
    assert lgamma(4) == pytest.approx(np.log(6))
    assert lgamma(5) == pytest.approx(np.log(24))


def test_lgamma(xp: ArrayNamespaceFull) -> None:
    assert xp.all(
        xpx.isclose(lgamma(xp.asarray([1, 2, 3, 4, 5])), xp.log(xp.asarray([1, 1, 2, 6, 24])))
    )


def test_binom_python() -> None:
    assert binom(5, 2) == pytest.approx(10)
    assert binom(5, 0) == pytest.approx(1)
    assert binom(5, 5) == pytest.approx(1)
    assert binom(6, 3) == pytest.approx(20)


def test_binom(xp: ArrayNamespaceFull) -> None:
    assert xp.all(
        xpx.isclose(
            binom(
                xp.asarray([5, 5, 5, 6], dtype=xp.float64),
                xp.asarray([2, 0, 5, 3], dtype=xp.float64),
            ),
            xp.asarray([10, 1, 1, 20], dtype=xp.float64),
        )
    )
