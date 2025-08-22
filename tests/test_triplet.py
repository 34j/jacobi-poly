import array_api_extra as xpx
import pytest
from array_api._2024_12 import ArrayNamespaceFull
from array_api_compat import to_device
from scipy.special import eval_jacobi, roots_jacobi

from jacobi_poly import jacobi_triplet_integral


@pytest.mark.parametrize("alpha_eq_beta", [True, False])
def test_jacobi_triplet_integral(alpha_eq_beta: bool, xp: ArrayNamespaceFull) -> None:
    n_samples = 20
    alphas = xp.random.integers(0, 5, shape=(3, n_samples))
    alphas[2, ...] = alphas[0, ...] + alphas[1, ...]
    betas = xp.random.integers(0, 5, shape=(3, n_samples))
    betas[2, ...] = betas[0, ...] + betas[1, ...]
    if alpha_eq_beta:
        betas = alphas

    ns = xp.random.integers(0, 5, shape=(3, n_samples))

    expected = []
    for sample in range(n_samples):
        alpha, beta, n = alphas[..., sample], betas[..., sample], ns[..., sample]

        # expected
        x, w = roots_jacobi(24, float(xp.sum(alpha)) / 2, float(xp.sum(beta)) / 2)
        js = [
            eval_jacobi(
                to_device(n[i], "cpu"),
                to_device(alpha[i], "cpu"),
                to_device(beta[i], "cpu"),
                to_device(x, "cpu"),
            )
            for i in range(3)
        ]
        expected.append(xp.sum(js[0] * js[1] * js[2] * w, axis=-1))
    expected = xp.stack(expected, axis=-1)

    # actual
    try:
        actual = jacobi_triplet_integral(
            alpha1=alphas[0, ...],
            alpha2=alphas[1, ...],
            alpha3=alphas[2, ...],
            beta1=betas[0, ...],
            beta2=betas[1, ...],
            beta3=betas[2, ...],
            n1=ns[0, ...],
            n2=ns[1, ...],
            n3=ns[2, ...],
            normalized=False,
        )
    except ImportError:
        pytest.skip("py3nj is not installed")
        return
    expected = xp.astype(expected, actual.dtype, device=actual.device)
    assert xp.all(xpx.isclose(expected, actual, rtol=1e-3, atol=1e-3))
