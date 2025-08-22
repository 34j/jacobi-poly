import numpy as np
from array_api._2024_12 import Array
from array_api_compat import array_namespace

from jacobi_poly._main import log_jacobi_normalization_constant


def minus_1_power(x: Array, /) -> Array:
    return 1 - 2 * (x % 2)


def jacobi_triplet_integral(
    alpha1: Array,
    alpha2: Array,
    alpha3: Array | None,
    beta1: Array,
    beta2: Array,
    beta3: Array | None,
    n1: Array,
    n2: Array,
    n3: Array,
    *,
    normalized: bool,
) -> Array:
    r"""
    Integral of three Jacobi polynomials.

    .. math::
        \int_{-1}^{1} P_n^{(\alpha_1, \beta_1)}(x) P_n^{(\alpha_2, \beta_2)}(x)
        P_n^{(\alpha_3, \beta_3)}(x) dx

    The special case (alpha_a = beta_a)
    would be Gaunt coefficients (for associated Legendre polynomials).

    Parameters
    ----------
    alpha1 : Array
        The alpha parameter of the first Jacobi polynomial.
    alpha2 : Array
        The alpha parameter of the second Jacobi polynomial.
    alpha3 : Array | None
        The alpha parameter of the third Jacobi polynomial.
        Must be alpha_1 + alpha_2.
    beta1 : Array
        The beta parameter of the first Jacobi polynomial.
    beta2 : Array
        The beta parameter of the second Jacobi polynomial.
    beta3 : Array | None
        The beta parameter of the third Jacobi polynomial.
        Must be beta_1 + beta_2.
    n1 : Array
        The order of the first Jacobi polynomial.
    n2 : Array
        The order of the second Jacobi polynomial.
    n3 : Array
        The order of the third Jacobi polynomial.
    normalized : bool
        Whether all Jacobi polynomials are normalized.
        The computation is faster when True.

    Returns
    -------
    Array
        The integral of the three Jacobi polynomials.

    Raises
    ------
    ValueError
        If the sum of the orders is not an integer.
        If alpha3 is not None and alpha3 is not alpha1 + alpha2.
        If beta3 is not None and beta3 is not beta1 + beta2.

    """
    xp = array_namespace(alpha1, alpha2, alpha3, beta1, beta2, beta3, n1, n2, n3)
    from py3nj import wigner3j

    if alpha3 is None:
        alpha3 = alpha1 + alpha2
    elif xp.any(alpha3 != alpha1 + alpha2):
        raise ValueError(f"{alpha3=} should be {(alpha1 + alpha2)=}")
    if beta3 is None:
        beta3 = beta1 + beta2
    elif xp.any(beta3 != beta1 + beta2):
        raise ValueError(f"{beta3=} should be {(beta1 + beta2)=}")

    alphas = xp.stack(xp.broadcast_arrays(alpha1, alpha2, alpha3), axis=0)
    betas = xp.stack(xp.broadcast_arrays(beta1, beta2, beta3), axis=0)
    ns = xp.stack(xp.broadcast_arrays(n1, n2, n3), axis=0)
    del alpha1, alpha2, alpha3, beta1, beta2, beta3, n1, n2, n3

    # wigner arguments
    Ls2 = 2 * ns + alphas + betas
    Ms2 = alphas + betas
    Ns2 = betas - alphas
    # check if Ls2, Ms2, Ns2 are integers
    if xp.any(Ls2 != xp.round(Ls2)) or xp.any(Ms2 != xp.round(Ms2)) or xp.any(Ns2 != xp.round(Ns2)):
        raise ValueError(f"The sum of the orders should be an integer. {Ls2=}, {Ms2=}, {Ns2=}")
    # round and cast to int
    Ls2, Ms2, Ns2 = xp.round(Ls2), xp.round(Ms2), xp.round(Ns2)
    Ls2, Ms2, Ns2 = xp.astype(Ls2, int), xp.astype(Ms2, int), xp.astype(Ns2, int)

    # coefficients
    # note that there is no need to sqrt the normalization constant
    logcoefs = [
        (
            0
            if normalized
            else -log_jacobi_normalization_constant(
                alpha=alphas[i, ...], beta=betas[i, ...], n=ns[i, ...]
            )
        )
        + 0.5 * xp.log(2 * ns[i, ...] + alphas[i, ...] + betas[i, ...] + 1)
        for i in range(3)
    ]
    phase = minus_1_power(-Ls2[0] + Ls2[1] - betas[2, ...])
    coef = phase / np.sqrt(2) * xp.exp(xp.sum(xp.stack(logcoefs), axis=0))
    return coef * xp.asarray(
        wigner3j(
            Ls2[0, ...],
            Ls2[1, ...],
            Ls2[2, ...],
            Ms2[0, ...],
            Ms2[1, ...],
            -Ms2[2, ...],
            ignore_invalid=True,
        )
        * wigner3j(
            Ls2[0, ...],
            Ls2[1, ...],
            Ls2[2, ...],
            Ns2[0, ...],
            Ns2[1, ...],
            -Ns2[2, ...],
            ignore_invalid=True,
        ),
        device=coef.device,
        dtype=coef.dtype,
    )
