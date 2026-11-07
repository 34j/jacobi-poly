import pytest
from array_api._2024_12 import ArrayNamespaceFull


@pytest.fixture(scope="session", params=["numpy", "torch"])
def xp(request: pytest.FixtureRequest) -> ArrayNamespaceFull:
    """Get the array namespace for the given backend.

    Returns
    -------
    ArrayNamespaceFull
        The selected array namespace.

    Raises
    ------
    ValueError
        If the backend is unknown.

    """
    backend = request.param
    if backend == "numpy":
        from array_api_compat import numpy as xp  # ruff:ignore[import-outside-top-level]

        rng = xp.random.default_rng()

        def random_uniform(low=0, high=1, shape=None):
            return rng.random(shape) * (high - low) + low

        def integers(low, high=None, shape=None):
            return rng.integers(low, high, size=shape)

        xp.random.random_uniform = random_uniform
        xp.random.integers = integers
    elif backend == "torch":
        from array_api_compat import torch as xp  # ruff:ignore[import-outside-top-level]

        def random_uniform(low=0, high=1, shape=None):
            return xp.rand(shape) * (high - low) + low

        def integers(low, high=None, shape=None):
            return xp.randint(low, high, size=shape)

        xp.random.random_uniform = random_uniform
        xp.random.integers = integers
        import torch  # ruff:ignore[import-outside-top-level]

        if torch.cuda.is_available():
            xp.set_default_device("cuda")
    else:
        raise ValueError(f"Unknown backend: {backend}")
    return xp
