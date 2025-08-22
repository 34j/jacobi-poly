# Jacobi Poly

<p align="center">
  <a href="https://github.com/34j/jacobi-poly/actions/workflows/ci.yml?query=branch%3Amain">
    <img src="https://img.shields.io/github/actions/workflow/status/34j/jacobi-poly/ci.yml?branch=main&label=CI&logo=github&style=flat-square" alt="CI Status" >
  </a>
  <a href="https://jacobi-poly.readthedocs.io">
    <img src="https://img.shields.io/readthedocs/jacobi-poly.svg?logo=read-the-docs&logoColor=fff&style=flat-square" alt="Documentation Status">
  </a>
  <a href="https://codecov.io/gh/34j/jacobi-poly">
    <img src="https://img.shields.io/codecov/c/github/34j/jacobi-poly.svg?logo=codecov&logoColor=fff&style=flat-square" alt="Test coverage percentage">
  </a>
</p>
<p align="center">
  <a href="https://github.com/astral-sh/uv">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv">
  </a>
  <a href="https://github.com/astral-sh/ruff">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff">
  </a>
  <a href="https://github.com/pre-commit/pre-commit">
    <img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white&style=flat-square" alt="pre-commit">
  </a>
</p>
<p align="center">
  <a href="https://pypi.org/project/jacobi-poly/">
    <img src="https://img.shields.io/pypi/v/jacobi-poly.svg?logo=python&logoColor=fff&style=flat-square" alt="PyPI Version">
  </a>
  <img src="https://img.shields.io/pypi/pyversions/jacobi-poly.svg?style=flat-square&logo=python&amp;logoColor=fff" alt="Supported Python versions">
  <img src="https://img.shields.io/pypi/l/jacobi-poly.svg?style=flat-square" alt="License">
</p>

---

**Documentation**: <a href="https://jacobi-poly.readthedocs.io" target="_blank">https://jacobi-poly.readthedocs.io </a>

**Source Code**: <a href="https://github.com/34j/jacobi-poly" target="_blank">https://github.com/34j/jacobi-poly </a>

---

Compute Jacobi polynomial from order 0 to N efficiently in NumPy / PyTorch / JAX / array API

## Installation

Install this via pip (or your favourite package manager):

```shell
pip install jacobi-poly
```

## Usage

```python
from jacobi_poly import jacobi_all, gegenbauer_all, legendre_all
import torch

# Both CPU (Numba) and CUDA (Numba for CUDA) are supported.
torch.set_default_device("cuda")

x = torch.asarray(1.0)  # the points to evaluate at
alpha = torch.asarray(2.0)
beta = torch.asarray(3.0)
n_end = 4  # from order 0 to order 3

# Jacobi polynomial from order 0 to 3
jacobi = jacobi_all(x, alpha=alpha, beta=beta, n_end=n_end)
print(jacobi)
# tensor([ 1.,  3.,  6., 10.], device='cuda:0')

# Gegenbauer polynomial from order 0 to 3
gegenbauer = gegenbauer_all(x, alpha=alpha, n_end=n_end)
print(gegenbauer)
# tensor([ 1.0000,  4.0000, 10.0000, 20.0000], device='cuda:0')

# Generalized Legendre polynomial from order 0 to 3
# If ndim == 3, same as Legendre polynomial
legendre = legendre_all(x, n_end=n_end, ndim=3)
print(legendre)
# tensor([1., 1., 1., 1.], device='cuda:0')
```

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- prettier-ignore-start -->
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- markdownlint-disable -->
<!-- markdownlint-enable -->
<!-- ALL-CONTRIBUTORS-LIST:END -->
<!-- prettier-ignore-end -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

## Credits

[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-orange.json)](https://github.com/copier-org/copier)

This package was created with
[Copier](https://copier.readthedocs.io/) and the
[browniebroke/pypackage-template](https://github.com/browniebroke/pypackage-template)
project template.
