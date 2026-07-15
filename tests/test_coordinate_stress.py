from __future__ import annotations

import torch

from graphatlas.nn.charts import make_reparameterization


def test_zero_strength_is_identity_for_all_families():
    x = torch.randn(5, 4)
    for kind in ("affine", "asinh_affine", "triangular_coupling", "radial"):
        transform = make_reparameterization(kind, 4, x.device, x.dtype, 7, 0.0)
        assert torch.equal(transform.forward(x), x)
        assert torch.equal(transform.inverse(x), x)


def test_stress_strength_is_not_fixed():
    transform = make_reparameterization("radial", 4, torch.device("cpu"), torch.float32, 1, 0.5)
    assert transform.strength == 0.5
