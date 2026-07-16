from __future__ import annotations

import torch

from graphatlas.nn.charts import make_reparameterization


def test_reparameterizations_are_exactly_invertible_and_have_jvps():
    x = torch.randn(9, 4)
    v = torch.randn(9, 4, 3)
    for kind in ("affine", "asinh_affine", "triangular_coupling", "radial"):
        transform = make_reparameterization(kind, 4, x.device, x.dtype, 3, 0.6)
        y, jvp = transform.linearize_forward(x)
        restored, inverse_jvp = transform.linearize_inverse(y)
        assert torch.allclose(restored, x, atol=2e-5, rtol=2e-5)
        assert torch.isfinite(jvp(v)).all() and torch.isfinite(inverse_jvp(jvp(v))).all()
