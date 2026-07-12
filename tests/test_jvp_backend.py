from __future__ import annotations

import torch

from graphatlas.nn.charts import SmoothDiffeomorphism
from graphatlas.nn.functional import SmoothMLP, channel_jvp


def test_cached_analytic_jvp_matches_autograd_jvp():
    torch.manual_seed(11)
    network = SmoothMLP(5, 7, 3, layers=3)
    x = torch.randn(13, 5)
    vectors = torch.randn(13, 5, 4)
    _, analytic = network.linearize(x, chunk_size=5)
    expected = channel_jvp(network, x, vectors, chunk_size=5)
    assert torch.allclose(analytic(vectors), expected, atol=2e-6, rtol=2e-5)


def test_diffeomorphism_forward_inverse_jvps_cancel():
    torch.manual_seed(12)
    x = torch.randn(17, 3)
    vectors = torch.randn(17, 3, 2)
    rep = SmoothDiffeomorphism.random(3, x.device, x.dtype, seed=91, nonlinear=True)
    y, forward_jvp = rep.linearize_forward(x, chunk_size=6)
    recovered, inverse_jvp = rep.linearize_inverse(y, chunk_size=6)
    assert torch.allclose(recovered, x, atol=2e-6, rtol=2e-5)
    assert torch.allclose(inverse_jvp(forward_jvp(vectors)), vectors, atol=3e-6, rtol=3e-5)
