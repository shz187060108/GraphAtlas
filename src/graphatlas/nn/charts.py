from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import torch
from torch import nn

from .functional import SmoothMLP


@dataclass
class ChartLinearization:
    coordinate: torch.Tensor
    reconstruction: torch.Tensor
    pull: Callable[[torch.Tensor], torch.Tensor]
    push: Callable[[torch.Tensor], torch.Tensor]


class LocalChart(nn.Module):
    def __init__(self, observation_dim: int, chart_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = SmoothMLP(observation_dim, hidden_dim, chart_dim, layers=3)
        self.decoder = SmoothMLP(chart_dim, hidden_dim, observation_dim, layers=3)

    def linearize(
        self,
        observation: torch.Tensor,
        reparameterization: SmoothDiffeomorphism | None = None,
        chunk_size: int = 0,
    ) -> ChartLinearization:
        base_coordinate, encoder_jvp = self.encoder.linearize(observation, chunk_size)
        if reparameterization is None:
            coordinate = base_coordinate
            pull = encoder_jvp
            decoder_base = base_coordinate
            reconstruction, decoder_jvp = self.decoder.linearize(decoder_base, chunk_size)
            push = decoder_jvp
        else:
            coordinate, rho_jvp = reparameterization.linearize_forward(base_coordinate, chunk_size)
            decoder_base, rho_inverse_jvp = reparameterization.linearize_inverse(coordinate, chunk_size)
            reconstruction, decoder_jvp = self.decoder.linearize(decoder_base, chunk_size)

            def pull(vectors: torch.Tensor) -> torch.Tensor:
                return rho_jvp(encoder_jvp(vectors))

            def push(vectors: torch.Tensor) -> torch.Tensor:
                return decoder_jvp(rho_inverse_jvp(vectors))

        return ChartLinearization(coordinate, reconstruction, pull, push)


class SmoothDiffeomorphism:
    """A smooth global chart reparameterization with exact cached JVPs."""

    def __init__(self, orthogonal: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, nonlinear: bool = True, kind: str | None = None, strength: float = 1.0):
        self.orthogonal = orthogonal
        self.scale = scale
        self.shift = shift
        self.nonlinear = nonlinear
        self.kind = kind or ("asinh_affine" if nonlinear else "affine")
        self.strength = float(strength)

    @classmethod
    def random(
        cls,
        dim: int,
        device: torch.device,
        dtype: torch.dtype,
        seed: int,
        nonlinear: bool = True,
    ) -> SmoothDiffeomorphism:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        matrix = torch.randn(dim, dim, generator=generator, dtype=dtype)
        orthogonal, _ = torch.linalg.qr(matrix)
        scale = torch.empty(dim, dtype=dtype).uniform_(0.65, 1.45, generator=generator)
        shift = torch.randn(dim, generator=generator, dtype=dtype) * 0.2
        return cls(orthogonal.to(device), scale.to(device), shift.to(device), nonlinear=nonlinear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "triangular_coupling":
            split = x.shape[-1] // 2
            if split == 0:
                return x
            left, right = x[..., :split], x[..., split:]
            return torch.cat([left, right + self.strength * torch.tanh(left[..., :right.shape[-1]])], dim=-1)
        if self.kind == "radial":
            return x * torch.sqrt(1.0 + self.strength * x.square().sum(dim=-1, keepdim=True))
        base = torch.asinh(x) if self.nonlinear else x
        return (base * self.scale) @ self.orthogonal.T + self.shift

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        if self.kind == "triangular_coupling":
            split = y.shape[-1] // 2
            if split == 0:
                return y
            left, right = y[..., :split], y[..., split:]
            return torch.cat([left, right - self.strength * torch.tanh(left[..., :right.shape[-1]])], dim=-1)
        if self.kind == "radial":
            if self.strength == 0.0:
                return y
            r2 = y.square().sum(dim=-1, keepdim=True)
            x2 = (torch.sqrt(1.0 + 4.0 * self.strength * r2) - 1.0) / (2.0 * self.strength)
            return y / torch.sqrt(1.0 + self.strength * x2)
        base = ((y - self.shift) @ self.orthogonal) / self.scale
        return torch.sinh(base) if self.nonlinear else base

    @staticmethod
    def _ranges(length: int, chunk_size: int) -> list[tuple[int, int]]:
        if chunk_size <= 0 or chunk_size >= length:
            return [(0, length)]
        return [(start, min(start + chunk_size, length)) for start in range(0, length, chunk_size)]

    def linearize_forward(
        self,
        x: torch.Tensor,
        chunk_size: int = 0,
    ) -> tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
        if self.kind in {"triangular_coupling", "radial"}:
            output = self.forward(x)
            def apply_special(vectors: torch.Tensor) -> torch.Tensor:
                if self.kind == "triangular_coupling":
                    split = x.shape[-1] // 2
                    if split == 0:
                        return vectors
                    left, right = vectors[..., :split, :], vectors[..., split:, :]
                    derivative = self.strength * (1.0 - torch.tanh(x[..., :split]).square())
                    return torch.cat([left, right + derivative[..., :right.shape[-2], None] * left[..., :right.shape[-2], :]], dim=-2)
                scale = torch.sqrt(1.0 + self.strength * x.square().sum(dim=-1, keepdim=True))
                dot = (x[..., :, None] * vectors).sum(dim=-2, keepdim=True)
                return scale[..., None] * vectors + x[..., :, None] * (self.strength * dot / scale[..., None])
            return output, apply_special
        base = torch.asinh(x) if self.nonlinear else x
        output = (base * self.scale) @ self.orthogonal.T + self.shift
        derivative = torch.rsqrt(1.0 + x.square()) if self.nonlinear else torch.ones_like(x)

        def apply(vectors: torch.Tensor) -> torch.Tensor:
            pieces: list[torch.Tensor] = []
            for start, end in self._ranges(x.shape[0], chunk_size):
                tangent = vectors[start:end] * derivative[start:end, :, None]
                tangent = tangent * self.scale[None, :, None]
                pieces.append(torch.einsum("oi,nic->noc", self.orthogonal, tangent))
            return torch.cat(pieces, dim=0)

        return output, apply

    def linearize_inverse(
        self,
        y: torch.Tensor,
        chunk_size: int = 0,
    ) -> tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
        if self.kind in {"triangular_coupling", "radial"}:
            output = self.inverse(y)
            def apply_special(vectors: torch.Tensor) -> torch.Tensor:
                if self.kind == "triangular_coupling":
                    split = y.shape[-1] // 2
                    if split == 0:
                        return vectors
                    left, right = vectors[..., :split, :], vectors[..., split:, :]
                    derivative = self.strength * (1.0 - torch.tanh(y[..., :split]).square())
                    return torch.cat([left, right - derivative[..., :right.shape[-2], None] * left[..., :right.shape[-2], :]], dim=-2)
                if self.strength == 0.0:
                    return vectors
                r2 = y.square().sum(dim=-1, keepdim=True)
                root = torch.sqrt(1.0 + 4.0 * self.strength * r2)
                x2 = (root - 1.0) / (2.0 * self.strength)
                scale = torch.sqrt(1.0 + self.strength * x2)
                g = 1.0 / scale
                dg = -0.5 * self.strength / (scale.pow(3) * root)
                dot = (y[..., :, None] * vectors).sum(dim=-2, keepdim=True)
                return g[..., None] * vectors + y[..., :, None] * (2.0 * dg[..., None] * dot)
            return output, apply_special
        base = ((y - self.shift) @ self.orthogonal) / self.scale
        output = torch.sinh(base) if self.nonlinear else base
        derivative = torch.cosh(base) if self.nonlinear else torch.ones_like(base)

        def apply(vectors: torch.Tensor) -> torch.Tensor:
            pieces: list[torch.Tensor] = []
            for start, end in self._ranges(y.shape[0], chunk_size):
                tangent = torch.einsum("oi,noc->nic", self.orthogonal, vectors[start:end])
                tangent = tangent / self.scale[None, :, None]
                pieces.append(tangent * derivative[start:end, :, None])
            return torch.cat(pieces, dim=0)

        return output, apply


def make_reparameterization(
    kind: str,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    strength: float = 1.0,
) -> SmoothDiffeomorphism:
    """Deterministic exact chart-coordinate transforms used by interventions."""
    if kind not in {"affine", "asinh_affine", "triangular_coupling", "radial"}:
        raise ValueError(f"Unknown reparameterization kind: {kind}")
    if strength < 0:
        raise ValueError("reparameterization strength must be non-negative")
    if strength == 0:
        eye = torch.eye(dim, device=device, dtype=dtype)
        return SmoothDiffeomorphism(eye, torch.ones(dim, device=device, dtype=dtype), torch.zeros(dim, device=device, dtype=dtype), nonlinear=False, kind="affine", strength=0.0)
    base = SmoothDiffeomorphism.random(dim, device, dtype, seed, nonlinear=(kind == "asinh_affine"))
    if kind in {"affine", "asinh_affine"}:
        base.kind, base.strength = kind, strength
        base.scale = 1.0 + (base.scale - 1.0) * strength
        base.shift = base.shift * strength
        return base
    base.kind, base.strength = kind, strength
    return base
