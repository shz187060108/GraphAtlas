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

    def __init__(self, orthogonal: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, nonlinear: bool = True):
        self.orthogonal = orthogonal
        self.scale = scale
        self.shift = shift
        self.nonlinear = nonlinear

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
        base = torch.asinh(x) if self.nonlinear else x
        return (base * self.scale) @ self.orthogonal.T + self.shift

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
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
