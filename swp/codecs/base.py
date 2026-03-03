"""Codec interface: Protocol + ReconstructionResult TypedDict."""

from __future__ import annotations

from typing import Protocol, TypedDict, runtime_checkable

import torch
from torch import Tensor


class ReconstructionResult(TypedDict):
    """Standardised output of any codec's reconstruct() call."""

    recon_wav: Tensor
    """Reconstructed waveform, shape [T]. Always on CPU."""

    recon_sr: int
    """Sample rate of recon_wav (codec native SR, e.g. 24000 for EnCodec)."""

    n_tokens: int
    """Total number of discrete code assignments: Q * T_frames."""

    tokens: Tensor | None
    """Raw token codes, shape [Q, T_frames], or None if codec doesn't expose them.
    Q = number of codebook levels; T_frames = temporal frames.
    Used by reencode_consistency(); set to None if unavailable."""


@runtime_checkable
class AudioCodec(Protocol):
    """Interface every codec must satisfy.

    The pipeline never imports a concrete codec class; it only uses this Protocol.
    Concrete codecs do not need to inherit from AudioCodec — structural typing is enough.
    """

    @property
    def sample_rate(self) -> int:
        """Native sample rate the codec operates at (e.g. 24000 for EnCodec)."""
        ...

    @property
    def name(self) -> str:
        """Short identifier used in output paths (e.g. "encodec", "dac")."""
        ...

    def reconstruct(self, wav: Tensor, sr: int) -> ReconstructionResult:
        """Full encode-decode round-trip.

        Args:
            wav: Input waveform, shape [T] or [1, T]. Any sample rate.
            sr:  Sample rate of wav.

        Returns:
            ReconstructionResult dict.

        The implementation must handle resampling internally if sr != self.sample_rate.
        """
        ...

    # ------------------------------------------------------------------
    # Optional hooks — default to NotImplementedError.
    # Implement if the codec exposes encode/decode separately.
    # ------------------------------------------------------------------

    def encode(self, wav: Tensor, sr: int) -> Tensor:
        """Encode wav → token tensor [Q, T_frames].  Optional."""
        raise NotImplementedError

    def decode(self, tokens: Tensor) -> Tensor:
        """Decode token tensor [Q, T_frames] → waveform.  Optional."""
        raise NotImplementedError
