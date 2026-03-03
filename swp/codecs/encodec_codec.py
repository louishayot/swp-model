"""EnCodec codec wrapper using HuggingFace Transformers.

Model: facebook/encodec_24khz
  - Native SR: 24000 Hz
  - Supported bandwidths (kbps): 1.5, 3.0, 6.0, 12.0, 24.0
  - Default: 6.0 kbps (8 codebooks at 75 frames/sec)
"""

from __future__ import annotations

import torch
import torchaudio
from torch import Tensor
from transformers import AutoProcessor, EncodecModel

from swp.codecs.base import ReconstructionResult
from swp.utils.setup import set_device


class EnCodecCodec:
    """Implements AudioCodec for facebook/encodec_24khz."""

    _NATIVE_SR: int = 24_000
    _MODEL_ID: str = "facebook/encodec_24khz"

    def __init__(
        self,
        bandwidth: float = 6.0,
        device: str | torch.device | None = None,
    ) -> None:
        """Load the pretrained EnCodec model.

        Args:
            bandwidth: Compression bandwidth in kbps.
                       Must be one of {1.5, 3.0, 6.0, 12.0, 24.0}.
            device:    Torch device.  None → auto-select via set_device().
        """
        if device is None:
            device = set_device()
        self.device = torch.device(device)
        self._bandwidth = bandwidth

        self._model: EncodecModel = (
            EncodecModel.from_pretrained(self._MODEL_ID).to(self.device).eval()
        )
        self._processor: AutoProcessor = AutoProcessor.from_pretrained(self._MODEL_ID)

    # ------------------------------------------------------------------
    # AudioCodec interface
    # ------------------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        return self._NATIVE_SR

    @property
    def name(self) -> str:
        return "encodec"

    def reconstruct(self, wav: Tensor, sr: int) -> ReconstructionResult:
        """Full encode-decode round-trip.

        Args:
            wav: Input waveform, shape [T] or [1, T], any sample rate.
            sr:  Sample rate of wav.

        Returns:
            ReconstructionResult with all fields on CPU.
        """
        # Normalise to 1-D
        wav_1d = wav.squeeze(0) if wav.dim() == 2 else wav

        # Resample to native SR if needed
        if sr != self._NATIVE_SR:
            wav_1d = torchaudio.functional.resample(
                wav_1d.cpu(), orig_freq=sr, new_freq=self._NATIVE_SR
            )

        # Processor expects a numpy array (CPU)
        wav_np = wav_1d.cpu().numpy()
        inputs = self._processor(
            raw_audio=wav_np,
            sampling_rate=self._NATIVE_SR,
            return_tensors="pt",
        )
        input_values: Tensor = inputs["input_values"].to(self.device)
        padding_mask: Tensor = inputs["padding_mask"].to(self.device)

        with torch.no_grad():
            encoder_output = self._model.encode(
                input_values, padding_mask, bandwidth=self._bandwidth
            )
            audio_codes: Tensor = encoder_output.audio_codes
            audio_scales = encoder_output.audio_scales

            decoded = self._model.decode(audio_codes, audio_scales, padding_mask)
            # audio_values: [B, channels, T_decoded]
            recon_wav: Tensor = decoded.audio_values[0, 0].cpu()

        tokens = self._codes_to_qt(audio_codes)
        n_tokens = int(tokens.numel())

        return ReconstructionResult(
            recon_wav=recon_wav,
            recon_sr=self._NATIVE_SR,
            n_tokens=n_tokens,
            tokens=tokens,
        )

    def encode(self, wav: Tensor, sr: int) -> Tensor:
        """Encode wav → token tensor [Q, T].  Calls reconstruct() internally."""
        result = self.reconstruct(wav, sr)
        assert result["tokens"] is not None
        return result["tokens"]

    def decode(self, tokens: Tensor) -> Tensor:
        """Not fully implemented: EnCodec decode() also requires audio_scales.
        Use reconstruct() for the full round-trip."""
        raise NotImplementedError(
            "EnCodecCodec.decode(tokens) is not standalone — audio_scales are "
            "also required by the model.  Use reconstruct() instead."
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _codes_to_qt(audio_codes: Tensor) -> Tensor:
        """Reshape audio_codes → [Q, T], handling both 3-D and 4-D inputs.

        HuggingFace EncodecModel.encode() may return codes in two layouts:
          - [B, Q, T]          — no explicit chunk dimension
          - [B, nb_chunks, Q, T_per_chunk] — chunked encoding

        We normalise to [Q, T] (single item, Q codebooks, T total frames).
        """
        # Drop batch dimension (B=1 for single-item inference)
        codes = audio_codes.squeeze(0).cpu()  # [Q, T]  or  [nb_chunks, Q, T_per_chunk]

        if codes.dim() == 2:
            # Already [Q, T]
            return codes

        if codes.dim() == 3:
            # [nb_chunks, Q, T_per_chunk] → [Q, nb_chunks * T_per_chunk]
            nb_chunks, Q, T_per_chunk = codes.shape
            return codes.permute(1, 0, 2).reshape(Q, nb_chunks * T_per_chunk)

        raise ValueError(
            f"Unexpected audio_codes shape after batch squeeze: {codes.shape}. "
            "Expected 2-D [Q, T] or 3-D [nb_chunks, Q, T_per_chunk]."
        )
