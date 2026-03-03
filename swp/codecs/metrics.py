"""Audio reconstruction metrics for codec evaluation.

All functions operate on in-memory tensors (no file I/O).
All handle length mismatches by truncating to the shorter signal.
Dependencies: torch, torchaudio (standard with PyTorch).
"""

from __future__ import annotations

import math

import torch
import torchaudio.transforms as _T
from torch import Tensor

from swp.codecs.base import AudioCodec, ReconstructionResult


# ---------------------------------------------------------------------------
# SI-SDR
# ---------------------------------------------------------------------------

def si_sdr(ref: Tensor, est: Tensor) -> float:
    """Scale-Invariant Signal-to-Distortion Ratio (dB).  Higher is better.

    Args:
        ref: Reference waveform, shape [..., T_ref].
        est: Estimated waveform, shape [..., T_est].

    Returns:
        SI-SDR in dB, or:
          - float("nan")  if either signal is silent (max abs < 1e-8 after mean removal)
          - float("inf")  if the noise component is negligible (near-perfect reconstruction)

    Length mismatch is handled by truncating both to min(T_ref, T_est).
    """
    # Flatten to 1-D for the computation
    ref = ref.flatten()
    est = est.flatten()

    # Truncate to shorter length
    T = min(ref.shape[0], est.shape[0])
    ref = ref[:T]
    est = est[:T]

    # Remove DC offset
    ref = ref - ref.mean()
    est = est - est.mean()

    # Guard against silent signals
    if ref.abs().max() < 1e-8 or est.abs().max() < 1e-8:
        return float("nan")

    # Scale-invariant projection
    dot = torch.dot(est, ref)
    ref_energy = torch.dot(ref, ref)
    s_target = (dot / ref_energy) * ref       # projection of est onto ref
    e_noise = est - s_target

    s_pow = torch.dot(s_target, s_target)
    n_pow = torch.dot(e_noise, e_noise)

    if n_pow < 1e-8:
        return float("inf")

    return 10.0 * torch.log10(s_pow / n_pow).item()


# ---------------------------------------------------------------------------
# Log-mel spectrogram distance
# ---------------------------------------------------------------------------

def mel_distance(
    ref: Tensor,
    est: Tensor,
    sr: int = 24_000,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80,
) -> float:
    """L1 distance between log-mel spectrograms.  Lower means better reconstruction.

    Args:
        ref, est:   Waveforms, shape [T] or [1, T].  Same sample rate expected.
        sr:         Sample rate shared by both signals.
        n_fft, hop_length, n_mels: MelSpectrogram parameters.

    Returns:
        Mean absolute difference of log-mel spectrograms, float >= 0.
        Returns float("nan") if either spectrogram is all-zero.

    Length mismatch is handled by truncating to the shorter signal *before*
    computing the spectrogram, then again on the frame dimension.
    """
    # Ensure 2-D [1, T] for the transform
    ref = ref.squeeze(0).unsqueeze(0) if ref.dim() == 2 else ref.unsqueeze(0)
    est = est.squeeze(0).unsqueeze(0) if est.dim() == 2 else est.unsqueeze(0)

    # Truncate to shorter waveform
    T = min(ref.shape[-1], est.shape[-1])
    ref = ref[..., :T]
    est = est[..., :T]

    mel_transform = _T.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )

    mel_ref = mel_transform(ref)   # [1, n_mels, F]
    mel_est = mel_transform(est)   # [1, n_mels, F]

    # Guard against all-zero spectrograms
    if mel_ref.max() < 1e-10 or mel_est.max() < 1e-10:
        return float("nan")

    log_mel_ref = torch.log(mel_ref.clamp(min=1e-5))
    log_mel_est = torch.log(mel_est.clamp(min=1e-5))

    # Truncate frame dimension (rounding in the STFT may differ by ±1 frame)
    F = min(log_mel_ref.shape[-1], log_mel_est.shape[-1])
    log_mel_ref = log_mel_ref[..., :F]
    log_mel_est = log_mel_est[..., :F]

    return (log_mel_ref - log_mel_est).abs().mean().item()


# ---------------------------------------------------------------------------
# Re-encode consistency
# ---------------------------------------------------------------------------

def reencode_consistency(codec: AudioCodec, result: ReconstructionResult) -> float:
    """Token-level re-encode consistency.

    Encodes the reconstructed audio and compares the resulting token codes
    against the original codes from result["tokens"].

    Token shape convention: [Q, T]
      Q = number of quantizer codebooks
      T = number of time frames
    A batch dimension [B, Q, T] (B=1) is squeezed automatically.

    Length mismatch between T_orig and T_recon (±1 frame is normal due to
    codec padding) is handled by comparing on min(T_orig, T_recon).

    Agreement is computed per quantizer and then averaged across Q codebooks,
    so each codebook contributes equally regardless of depth.

    Args:
        codec:  Any codec implementing AudioCodec.
        result: Output of a previous codec.reconstruct() call.

    Returns:
        Fraction of matching code indices in [0.0, 1.0].
        Returns float("nan") if tokens are unavailable (result["tokens"] is None).
    """
    tokens_orig = result.get("tokens")
    if tokens_orig is None:
        return float("nan")

    # Re-encode the reconstructed waveform
    recon_result = codec.reconstruct(result["recon_wav"], result["recon_sr"])
    tokens_recon = recon_result.get("tokens")
    if tokens_recon is None:
        return float("nan")

    # Normalise to [Q, T] — squeeze a leading batch dim if present
    tokens_orig = _ensure_qt(tokens_orig)
    tokens_recon = _ensure_qt(tokens_recon)

    # Truncate time dimension to the shorter of the two
    T_min = min(tokens_orig.shape[1], tokens_recon.shape[1])
    tokens_orig = tokens_orig[:, :T_min]
    tokens_recon = tokens_recon[:, :T_min]

    # Per-quantizer agreement → mean over codebooks
    per_q: Tensor = (tokens_orig == tokens_recon).float().mean(dim=1)  # [Q]
    return per_q.mean().item()


def _ensure_qt(tokens: Tensor) -> Tensor:
    """Squeeze a batch dim if tokens are [B, Q, T] with B=1, return [Q, T]."""
    if tokens.dim() == 3:
        if tokens.shape[0] != 1:
            raise ValueError(
                f"reencode_consistency expects single-item tokens (B=1), "
                f"got shape {tokens.shape}."
            )
        tokens = tokens.squeeze(0)
    if tokens.dim() != 2:
        raise ValueError(
            f"Token tensor must be 2-D [Q, T] or 3-D [1, Q, T], got shape {tokens.shape}."
        )
    return tokens


# ---------------------------------------------------------------------------
# Convenience bundle
# ---------------------------------------------------------------------------

def compute_all(
    ref_wav: Tensor,
    result: ReconstructionResult,
    codec: AudioCodec,
    ref_sr: int,
) -> dict[str, float]:
    """Compute all three metrics and return as a flat dict.

    ref_wav is resampled to result["recon_sr"] before computing waveform metrics,
    so that SI-SDR and mel_distance compare signals at the same sample rate.

    Args:
        ref_wav:  Original waveform at ref_sr.
        result:   Output of codec.reconstruct(ref_wav, ref_sr).
        codec:    The codec (needed for reencode_consistency).
        ref_sr:   Sample rate of ref_wav.
    """
    import torchaudio

    recon_sr = result["recon_sr"]
    recon_wav = result["recon_wav"]

    # Resample reference to codec SR for waveform-level comparison
    if ref_sr != recon_sr:
        ref_at_codec_sr = torchaudio.functional.resample(
            ref_wav.cpu(), orig_freq=ref_sr, new_freq=recon_sr
        )
    else:
        ref_at_codec_sr = ref_wav.cpu()

    sdr = si_sdr(ref_at_codec_sr, recon_wav)
    mel_d = mel_distance(ref_at_codec_sr, recon_wav, sr=recon_sr)
    reencode = reencode_consistency(codec, result)

    return {
        "si_sdr": sdr,
        "mel_distance": mel_d,
        "reencode_consistency": reencode,
    }


def is_valid(metrics: dict[str, float]) -> tuple[bool, list[str]]:
    """Check that a metrics dict contains no invalid values.

    Returns (ok, list_of_error_messages).
    """
    errors: list[str] = []
    for key, val in metrics.items():
        if math.isnan(val):
            errors.append(f"{key} is NaN")
        elif math.isinf(val):
            errors.append(f"{key} is inf")
    if "reencode_consistency" in metrics:
        rc = metrics["reencode_consistency"]
        if not math.isnan(rc) and not (0.0 <= rc <= 1.0):
            errors.append(f"reencode_consistency={rc:.4f} outside [0, 1]")
    return len(errors) == 0, errors
