"""Acoustic encoder for wav2vec2/HuBERT features.

This module provides the AcousticEncoder class that processes pre-extracted
acoustic features (from wav2vec2 or HuBERT) and produces hidden states
compatible with the existing phoneme decoder.

The encoder follows the VisualEncoder pattern (not PhonemeEncoder) because:
- It has no phoneme embedding (input is continuous features)
- It uses reshape_magic() for hidden state conversion
- bind() won't share embeddings (correct behavior for acoustic input)
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class AcousticEncoder(nn.Module):
    """Acoustic encoder that processes wav2vec2/HuBERT features.

    Follows the VisualEncoder pattern for Unimodel integration.
    Uses pack_padded_sequence for length-safe encoding (padding doesn't
    affect the hidden state representation).

    Args:
        input_dim: Feature dimension from acoustic model (768 for base, 1024 for large)
        hidden_size: Output hidden size (must match decoder)
        num_layers: Number of RNN layers
        dropout: Dropout rate
        recur_type: 'LSTM' or 'RNN'

    Attributes:
        hidden_shape: torch.Size for reshape_magic compatibility
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.0,
        recur_type: str = "LSTM",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.droprate = dropout
        self.recur_type = recur_type.upper()

        # Input projection (if input_dim != hidden_size)
        if input_dim != hidden_size:
            self.input_proj = nn.Linear(input_dim, hidden_size)
        else:
            self.input_proj = nn.Identity()

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Recurrent layer
        if self.recur_type == "LSTM":
            self.recurrent = nn.LSTM(
                hidden_size,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
            # For LSTM: need space for both h and c
            total_units = 2 * num_layers * hidden_size
        elif self.recur_type == "RNN":
            self.recurrent = nn.RNN(
                hidden_size,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
            # For RNN: only h
            total_units = num_layers * hidden_size
        else:
            raise ValueError(f"Unsupported recur_type: {recur_type}")

        # Hidden shape for reshape_magic (following VisualEncoder pattern)
        # This shape allows can_reshape_magic() to verify compatibility with decoder
        self.hidden_shape = torch.Size((-1, total_units))

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> tuple[None, torch.Tensor]:
        """Forward pass through the acoustic encoder.

        Args:
            features: (batch, time, input_dim) acoustic features (padded)
            lengths: (batch,) actual sequence lengths for each item.
                     If None, assumes all sequences have the same length.

        Returns:
            tuple of (None, hidden_flat) where:
                - None: placeholder for auxiliary output (like VisualEncoder's object_pred)
                - hidden_flat: flattened hidden state (batch, total_hidden_units)
                              ready for reshape_magic()
        """
        batch_size = features.size(0)
        max_time = features.size(1)

        # Project input features to hidden size
        x = self.input_proj(features)  # (batch, time, hidden_size)
        x = self.dropout(x)

        # Use pack_padded_sequence for length-safe encoding
        if lengths is not None:
            # Ensure lengths are on CPU and int64 for pack_padded_sequence
            lengths_cpu = lengths.cpu().to(torch.int64)
            # Clamp lengths to valid range
            lengths_cpu = lengths_cpu.clamp(min=1, max=max_time)
            packed = pack_padded_sequence(
                x, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            _, hidden = self.recurrent(packed)
        else:
            # No lengths provided - process full sequences
            _, hidden = self.recurrent(x)

        # hidden is (h, c) for LSTM or just h for RNN
        # h shape: (num_layers, batch, hidden_size)
        if self.recur_type == "LSTM":
            h, c = hidden
            # Flatten h and c, then concatenate
            # h: (num_layers, batch, hidden_size) -> (batch, num_layers * hidden_size)
            h_flat = h.permute(1, 0, 2).reshape(batch_size, -1)
            c_flat = c.permute(1, 0, 2).reshape(batch_size, -1)
            hidden_flat = torch.cat([h_flat, c_flat], dim=-1)
        else:
            h = hidden
            hidden_flat = h.permute(1, 0, 2).reshape(batch_size, -1)

        # Return (None, hidden_flat) to match VisualEncoder interface
        # None is placeholder for object_pred (not applicable for acoustic)
        return None, hidden_flat

    def to_unroll(self):
        """No-op for API compatibility with PhonemeEncoder."""
        pass

    def to_chain(self):
        """No-op for API compatibility with PhonemeEncoder."""
        pass
