"""Lazy-loading codec registry.

Adding a new codec:
  1. Create swp/codecs/<name>_codec.py implementing AudioCodec.
  2. Add one line to _CODEC_REGISTRY below.
  3. Run codec_reconstruct.py --codec <name> — no other changes needed.
"""

from __future__ import annotations

import importlib

from swp.codecs.base import AudioCodec

_CODEC_REGISTRY: dict[str, tuple[str, str]] = {
    # name -> (module_path, ClassName)
    "encodec": ("swp.codecs.encodec_codec", "EnCodecCodec"),
    # "dac":  ("swp.codecs.dac_codec",     "DACCodec"),   # future
    # "mimi": ("swp.codecs.mimi_codec",    "MimiCodec"),  # future
}


def get_codec(name: str, **kwargs) -> AudioCodec:
    """Instantiate a codec by registry name.

    Heavy dependencies (transformers, dac, etc.) are imported only here,
    not at module-load time.

    Args:
        name:   Registry key, e.g. "encodec".
        **kwargs: Forwarded verbatim to the codec constructor
                  (e.g. bandwidth=6.0, device="cpu").

    Returns:
        Instantiated codec satisfying the AudioCodec protocol.

    Raises:
        KeyError: If name is not registered.
    """
    if name not in _CODEC_REGISTRY:
        available = list(_CODEC_REGISTRY)
        raise KeyError(
            f"Unknown codec '{name}'. Available: {available}. "
            "To add a new codec, see swp/codecs/registry.py."
        )
    module_path, class_name = _CODEC_REGISTRY[name]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**kwargs)


def list_codecs() -> list[str]:
    """Return names of all registered codecs."""
    return list(_CODEC_REGISTRY)
