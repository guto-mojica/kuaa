"""
cinemateca.device
~~~~~~~~~~~~~~~~~
Detecção e seleção de device PyTorch (CPU / CUDA / MPS).

Centralizado aqui para que todos os módulos usem a mesma lógica,
sem repetir o bloco if/elif em cada arquivo.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_device(preference: str = "auto") -> torch.device:
    """
    Retorna o melhor device disponível.

    Args:
        preference: "auto" | "cpu" | "cuda" | "mps"
                    "auto" tenta MPS → CUDA → CPU nessa ordem.

    Returns:
        torch.device pronto para uso.
    """
    import torch

    pref = preference.lower()

    if pref == "cpu":
        device = torch.device("cpu")
        logger.info("Device: CPU (forçado por config)")
        return device

    if pref == "mps":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Device: Apple Silicon MPS (forçado por config)")
            return device
        logger.warning("MPS solicitado mas não disponível — caindo para CPU")
        return torch.device("cpu")

    if pref == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Device: NVIDIA CUDA (forçado por config)")
            return device
        logger.warning("CUDA solicitado mas não disponível — caindo para CPU")
        return torch.device("cpu")

    # "auto"
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Device: Apple Silicon MPS (auto-detectado)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Device: NVIDIA CUDA (auto-detectado)")
    else:
        device = torch.device("cpu")
        logger.info("Device: CPU (fallback — GPU não disponível)")

    return device


def device_from_config(cfg) -> torch.device:
    """
    Atalho para ler a preferência de device direto da config.

    Args:
        cfg: _Namespace da config (resultado de load_config()).
    """
    hw = cfg.hardware
    if getattr(hw, "force_cpu", False):
        return get_device("cpu")
    preference = getattr(hw, "device", "auto")
    return get_device(preference)
