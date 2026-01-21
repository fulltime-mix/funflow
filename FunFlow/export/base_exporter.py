#!/usr/bin/env python
"""Base exporter abstract class defining unified interface."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BaseExporter(ABC):
    """Base exporter class that all exporters must inherit from.

    Subclasses must implement the export() method.
    """

    SUPPORTED_PARAMS = []

    def __init__(self, device: str = "cpu", **kwargs):
        """Initialize exporter.

        Args:
            device: Device ('cpu', 'cuda')
            **kwargs: Format-specific parameters
        """
        self.device = torch.device(device)
        self.model: Optional[nn.Module] = None
        self.config: Optional[Dict[str, Any]] = None
        self._warn_unused_params(kwargs)

    def load_model(
        self,
        checkpoint_path: str,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
        model_loader: str = "pytorch",
    ) -> nn.Module:
        """Load model from checkpoint.

        Args:
            checkpoint_path: Path to model weights
            config: Model configuration dict
            config_path: Path to config file (if config is None)
            model_loader: Model loader type ('pytorch', 'qat', etc.)

        Returns:
            Loaded model
        """
        from FunFlow.registry import MODEL_LOADERS
        from FunFlow.utils import load_config

        if config is None:
            if config_path is None:
                config_path = Path(checkpoint_path).parent / "config.yaml"
                if not config_path.exists():
                    raise ValueError(f"Config not found at {config_path}")
            config = load_config(config_path)

        self.config = config

        loader_fn = MODEL_LOADERS.get(model_loader)
        if loader_fn is None:
            raise ValueError(
                f"Unknown model loader: {model_loader}. "
                f"Available: {MODEL_LOADERS.keys()}"
            )

        self.model = loader_fn(
            config=config, checkpoint_path=checkpoint_path, device=str(self.device)
        )
        self.model.eval()

        logger.info(f"Model loaded from {checkpoint_path}")
        return self.model

    @abstractmethod
    def export(
        self,
        model: nn.Module,
        dummy_input: Union[torch.Tensor, Dict[str, torch.Tensor]],
        output_path: str,
        verify: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Export model (must be implemented by subclass).

        Args:
            model: Model to export
            dummy_input: Example input for tracing
            output_path: Output file path
            verify: Whether to verify exported model
            **kwargs: Format-specific parameters

        Returns:
            Export result dict with keys: output_path, success, verified,
            file_size_mb, export_time_ms, error
        """
        pass

    def verify(
        self,
        output_path: str,
        model: nn.Module,
        dummy_input: Union[torch.Tensor, Dict[str, torch.Tensor]],
        **kwargs,
    ) -> bool:
        """Verify exported model (optional implementation).

        Args:
            output_path: Path to exported model
            model: Original PyTorch model
            dummy_input: Example input
            **kwargs: Verification parameters

        Returns:
            Whether verification passed
        """
        logger.warning(f"{self.__class__.__name__} does not implement verification")
        return True

    def _get_required_param(self, kwargs: dict, key: str, default=None):
        """Get required parameter from kwargs.

        Args:
            kwargs: Parameter dict
            key: Parameter key
            default: Default value (None means parameter is required)

        Returns:
            Parameter value

        Raises:
            ValueError: Required parameter missing
        """
        if key not in kwargs and default is None:
            raise ValueError(
                f"Required parameter '{key}' not found for {self.__class__.__name__}"
            )
        return kwargs.get(key, default)

    def _warn_unused_params(self, kwargs: dict):
        """Warn about unused parameters."""
        if not self.SUPPORTED_PARAMS or not kwargs:
            return

        unused = set(kwargs.keys()) - set(self.SUPPORTED_PARAMS)
        if unused:
            logger.warning(
                f"{self.__class__.__name__} does not use parameters: {unused}"
            )

    def _to_device(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]]):
        """Move input to target device."""
        if isinstance(x, dict):
            return {k: v.to(self.device) for k, v in x.items()}
        return x.to(self.device)


__all__ = ["BaseExporter"]
