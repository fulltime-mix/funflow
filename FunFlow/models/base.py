from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union

import torch
import torch.nn as nn
import torch.quantization as quant

from FunFlow.registry import PREPROCESSINGS, BACKBONES, NECKS, HEADS, LOSSES
from FunFlow.logger import get_logger

logger = get_logger("FunFlow")


class BaseModel(nn.Module, ABC):
    """
    Base model abstract class with standardized architecture:
    Preprocessing -> Backbone -> Neck -> Head -> Loss
    """

    def __init__(
        self,
        preprocessing: Dict[str, Any] = None,
        backbone: Dict[str, Any] = None,
        neck: Dict[str, Any] = None,
        head: Dict[str, Any] = None,
        loss: Dict[str, Any] = None,
        quantization: bool = False,
        **kwargs,
    ):
        """Initialize model components.

        Args:
            preprocessing: Preprocessing module config
            backbone: Backbone config
            neck: Neck config for feature fusion
            head: Classification head config
            loss: Loss function config
            quantization: Whether to enable quantization
        """
        super().__init__()
        self.quantization = quantization
        self._is_init = False

        self.quant = quant.QuantStub() if quantization else nn.Identity()
        self.dequant = quant.DeQuantStub() if quantization else nn.Identity()

        self.preprocessing = self._build_module(preprocessing, "preprocessing")
        self.backbone = self._build_module(backbone, "backbone")
        self.neck = self._build_module(neck, "neck")
        self.head = self._build_module(head, "head")
        self.loss_fn = self._build_module(loss, "loss")

    def _build_module(self, cfg: Dict[str, Any], module_type: str) -> nn.Module:
        """Build module via Registry.

        Args:
            cfg: Module config dict, must contain 'type' key
            module_type: Module type

        Returns:
            Built module or nn.Identity()
        """
        if not cfg or not cfg.get("type"):
            return nn.Identity()

        cfg = cfg.copy()
        module_name = cfg.get("type")

        registry_map = {
            "preprocessing": PREPROCESSINGS,
            "backbone": BACKBONES,
            "neck": NECKS,
            "head": HEADS,
            "loss": LOSSES,
        }

        registry = registry_map.get(module_type)
        if registry is None:
            raise ValueError(f"Unknown module type: {module_type}")

        if module_name not in registry:
            if module_type == "loss" and hasattr(nn, module_name):
                cfg_copy = cfg.copy()
                cfg_copy.pop("type")
                return getattr(nn, module_name)(**cfg_copy)

            logger.warning(
                f"'{module_name}' not registered in {module_type} registry. "
                f"Available: {registry.keys()}. Using Identity."
            )
            return nn.Identity()

        try:
            return registry.build(cfg)
        except Exception as e:
            logger.error(f"Failed to build {module_type} '{module_name}': {e}")
            raise

    @abstractmethod
    def _parse_inputs(
        self, inputs: Union[Dict[str, torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """Parse and preprocess input data.

        Args:
            inputs: Raw input

        Returns:
            Preprocessed tensor
        """
        pass

    def forward(
        self, inputs: Union[Dict[str, torch.Tensor], torch.Tensor], **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            inputs: Input data

        Returns:
            Dict containing logits, probs, preds, features
        """
        x = self._parse_inputs(inputs)
        x = self.quant(x)
        x = self.preprocessing(x)
        features = self.backbone(x)
        features = self.neck(features)
        logits = self.head(features)
        logits = self.dequant(logits)

        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        return {
            "features": features,
            "logits": logits,
            "probs": probs,
            "preds": preds,
        }

    def forward_train(
        self, inputs: Dict[str, torch.Tensor], **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training.

        Args:
            inputs: Input data dict

        Returns:
            Output dict containing loss
        """
        outputs = self.forward(inputs, **kwargs)

        if self.loss_fn is not None and not isinstance(self.loss_fn, nn.Identity):
            outputs["loss"] = self.loss_fn(outputs, inputs)

        return outputs

    def init_weights(self, init_cfg: Optional[Dict[str, Any]] = None):
        """Initialize model weights.

        Args:
            init_cfg: Initialization config dict
        """
        default_cfg = {
            "conv_init": "kaiming_normal",
            "conv_mode": "fan_out",
            "conv_nonlinearity": "relu",
            "linear_init": "kaiming_normal",
            "linear_mode": "fan_in",
            "linear_nonlinearity": "relu",
            "bn_weight": 1.0,
            "bn_bias": 0.0,
        }

        if init_cfg is not None:
            default_cfg.update(init_cfg)
        cfg = default_cfg

        for name, m in self.named_modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                if cfg["conv_init"] == "kaiming_normal":
                    nn.init.kaiming_normal_(
                        m.weight,
                        mode=cfg["conv_mode"],
                        nonlinearity=cfg["conv_nonlinearity"],
                    )
                elif cfg["conv_init"] == "kaiming_uniform":
                    nn.init.kaiming_uniform_(
                        m.weight,
                        mode=cfg["conv_mode"],
                        nonlinearity=cfg["conv_nonlinearity"],
                    )
                elif cfg["conv_init"] == "xavier_normal":
                    nn.init.xavier_normal_(m.weight)
                elif cfg["conv_init"] == "xavier_uniform":
                    nn.init.xavier_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                if cfg["linear_init"] == "kaiming_normal":
                    nn.init.kaiming_normal_(
                        m.weight,
                        mode=cfg["linear_mode"],
                        nonlinearity=cfg["linear_nonlinearity"],
                    )
                elif cfg["linear_init"] == "kaiming_uniform":
                    nn.init.kaiming_uniform_(
                        m.weight,
                        mode=cfg["linear_mode"],
                        nonlinearity=cfg["linear_nonlinearity"],
                    )
                elif cfg["linear_init"] == "xavier_normal":
                    nn.init.xavier_normal_(m.weight)
                elif cfg["linear_init"] == "xavier_uniform":
                    nn.init.xavier_uniform_(m.weight)
                elif cfg["linear_init"] == "normal":
                    std = cfg.get("linear_std", 0.02)
                    nn.init.normal_(m.weight, 0, std)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(
                m,
                (
                    nn.BatchNorm1d,
                    nn.BatchNorm2d,
                    nn.BatchNorm3d,
                    nn.GroupNorm,
                    nn.LayerNorm,
                    nn.InstanceNorm1d,
                    nn.InstanceNorm2d,
                    nn.InstanceNorm3d,
                ),
            ):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.constant_(m.weight, cfg["bn_weight"])
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, cfg["bn_bias"])

            elif isinstance(m, (nn.LSTM, nn.GRU, nn.RNN)):
                for param_name, param in m.named_parameters():
                    if "weight_ih" in param_name:
                        nn.init.xavier_uniform_(param)
                    elif "weight_hh" in param_name:
                        nn.init.orthogonal_(param)
                    elif "bias" in param_name:
                        nn.init.constant_(param, 0)
                        if isinstance(m, nn.LSTM):
                            n = param.size(0)
                            param.data[n // 4 : n // 2].fill_(1.0)

            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.01)

        self._is_init = True
        logger.info("Model weights initialized")

    def get_num_params(self, trainable_only: bool = True) -> int:
        """Get number of parameters.

        Args:
            trainable_only: Whether to count only trainable parameters

        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    @property
    def device(self) -> torch.device:
        """Get model device.

        Returns:
            Device
        """
        return next(self.parameters()).device


__all__ = ["BaseModel"]
