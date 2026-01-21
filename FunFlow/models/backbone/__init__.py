"""Backbone module for feature extraction networks."""

from FunFlow.models.backbone.resnet import ResNet, ResNetBackbone
from FunFlow.models.backbone.tinyvit import TinyVit, TinyVitBackbone
from FunFlow.models.backbone.tcn import TCNBackbone
from FunFlow.models.backbone.resnet1d import ResNetBackbone1D

TCN = TCNBackbone
ResNet1D = ResNetBackbone1D

__all__ = [
    "ResNet",
    "ResNetBackbone",
    "TinyVit",
    "TinyVitBackbone",
    "TCN",
    "TCNBackbone",
    "ResNet1D",
    "ResNetBackbone1D",
]
