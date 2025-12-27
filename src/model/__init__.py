"""
Neural Network モジュール

AlphaZero方式のDual-Head ResNetを提供
"""

from .net import (
    OthelloResNet,
    ConvBlock,
    ResBlock,
    PolicyHead,
    ValueHead,
    create_model,
)

__all__ = [
    "OthelloResNet",
    "ConvBlock",
    "ResBlock",
    "PolicyHead",
    "ValueHead",
    "create_model",
]
