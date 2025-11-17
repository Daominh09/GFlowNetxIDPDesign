"""
GFN-xIDP: GFlowNet-based intrinsic disorder protein design utilities.

This module exposes the main factory functions and classes commonly used
throughout the package.
"""

from .args import get_default_args
from .generator import get_generator
from .oracle import ProtVec, get_oracle
from .dataset import IDPDataset, get_dataset
from .tokenizer import get_tokenizer
from .proxy import get_proxy
from . import utils

__all__ = [
    "get_default_args",
    "get_generator",
    "get_oracle",
    "ProtVec",
    "IDPDataset",
    "get_dataset",
    "get_tokenizer",
    "get_proxy",
    "utils",
]

__version__ = "0.1.0"

