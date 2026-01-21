"""Neck module for feature fusion between backbone and head."""

from FunFlow.models.neck.identity import Identity
from FunFlow.models.neck.gap import GlobalAveragePooling
from FunFlow.models.neck.flatten import Flatten

__all__ = [
    "Identity",
    "GlobalAveragePooling",
    "Flatten",
]
