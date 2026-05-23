"""
Core module for AutoML Framework.
Exposes key classes and functions for data loading, model definition, and visualization.
"""

from .data_loader import DataLoader
from .models import ModelPool
from .visualizer import Visualizer

__all__ = ["DataLoader", "ModelPool", "Visualizer"]
