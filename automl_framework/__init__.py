"""
Antigravity AutoML Regression Framework package.
Exposes core classes for standard library usage.
"""

from automl_framework.dataloader import DataLoader
from automl_framework.model import ModelPool, StandardBenchmarkExecutor
from automl_framework.util import Visualizer

__all__ = ["DataLoader", "ModelPool", "StandardBenchmarkExecutor", "Visualizer"]
