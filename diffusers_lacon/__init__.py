"""
LaCon (Late-Constraint Diffusion) implementation for diffusers

This package provides a diffusers-compatible implementation of the LaCon method
for controllable image generation using condition aligners.
"""

from .models.condition_aligner import ConditionAligner
from .pipelines.pipeline_lacon import LaConPipeline
from .utils.feature_extractor import UNetFeatureExtractor, SimpleFeatureExtractor

__version__ = "0.1.0"
__author__ = "LaCon Diffusers Implementation"

__all__ = [
    "ConditionAligner",
    "LaConPipeline", 
    "UNetFeatureExtractor",
    "SimpleFeatureExtractor",
]