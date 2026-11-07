"""Cartan-DEC-MiAE: discrete differential geometry for protein self-supervision."""

from .data import SyntheticBackboneDataset, TEDBenchDataset, collate_backbones, make_dataset
from .geometry import (
    backbone_features,
    connection_forms,
    connection_heat_kernel,
    connection_laplacian,
    contact_adjacency,
    frames_from_backbone,
    geometry_targets,
    integrate_sequence_paths,
    nearest_visible_anchor,
    so3_exp,
    so3_log,
    triangle_curvature,
)
from .losses import cartan_loss
from .masking import geometry_masking, make_mask, random_masking, span_masking
from .model import CartanDEC, CartanDECConfig, build_model

__all__ = [
    "CartanDEC",
    "CartanDECConfig",
    "build_model",
    "SyntheticBackboneDataset",
    "TEDBenchDataset",
    "make_dataset",
    "collate_backbones",
    "frames_from_backbone",
    "contact_adjacency",
    "connection_forms",
    "connection_laplacian",
    "connection_heat_kernel",
    "triangle_curvature",
    "backbone_features",
    "geometry_targets",
    "nearest_visible_anchor",
    "integrate_sequence_paths",
    "so3_exp",
    "so3_log",
    "cartan_loss",
    "make_mask",
    "random_masking",
    "span_masking",
    "geometry_masking",
]
