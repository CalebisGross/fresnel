# Fresnel v2 Distillation Module
#
# Tools for distilling TRELLIS's 3D knowledge into fast direct predictors.

from .trellis_dataset import (
    TrellisDistillationDataset,
    GaussianTargetDataset,
    create_dataloaders,
    collate_variable_gaussians,
)

__all__ = [
    'TrellisDistillationDataset',
    'GaussianTargetDataset',
    'create_dataloaders',
    'collate_variable_gaussians',
]
