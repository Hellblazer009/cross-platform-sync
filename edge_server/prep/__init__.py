"""
PREP — Physics-Residual Edge Predictor
=======================================
Package exposing the four algorithm components:

  ClockSyncRegistry    — per-client affine clock correction
  PosePredictor        — SE(3) second-order kinematic predictor
  TCNResidualPredictor — TCN residual correction with uncertainty
  BlendGate / BlendGateState — Mahalanobis-gated soft blender
"""

from .clock_sync      import ClockSyncRegistry, ClockSyncModel
from .physics_predictor import PosePredictor, qlog, qexp, qmul, qnorm
from .tcn_residual    import TCNResidualPredictor
from .blend_gate      import BlendGate, BlendGateState, gate

__all__ = [
    "ClockSyncRegistry", "ClockSyncModel",
    "PosePredictor", "qlog", "qexp", "qmul", "qnorm",
    "TCNResidualPredictor",
    "BlendGate", "BlendGateState", "gate",
]
