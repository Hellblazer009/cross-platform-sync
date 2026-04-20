"""
blend_gate.py
─────────────
Mahalanobis-gated soft blender for PREP.

Given:
  x_phys  — physics predictor output (position + quaternion)
  delta_x — TCN residual correction [Δp(3), Δφ(3)]
  sigma2  — TCN per-dimension uncertainty from log-variance head
  epsilon — 6D prediction error [pos_err(3), rot_err(3)] vs last physics estimate

The blend gate computes a scalar α ∈ [0,1] and produces:

  x_out = x_phys ⊕ α·delta_x

where ⊕ for position is addition and for orientation is:
    q_out = q_phys ⊗ qexp(α · Δφ)

α is determined by:
  1. Mahalanobis distance d = sqrt(ε^T Σ^{-1} ε)
     where Σ is a running estimate of the prediction error covariance
  2. TCN uncertainty: high σ² → reduces α even if d is large
  3. Soft sigmoid blend: α = sigmoid((d - θ_mid) / τ) · w_tcn
     where w_tcn = 1 / (1 + mean(σ²) / σ²_ref)

Parameters (tuneable for experiments):
  THETA_LOW   d below this → physics-only (α ≈ 0)
  THETA_HIGH  d above this → full residual (α ≈ 1)
  SIGMA2_REF  reference TCN uncertainty; lower σ² → higher TCN weight
  COV_WINDOW  frames kept for running covariance estimate
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from .physics_predictor import qmul, qexp, qnorm


# ── Tunable gate parameters ───────────────────────────────────────────────

THETA_LOW   = 0.05   # Mahalanobis below this: physics dominates
THETA_HIGH  = 2.0    # Mahalanobis above this: full residual correction
SIGMA2_REF  = 1e-4   # TCN uncertainty reference scale (metres² / rad²)
COV_WINDOW  = 50     # frames for running covariance estimate


# ── Running covariance estimator ─────────────────────────────────────────

class RunningCovariance:
    """
    Online diagonal covariance estimator for the 6D prediction error.
    Uses Welford's algorithm for numerical stability.
    Kept diagonal (independent position and rotation axes) for efficiency.
    """
    def __init__(self, dim: int = 6, window: int = COV_WINDOW):
        self._dim    = dim
        self._window = window
        self._buf    = deque(maxlen=window)
        self._cov_diag = np.ones(dim) * 0.01   # prior: small variance

    def update(self, error: np.ndarray) -> None:
        """Add a new 6D error sample."""
        self._buf.append(error.copy())
        if len(self._buf) >= 5:
            arr = np.stack(list(self._buf))           # (N, 6)
            self._cov_diag = np.var(arr, axis=0) + 1e-8   # avoid zero

    def inv_diag(self) -> np.ndarray:
        """Returns element-wise inverse of the diagonal covariance."""
        return 1.0 / self._cov_diag

    def mahalanobis(self, error: np.ndarray) -> float:
        """
        Mahalanobis distance for a diagonal covariance:
            d = sqrt(ε^T diag(Σ)^{-1} ε)
        """
        return float(np.sqrt(np.dot(error ** 2, self.inv_diag())))


# ── Per-client gate state ─────────────────────────────────────────────────

@dataclass
class BlendGateState:
    """Holds the running covariance and diagnostics for one client."""
    cov: RunningCovariance = field(default_factory=RunningCovariance)

    # Diagnostics (written each frame, readable for telemetry)
    last_mahalanobis: float = 0.0
    last_alpha:       float = 0.0
    last_tcn_weight:  float = 0.0


# ── Main gate ────────────────────────────────────────────────────────────

class BlendGate:
    """
    Stateless gate — per-client state is held in BlendGateState and passed in.
    This makes it easy to instantiate one BlendGate shared across all clients.
    """

    @staticmethod
    def _sigmoid_alpha(d: float,
                       theta_low:  float = THETA_LOW,
                       theta_high: float = THETA_HIGH) -> float:
        """
        Smooth blend weight as function of Mahalanobis distance d:
          α(d) → 0 as d → theta_low
          α(d) → 1 as d → theta_high
        Uses a sigmoid centred at the midpoint.
        """
        theta_mid = (theta_low + theta_high) / 2.0
        tau       = (theta_high - theta_low) / 6.0   # width parameter
        if tau < 1e-9:
            return 1.0 if d >= theta_mid else 0.0
        raw = 1.0 / (1.0 + np.exp(-(d - theta_mid) / tau))
        # Clip so that below theta_low → 0, above theta_high → 1
        if d <= theta_low:
            return 0.0
        if d >= theta_high:
            return 1.0
        return float(raw)

    @staticmethod
    def _tcn_weight(sigma2: np.ndarray,
                    sigma2_ref: float = SIGMA2_REF) -> float:
        """
        Penalise high TCN uncertainty.
        w_tcn = sigma2_ref / (sigma2_ref + mean(sigma2))
        → 1.0 when TCN is very confident, → 0 as uncertainty grows.
        """
        mean_var = float(np.mean(sigma2))
        return sigma2_ref / (sigma2_ref + mean_var)

    def compute(self,
                state:    BlendGateState,
                error_6d: np.ndarray,
                delta_x:  np.ndarray,
                sigma2:   np.ndarray,
                p_phys:   np.ndarray,
                q_phys:   np.ndarray) -> tuple[np.ndarray, np.ndarray, float, str]:
        """
        Blend physics prediction with TCN residual correction.

        Parameters
        ──────────
        state     : BlendGateState for this client (mutated in-place)
        error_6d  : 6D prediction error [pos_err, rot_err] from physics_predictor
        delta_x   : TCN output [Δp(3), Δφ(3)]
        sigma2    : TCN per-dim variance (6,)
        p_phys    : physics position prediction (3,)
        q_phys    : physics quaternion prediction [w,x,y,z] (4,)

        Returns
        ───────
        p_out     : blended position (3,)
        q_out     : blended quaternion [w,x,y,z] (4,)
        alpha     : blend weight used (0 = physics, 1 = full residual)
        source    : "physics" | "blended" | "residual"
        """
        # Update running covariance with latest error
        state.cov.update(error_6d)

        # Mahalanobis distance
        d = state.cov.mahalanobis(error_6d)

        # Soft blend weight from Mahalanobis distance
        alpha_geo = self._sigmoid_alpha(d)

        # Penalise by TCN uncertainty
        w_tcn = self._tcn_weight(sigma2)
        alpha = alpha_geo * w_tcn

        # ── Apply correction ──────────────────────────────────────────
        delta_p = delta_x[:3]
        delta_phi = delta_x[3:]

        # Position: linear blend
        p_out = p_phys + alpha * delta_p

        # Orientation: apply fractional rotation in Lie algebra
        # q_out = q_phys ⊗ qexp(α · Δφ)
        q_corr = qexp(alpha * delta_phi)
        q_out  = qnorm(qmul(q_phys, q_corr))

        # Diagnostic labels
        if alpha < 0.05:
            source = "physics"
        elif alpha > 0.95:
            source = "residual"
        else:
            source = "blended"

        # Store diagnostics
        state.last_mahalanobis = d
        state.last_alpha       = alpha
        state.last_tcn_weight  = w_tcn

        return p_out, q_out, alpha, source


# ── Module-level singleton ────────────────────────────────────────────────
# Edge server instantiates one BlendGate and one BlendGateState per client.

gate = BlendGate()
