"""
physics_predictor.py
────────────────────
Second-order kinematic predictor operating on SE(3).

Position model  — constant acceleration:
    v_t = (p_t - p_{t-1}) / Δt
    a_t = (v_t - v_{t-1}) / Δt
    p̂_{t+h} = p_t + v_t·h + 0.5·a_t·h²

Orientation model — angular velocity integration on SO(3):
    ω_t  = 2 · qlog(q_{t-1}^{-1} ⊗ q_t) / Δt        [rad/s, body frame]
    q̂_{t+h} = q_t ⊗ qexp(0.5 · ω_t · h)

Quaternion convention: [w, x, y, z] throughout internally.
Unity sends [x, y, z, w] — callers must reorder before passing in.

The predictor maintains separate state per body part (head, left hand,
right hand) so a single BodyPartPredictor instance handles one transform.
A PosePredictor aggregates all three.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── Quaternion primitives ──────────────────────────────────────────────────

def qnorm(q: np.ndarray) -> np.ndarray:
    """Normalise quaternion [w, x, y, z] to unit length."""
    n = np.linalg.norm(q)
    return q / n if n > 1e-9 else np.array([1.0, 0.0, 0.0, 0.0])


def qmul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 ⊗ q2, both [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def qconj(q: np.ndarray) -> np.ndarray:
    """Conjugate (inverse for unit quaternion) [w, x, y, z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def qlog(q: np.ndarray) -> np.ndarray:
    """
    Logarithmic map: unit quaternion → so(3) vector (axis-angle).
    Returns a 3-vector v such that qexp(v) ≈ q.
    """
    q = qnorm(q)
    w = float(np.clip(q[0], -1.0, 1.0))
    vec = q[1:]
    vec_norm = np.linalg.norm(vec)
    if vec_norm < 1e-9:
        # Near identity — first-order approximation
        return vec * 2.0
    theta = np.arccos(w)           # θ ∈ [0, π]
    return (theta / vec_norm) * vec


def qexp(v: np.ndarray) -> np.ndarray:
    """
    Exponential map: so(3) vector → unit quaternion.
    v is a 3-vector (axis-angle representation).
    """
    theta = np.linalg.norm(v)
    if theta < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = v / theta
    return np.array([np.cos(theta), *(np.sin(theta) * axis)])


def qslerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation between q0 and q1.
    t=0 → q0, t=1 → q1. Handles antipodal case.
    """
    q0, q1 = qnorm(q0), qnorm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:          # take the shorter arc
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:       # quaternions nearly identical → linear interpolation
        return qnorm(q0 + t * (q1 - q0))
    theta_full = np.arccos(dot)
    theta = theta_full * t
    q_perp = qnorm(q1 - dot * q0)
    return np.cos(theta) * q0 + np.sin(theta) * q_perp


# ── Per body-part predictor ────────────────────────────────────────────────

@dataclass
class BodyPartPredictor:
    """
    Tracks kinematic state for a single transform (head / hand).
    Call update() with each new measured pose; call predict() to
    extrapolate forward by h seconds.
    """
    # Internal state — all in edge-server clock domain
    _p:     Optional[np.ndarray] = None   # last position [3]
    _q:     Optional[np.ndarray] = None   # last quaternion [w,x,y,z]
    _v:     np.ndarray = field(default_factory=lambda: np.zeros(3))   # linear velocity
    _a:     np.ndarray = field(default_factory=lambda: np.zeros(3))   # linear acceleration
    _omega: np.ndarray = field(default_factory=lambda: np.zeros(3))   # angular velocity [rad/s]
    _t:     Optional[float] = None        # timestamp of last update

    # Previous step values for second-order finite differences
    _v_prev: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def update(self, p: np.ndarray, q: np.ndarray, t: float) -> None:
        """
        Ingest a new measured pose (p: position [3], q: quaternion [w,x,y,z],
        t: corrected timestamp in edge clock domain).
        Updates internal velocity, acceleration and angular velocity estimates.
        """
        if self._p is None:
            # First sample — initialise, no derivatives yet
            self._p = p.copy()
            self._q = qnorm(q)
            self._t = t
            return

        dt = t - self._t
        if dt < 1e-6:
            return                        # duplicate / out-of-order frame

        # ── Linear kinematics ──────────────────────────────────────────
        v_new = (p - self._p) / dt
        self._a = (v_new - self._v) / dt
        self._v_prev = self._v.copy()
        self._v = v_new

        # ── Angular velocity via SO(3) finite difference ───────────────
        # ω_t = 2 · log(q_{t-1}^{-1} ⊗ q_t) / Δt
        q_prev_inv = qconj(self._q)
        dq = qmul(q_prev_inv, qnorm(q))
        self._omega = 2.0 * qlog(dq) / dt

        # Update stored state
        self._p = p.copy()
        self._q = qnorm(q)
        self._t = t

    def predict(self, h: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Extrapolate current state forward by h seconds.
        Returns (p_pred [3], q_pred [w,x,y,z]).
        Falls back to last known pose if uninitialised.
        """
        if self._p is None:
            return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])

        # Position: constant-acceleration model
        p_pred = self._p + self._v * h + 0.5 * self._a * (h ** 2)

        # Orientation: integrate angular velocity on SO(3)
        # q̂_{t+h} = q_t ⊗ exp(0.5 · ω · h)
        half_angle_vec = 0.5 * self._omega * h
        dq_pred = qexp(half_angle_vec)
        q_pred = qnorm(qmul(self._q, dq_pred))

        return p_pred, q_pred

    def current_pose(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return the last filtered position and quaternion."""
        return self._p, self._q

    def is_initialised(self) -> bool:
        return self._p is not None


# ── Full pose predictor (head + two hands) ────────────────────────────────

class PosePredictor:
    """
    Wraps three BodyPartPredictor instances for head, left hand, right hand.
    Input quaternions must be in [w, x, y, z] order.
    """
    def __init__(self):
        self.head       = BodyPartPredictor()
        self.left_hand  = BodyPartPredictor()
        self.right_hand = BodyPartPredictor()

    def update(self,
               head_p: np.ndarray, head_q: np.ndarray,
               lhand_p: np.ndarray, lhand_q: np.ndarray,
               rhand_p: np.ndarray, rhand_q: np.ndarray,
               t: float) -> None:
        """Update all three body-part predictors with a new measurement."""
        self.head.update(head_p, head_q, t)
        self.left_hand.update(lhand_p, lhand_q, t)
        self.right_hand.update(rhand_p, rhand_q, t)

    def predict(self, h: float = 0.0) -> dict:
        """
        Predict all body parts h seconds into the future.
        h=0 returns the current best estimate (filtered, no extrapolation).
        Returns dict with keys matching InterpretedPoseFrame fields.
        """
        hp, hq   = self.head.predict(h)
        lp, lq   = self.left_hand.predict(h)
        rp, rq   = self.right_hand.predict(h)
        return {
            "head_position":        hp.tolist(),
            "head_rotation":        hq.tolist(),          # [w,x,y,z]
            "left_hand_position":   lp.tolist(),
            "left_hand_rotation":   lq.tolist(),
            "right_hand_position":  rp.tolist(),
            "right_hand_rotation":  rq.tolist(),
        }

    def prediction_error_6d(self,
                             p_meas: np.ndarray,
                             q_meas: np.ndarray,
                             part: str = "head") -> np.ndarray:
        """
        Compute the 6D prediction error vector for one body part:
            ε = [p_meas - p̂_phys,  log(q̂_phys^{-1} ⊗ q_meas)]  ∈ R^6
        Used by the blend gate to compute the Mahalanobis distance.
        """
        predictor = getattr(self, part)
        p_phys, q_phys = predictor.current_pose()

        if p_phys is None:
            return np.zeros(6)

        pos_err = p_meas - p_phys

        q_phys_inv = qconj(q_phys)
        rot_err = qlog(qmul(q_phys_inv, qnorm(q_meas)))   # ∈ R^3, so(3)

        return np.concatenate([pos_err, rot_err])
