"""
clock_sync.py
─────────────
Per-client affine clock correction model.

Each XR client has its own local clock that drifts relative to the edge
server. We model the relationship as:

    C_client(t) = alpha_i * C_edge(t) + beta_i

where alpha_i is the skew (rate difference) and beta_i is the static offset.

Protocol (runs every SYNC_INTERVAL seconds, independent of pose stream):
  1. Client sends POST /sync_clock  { "user_id": ..., "t_c1": <client time> }
  2. Edge server records t_e (its own clock) and responds { "t_e": <edge time> }
  3. Client records t_c2 and calls POST /sync_ack { "user_id": ...,
                                                     "t_c1": ...,
                                                     "t_e": ...,
                                                     "t_c2": ... }
  4. Edge server computes offset estimate:
        beta_hat = t_e - (t_c1 + t_c2) / 2
  5. Over multiple rounds, skew alpha_i is estimated via linear regression
     over (beta_hat, t_e) samples.

All incoming pose timestamps are then corrected before entering PREP:
    t_corrected = (t_client - beta_hat) / alpha_hat
"""

import time
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


# How many sync rounds to keep for skew regression
_SKEW_WINDOW = 20


@dataclass
class ClockSyncModel:
    """
    Maintains per-client clock correction state.
    Thread-safety: callers should hold an asyncio lock if shared across tasks.
    """
    user_id: str

    # Affine parameters (edge clock domain)
    offset: float = 0.0        # beta_hat  — additive offset in seconds
    skew: float = 1.0          # alpha_hat — multiplicative rate correction

    # Rolling window for regression: list of (t_e, raw_beta_estimate)
    _samples: deque = field(default_factory=lambda: deque(maxlen=_SKEW_WINDOW))

    # Pending t_c1 waiting for ack
    _pending_t_c1: Optional[float] = None
    _pending_t_e: Optional[float] = None

    # Diagnostics
    n_syncs: int = 0
    last_sync_rtt: float = 0.0     # round-trip time of last sync exchange

    def record_sync_request(self, t_c1: float) -> float:
        """
        Called when the client's sync ping arrives.
        Records t_c1 and the server reception time t_e.
        Returns t_e so it can be sent back to the client.
        """
        t_e = time.monotonic()
        self._pending_t_c1 = t_c1
        self._pending_t_e = t_e
        return t_e

    def record_sync_ack(self, t_c1: float, t_e: float, t_c2: float) -> None:
        """
        Called when the client sends back its t_c2 (time it received t_e).
        Estimates beta and accumulates for skew regression.
        """
        # Symmetric network delay assumption → offset estimate
        rtt = t_c2 - t_c1
        beta_estimate = t_e - (t_c1 + t_c2) / 2.0

        self.offset = beta_estimate        # simple: use latest estimate
        self.last_sync_rtt = rtt
        self.n_syncs += 1

        # Store sample for skew regression: (edge_time, raw_offset)
        self._samples.append((t_e, beta_estimate))
        self._update_skew()

    def _update_skew(self) -> None:
        """
        Linear regression over accumulated (t_e, beta) samples to estimate
        clock skew alpha. Requires at least 5 samples for meaningful fit.
        Skew close to 1.0 means clocks run at the same rate.
        """
        if len(self._samples) < 5:
            return

        t_vals = np.array([s[0] for s in self._samples])
        b_vals = np.array([s[1] for s in self._samples])

        # beta(t) = (1 - alpha) * t + const  →  linear fit
        coeffs = np.polyfit(t_vals, b_vals, deg=1)
        slope = coeffs[0]          # d(beta)/dt = 1 - alpha

        alpha_estimate = 1.0 - slope
        # Clamp to a sane range to avoid instability on few samples
        self.skew = float(np.clip(alpha_estimate, 0.95, 1.05))

    def correct(self, t_client: float) -> float:
        """
        Map a client timestamp into the edge server's clock domain.

        t_corrected = (t_client - beta) / alpha
        """
        return (t_client - self.offset) / self.skew

    def __repr__(self) -> str:
        return (f"ClockSyncModel(user={self.user_id}, "
                f"offset={self.offset*1000:.2f}ms, "
                f"skew={self.skew:.6f}, "
                f"n_syncs={self.n_syncs})")


class ClockSyncRegistry:
    """
    Holds ClockSyncModel instances for all connected clients.
    """
    def __init__(self):
        self._models: dict[str, ClockSyncModel] = {}

    def get_or_create(self, user_id: str) -> ClockSyncModel:
        if user_id not in self._models:
            self._models[user_id] = ClockSyncModel(user_id=user_id)
        return self._models[user_id]

    def correct_timestamp(self, user_id: str, t_client: float) -> float:
        """Convenience: correct a client timestamp, creating model if needed."""
        return self.get_or_create(user_id).correct(t_client)

    def all_diagnostics(self) -> dict:
        return {uid: repr(m) for uid, m in self._models.items()}
