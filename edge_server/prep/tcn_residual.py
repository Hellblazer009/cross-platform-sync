"""
tcn_residual.py
───────────────
Temporal Convolutional Network (TCN) that learns residual corrections
Δx ∈ R^6 on top of the physics predictor's output.

Architecture
────────────
Input  : last K pose frames, each encoded as 13-D vector
             [p(3), q_wxyz(4), v(3), omega(3)]  — position, quaternion,
             linear velocity, angular velocity
         Shape: (1, 13, K) — batch=1, channels, time

Stack  : 4 dilated causal conv blocks (dilations 1,2,4,8)
         Each block: Conv1d → LayerNorm → GELU → residual add

Output : 12-D vector — mean Δx (6) + log-variance log σ² (6)
             Δx   = [Δp(3), Δφ(3)]
             Δφ is in so(3) — applied as q̂ = q̂_phys ⊗ qexp(Δφ)

Uncertainty
───────────
log σ² is used by the blend gate as the slow-path confidence signal.
High σ² → TCN is uncertain → blend gate reduces its weight.

Fallback
────────
If PyTorch is not installed the module degrades gracefully to a
ZeroResidualFallback that returns Δx=0 and high σ² (physics-only mode).
This keeps the rest of the pipeline functional without ML dependencies.

Training
────────
See train_tcn.py for the offline training loop on PoseLogger CSV data.
At runtime the edge server loads weights from WEIGHTS_PATH if the file
exists, otherwise falls back to zero-residual mode.
"""

import numpy as np
from pathlib import Path

WEIGHTS_PATH = Path(__file__).parent.parent / "models" / "tcn_residual.pt"

# Input / output dimensions
INPUT_DIM  = 13    # [p(3), q(4), v(3), omega(3)]
OUTPUT_DIM = 6     # [Δp(3), Δφ(3)]
K_HISTORY  = 10    # number of past frames fed to the network
HIDDEN_CH  = 32    # TCN hidden channels
DILATIONS  = [1, 2, 4, 8]


# ── Try to import PyTorch ─────────────────────────────────────────────────

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ── TCN building blocks (PyTorch) ─────────────────────────────────────────

if _TORCH_AVAILABLE:
    class _CausalConvBlock(nn.Module):
        """
        Single dilated causal convolution block with residual connection.
        Causality is enforced by left-padding only and trimming the future
        timesteps that leak in from right-padding.
        """
        def __init__(self, channels: int, dilation: int, kernel_size: int = 3):
            super().__init__()
            # Left-pad to keep sequence length constant and remain causal
            self._pad = (kernel_size - 1) * dilation
            self.conv  = nn.Conv1d(channels, channels, kernel_size,
                                   dilation=dilation, padding=0)
            self.norm  = nn.LayerNorm(channels)
            self.act   = nn.GELU()

            # 1×1 conv for residual projection (identity if same channels)
            self.residual_proj = nn.Identity()

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (B, C, T)
            residual = x
            # Manual causal left-padding
            x_pad = F.pad(x, (self._pad, 0))
            out = self.conv(x_pad)               # (B, C, T)
            # LayerNorm expects (B, T, C)
            out = self.norm(out.transpose(1, 2)).transpose(1, 2)
            out = self.act(out)
            return out + self.residual_proj(residual)

    class TCNResidualNet(nn.Module):
        """
        Full TCN that maps a history of K poses to a residual correction
        Δx ∈ R^6 and associated log-variance log σ² ∈ R^6.
        """
        def __init__(self,
                     input_dim: int = INPUT_DIM,
                     hidden: int = HIDDEN_CH,
                     output_dim: int = OUTPUT_DIM,
                     dilations: list = None,
                     k_history: int = K_HISTORY):
            super().__init__()
            if dilations is None:
                dilations = DILATIONS

            self.k_history = k_history

            # Input projection
            self.input_proj = nn.Conv1d(input_dim, hidden, kernel_size=1)

            # Dilated causal conv stack
            self.blocks = nn.ModuleList([
                _CausalConvBlock(hidden, d) for d in dilations
            ])

            # Output heads: mean and log-variance
            self.mean_head    = nn.Linear(hidden, output_dim)
            self.logvar_head  = nn.Linear(hidden, output_dim)

        def forward(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
            """
            x: (B=1, input_dim, K) — batch of one, channels, history
            Returns:
                mean   (B, output_dim) — Δx residual correction
                logvar (B, output_dim) — log σ² uncertainty estimate
            """
            out = self.input_proj(x)      # (B, hidden, K)
            for block in self.blocks:
                out = block(out)
            # Use the last time step as the prediction output
            last = out[:, :, -1]          # (B, hidden)
            mean   = self.mean_head(last)
            logvar = self.logvar_head(last)
            return mean, logvar


# ── Runtime wrapper ───────────────────────────────────────────────────────

class TCNResidualPredictor:
    """
    Runtime wrapper around TCNResidualNet.
    Maintains per-client pose history and runs inference.

    Usage:
        predictor = TCNResidualPredictor()
        delta_x, sigma2 = predictor.predict(user_id, feature_vector)
    """
    def __init__(self):
        self._model = None
        self._histories: dict[str, list] = {}   # user_id → list of feature vecs

        if _TORCH_AVAILABLE:
            self._model = TCNResidualNet()
            self._model.eval()
            if WEIGHTS_PATH.exists():
                state = torch.load(str(WEIGHTS_PATH),
                                   map_location="cpu",
                                   weights_only=True)
                self._model.load_state_dict(state)
                print(f"[TCN] Loaded weights from {WEIGHTS_PATH}")
            else:
                print("[TCN] No weights file found — running zero-residual "
                      f"fallback until {WEIGHTS_PATH} is provided. "
                      "Run train_tcn.py to generate weights.")
        else:
            print("[TCN] PyTorch not available — zero-residual fallback active.")

    # ── Feature encoding ─────────────────────────────────────────────────

    @staticmethod
    def encode_frame(p: np.ndarray,
                     q: np.ndarray,
                     v: np.ndarray,
                     omega: np.ndarray) -> np.ndarray:
        """
        Encode a single pose measurement into a 13-D feature vector.
        All arrays use the internal [w,x,y,z] quaternion convention.
        """
        return np.concatenate([p, q, v, omega]).astype(np.float32)

    def push_frame(self, user_id: str, feature: np.ndarray) -> None:
        """Add a new frame to the rolling history for a client."""
        if user_id not in self._histories:
            self._histories[user_id] = []
        hist = self._histories[user_id]
        hist.append(feature)
        # Keep only the last K_HISTORY frames
        if len(hist) > K_HISTORY:
            self._histories[user_id] = hist[-K_HISTORY:]

    # ── Inference ─────────────────────────────────────────────────────────

    def predict(self, user_id: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict residual correction for head pose (primary body part).

        Returns:
            delta_x  np.ndarray (6,) — [Δp(3), Δφ(3)] correction
            sigma2   np.ndarray (6,) — per-dimension variance (uncertainty)

        Falls back to zeros if history is too short or model unavailable.
        """
        hist = self._histories.get(user_id, [])

        # Not enough history yet or no model
        if len(hist) < K_HISTORY or self._model is None or not _TORCH_AVAILABLE:
            return np.zeros(6, dtype=np.float32), \
                   np.ones(6, dtype=np.float32) * 1e3   # very high uncertainty

        # Build (1, INPUT_DIM, K) tensor
        arr = np.stack(hist[-K_HISTORY:], axis=-1)        # (13, K)
        x   = torch.from_numpy(arr).unsqueeze(0).float()  # (1, 13, K)

        with torch.no_grad():
            mean, logvar = self._model(x)

        delta_x = mean[0].numpy().astype(np.float32)
        sigma2  = np.exp(logvar[0].numpy()).astype(np.float32)
        return delta_x, sigma2

    def is_ready(self, user_id: str) -> bool:
        """True when enough history has accumulated for inference."""
        return (len(self._histories.get(user_id, [])) >= K_HISTORY
                and self._model is not None
                and _TORCH_AVAILABLE)


# ── Zero-residual fallback (no PyTorch) ───────────────────────────────────

class ZeroResidualFallback:
    """Drop-in replacement when PyTorch is unavailable."""
    def push_frame(self, user_id, feature): pass
    def predict(self, user_id):
        return np.zeros(6, dtype=np.float32), \
               np.ones(6, dtype=np.float32) * 1e3
    def is_ready(self, user_id): return False
