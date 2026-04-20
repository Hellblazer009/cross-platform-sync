"""
train_tcn.py
────────────
Offline training script for the TCN residual network.

Input data
──────────
PoseLogger CSV files from Unity (one per session), with columns:
    timestamp, headPos, leftHandPos, rightHandPos
Example row:
    12.34,"(0.123, 1.543, -0.021)","(0.234, 1.12, 0.04)","(-0.21, 1.11, 0.03)"

Training objective
──────────────────
Given a window of K=10 past encoded frames [p, q, v, ω], predict the
residual Δx ∈ R^6 between the physics predictor's estimate and the
true next measurement:

    Δx_t = x_{t+1}^{true} - x_{t+1}^{physics}

The network outputs (mean, log_var) and is trained with a
Gaussian negative log-likelihood loss:

    L = 0.5 * sum((Δx - mean)^2 / exp(log_var) + log_var)

This trains the uncertainty head to be calibrated alongside the mean.

Usage
─────
    python train_tcn.py --data_dir /path/to/pose_logs --epochs 50

The trained model is saved to edge_server/models/tcn_residual.pt and
loaded automatically by TCNResidualPredictor at runtime.
"""

import argparse
import re
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False
    print("PyTorch not installed — cannot run training.")
    exit(1)

from prep.physics_predictor import (
    BodyPartPredictor, unity_to_wxyz_np,
    qlog, qmul, qnorm,
)
from prep.tcn_residual import TCNResidualNet, K_HISTORY, INPUT_DIM, OUTPUT_DIM

MODELS_DIR = Path(__file__).parent / "models"


# ── Unity string parsers ──────────────────────────────────────────────────

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def parse_unity_vec3(s: str) -> np.ndarray:
    """Parse Unity's Vector3.ToString() format: '(x, y, z)'"""
    nums = _NUM_RE.findall(s)
    return np.array([float(n) for n in nums[:3]], dtype=np.float64)


def parse_unity_quat_as_wxyz(s: str) -> np.ndarray:
    """
    Parse PoseLogger quaternion format: '(x, y, z, w)' (Unity convention)
    and return [w, x, y, z] for internal physics predictor use.
    """
    nums = _NUM_RE.findall(s)
    x, y, z, w = (float(n) for n in nums[:4])
    q = np.array([w, x, y, z], dtype=np.float64)
    norm = np.linalg.norm(q)
    return q / norm if norm > 1e-8 else np.array([1.0, 0.0, 0.0, 0.0])


# ── Dataset ───────────────────────────────────────────────────────────────

class PoseLogDataset(Dataset):
    """
    Loads PoseLogger CSV files and builds (input_sequence, target_delta)
    training pairs.

    For each frame t, input  = last K encoded feature vectors
                     target  = Δx = [Δp(3), Δφ(3)] between physics
                               prediction and true next head position
    """
    def __init__(self, csv_paths: list[Path], k: int = K_HISTORY):
        self.k = k
        self.sequences: list[np.ndarray] = []   # (K, INPUT_DIM)
        self.targets:   list[np.ndarray] = []   # (OUTPUT_DIM,)

        for path in csv_paths:
            self._load_csv(path)

        print(f"[Dataset] {len(self.sequences)} training samples "
              f"from {len(csv_paths)} files.")

    # Expected CSV columns (PoseLogger v2):
    #   timestamp, headPos, headRot, leftHandPos, leftHandRot,
    #   rightHandPos, rightHandRot
    # Legacy CSV (v1, position-only):
    #   timestamp, headPos, leftHandPos, rightHandPos
    # Both formats are handled: v1 falls back to identity quaternion + zero Δφ.

    def _load_csv(self, path: Path) -> None:
        predictor = BodyPartPredictor()
        features:    list[np.ndarray] = []
        positions:   list[np.ndarray] = []
        quaternions: list[np.ndarray] = []
        times:       list[float]      = []

        with open(path) as f:
            lines = f.readlines()

        if len(lines) < 2:
            return

        # Detect format from header
        header = lines[0].strip().lower()
        has_rotation = "headrot" in header or "head_rot" in header

        _group_re = re.compile(r"\([^)]+\)")

        for line in lines[1:]:
            parts = line.strip().split(",", 1)
            if len(parts) < 2:
                continue
            try:
                t     = float(parts[0])
                groups = _group_re.findall(parts[1])
                if not groups:
                    continue

                if has_rotation:
                    # groups: [headPos, headRot, lhPos, lhRot, rhPos, rhRot]
                    if len(groups) < 2:
                        continue
                    p = parse_unity_vec3(groups[0])
                    q = parse_unity_quat_as_wxyz(groups[1])
                else:
                    # Legacy: groups[0] = headPos, no rotation
                    p = parse_unity_vec3(groups[0])
                    q = np.array([1.0, 0.0, 0.0, 0.0])
            except Exception:
                continue

            predictor.update(p, q, t)

            v  = predictor._v     if predictor._v     is not None else np.zeros(3)
            om = predictor._omega if predictor._omega is not None else np.zeros(3)

            feature = np.concatenate([p, q, v, om]).astype(np.float32)
            features.append(feature)
            positions.append(p)
            quaternions.append(q)
            times.append(t)

        if len(features) < self.k + 2:
            return

        # ── Rewind predictor to build (window, target) pairs ──────────────
        # We need the physics prediction at each step. Use a fresh predictor
        # fed up to step i-1 to predict step i.
        pred2 = BodyPartPredictor()
        for i in range(len(features)):
            pred2.update(positions[i], quaternions[i], times[i])

            if i < self.k:
                continue

            # Physics prediction for step i+1
            h = times[i+1] - times[i] if i + 1 < len(times) else (1.0 / 30.0)
            if h <= 0:
                h = 1.0 / 30.0
            p_pred, q_pred = pred2.predict(h)

            if i + 1 >= len(positions):
                break

            p_true = positions[i + 1]
            q_true = quaternions[i + 1]

            # Positional residual
            delta_p = p_true - p_pred

            # Orientation residual in Lie algebra: log(q_pred^{-1} ⊗ q_true)
            if has_rotation:
                q_pred_conj = np.array([ q_pred[0],
                                        -q_pred[1], -q_pred[2], -q_pred[3]])
                q_err = qmul(q_pred_conj, q_true)
                delta_phi = qlog(q_err)          # R^3
            else:
                delta_phi = np.zeros(3)

            window = np.stack(features[i - self.k:i], axis=0)   # (K, INPUT_DIM)
            target = np.concatenate([delta_p, delta_phi]).astype(np.float32)
            self.sequences.append(window)
            self.targets.append(target)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # x: (INPUT_DIM, K) — channel-first for Conv1d
        x = torch.from_numpy(self.sequences[idx].T)    # (13, K)
        y = torch.from_numpy(self.targets[idx])         # (6,)
        return x, y


# ── Training loop ─────────────────────────────────────────────────────────

def gaussian_nll_loss(mean: "torch.Tensor",
                      log_var: "torch.Tensor",
                      target: "torch.Tensor") -> "torch.Tensor":
    """
    Gaussian negative log-likelihood:
        L = 0.5 * mean_over_dims( (y - μ)² / σ² + log σ² )
    Trains both the mean and uncertainty heads jointly.
    """
    var = torch.exp(log_var)
    return 0.5 * torch.mean((target - mean) ** 2 / var + log_var)


def train(data_dir: str,
          epochs:   int   = 50,
          lr:       float = 1e-3,
          batch:    int   = 64,
          val_frac: float = 0.1) -> None:

    csv_paths = list(Path(data_dir).glob("*.txt")) + \
                list(Path(data_dir).glob("*.csv"))
    if not csv_paths:
        print(f"No CSV/TXT files found in {data_dir}.")
        return

    dataset  = PoseLogDataset(csv_paths)
    if len(dataset) < 50:
        print("Too few training samples — collect more pose log data.")
        return

    # Train / val split
    n_val   = max(1, int(len(dataset) * val_frac))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch)

    model     = TCNResidualNet()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, epochs)

    best_val_loss = float("inf")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODELS_DIR / "tcn_residual.pt"

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            optimiser.zero_grad()
            mean, log_var = model(x)
            loss = gaussian_nll_loss(mean, log_var, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            train_loss += loss.item() * len(x)
        train_loss /= n_train

        # ── Validate ───────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                mean, log_var = model(x)
                val_loss += gaussian_nll_loss(mean, log_var, y).item() * len(x)
        val_loss /= n_val

        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), str(save_path))

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Weights saved → {save_path}")


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PREP TCN residual network")
    parser.add_argument("--data_dir", required=True,
                        help="Directory containing PoseLogger .txt/.csv files")
    parser.add_argument("--epochs",   type=int,   default=50)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--batch",    type=int,   default=64)
    args = parser.parse_args()

    train(args.data_dir, args.epochs, args.lr, args.batch)
