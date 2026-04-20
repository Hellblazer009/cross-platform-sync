"""
edge_server.py
──────────────
PREP Edge Server — Physics-Residual Edge Predictor

Endpoints
─────────
POST /pose              Receive raw pose frame from Unity client
POST /sync_clock        Step 1 of clock sync (client sends t_c1)
POST /sync_ack          Step 2 of clock sync (client sends t_c1, t_e, t_c2)
GET  /session           Current session roster (for debugging / coordinator)
GET  /telemetry         Per-client diagnostic snapshot
WS   /ws                Persistent WebSocket — Unity EdgePoseReceiver connects
                        here to receive interpreted/predicted pose frames

Architecture note
─────────────────
Unity clients POST their raw poses here. The server runs the PREP pipeline
and pushes interpreted poses to all subscribed WebSocket listeners.
The WebSocket server (not client) pattern means edge server can be remote —
Unity connects OUT to ws://edge_server_ip:8000/ws.

Operational modes (set via OPERATIONAL_MODE env var or config)
──────────────────────────────────────────────────────────────
  "baseline"  — bypass PREP; relay raw poses directly via Normcore model
                (used for comparison experiments)
  "edge"      — full PREP pipeline (default)

CWAR relay scheduling
─────────────────────
Each client has an adaptive relay interval τ_i (seconds). The server only
pushes a new frame to WebSocket subscribers when elapsed ≥ τ_i.
τ_i adapts based on motion velocity and device confidence weight.
"""

import asyncio
import json
import os
import time
from collections import defaultdict
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from prep import (
    ClockSyncRegistry,
    PosePredictor,
    TCNResidualPredictor,
    BlendGate, BlendGateState,
    qlog, qmul, qnorm, qexp,
)


# ── Configuration ─────────────────────────────────────────────────────────

OPERATIONAL_MODE  = os.environ.get("OPERATIONAL_MODE", "edge")   # "baseline" | "edge"
COORDINATOR_URL   = os.environ.get("COORDINATOR_URL", "http://localhost:9000")
EDGE_SERVER_ID    = os.environ.get("EDGE_SERVER_ID", "edge_1")

# CWAR relay parameters
RELAY_RATE_BASE   = 30.0    # Hz — base relay rate (f_p)
RELAY_RATE_MIN    = 5.0     # Hz — minimum (very idle client)
RELAY_RATE_MAX    = 90.0    # Hz — maximum (fast motion, high confidence)
VELOCITY_THRESH   = 0.5     # m/s — above this, relay rate starts scaling up

# Device confidence weights (placeholder — replace with empirical σ_min/σ_i)
DEVICE_WEIGHTS = {
    "VR_6DOF":   1.00,
    "AR_6DOF":   1.00,   # updated from measurement once empirical data is in
    "DOF_3":     1.00,
    "SYNTHETIC": 1.00,
}


# ── Data models (matching Unity RawPoseFrame.cs) ──────────────────────────

class RawPoseFrame(BaseModel):
    user_id:            str
    device_type:        str           # "VR_6DOF" | "AR_6DOF" | "DOF_3" | "SYNTHETIC"
    tracking_mode:      str           # "6DoF" | "3DoF" | "Synthetic"
    timestamp:          float         # client local time (monotonic seconds)
    # Head pose — Unity [x,y,z,w] quaternion convention
    head_position:      list[float]   # [x, y, z]
    head_rotation:      list[float]   # [x, y, z, w]
    # Left hand
    left_hand_position: list[float]
    left_hand_rotation: list[float]
    # Right hand
    right_hand_position: list[float]
    right_hand_rotation: list[float]
    confidence:          float


class SyncClockRequest(BaseModel):
    user_id: str
    t_c1:    float     # client monotonic time at send


class SyncAckRequest(BaseModel):
    user_id: str
    t_c1:    float
    t_e:     float     # edge time echoed back
    t_c2:    float     # client monotonic time at receive


# ── Per-client state ──────────────────────────────────────────────────────

class ClientState:
    """All mutable state for one connected client."""
    def __init__(self, user_id: str, device_type: str):
        self.user_id      = user_id
        self.device_type  = device_type
        self.predictor    = PosePredictor()
        self.gate_state   = BlendGateState()
        self.last_relay_t = 0.0          # edge time of last WebSocket push
        self.last_p       = None         # for velocity estimate
        self.last_t       = None
        self.n_frames     = 0
        # Telemetry
        self.latencies: list[float] = []


# ── Helpers ───────────────────────────────────────────────────────────────

def unity_to_wxyz(q: list[float]) -> np.ndarray:
    """Unity [x,y,z,w] → internal [w,x,y,z]."""
    x, y, z, w = q
    return np.array([w, x, y, z], dtype=np.float64)


def wxyz_to_unity(q: np.ndarray) -> list[float]:
    """Internal [w,x,y,z] → Unity [x,y,z,w]."""
    w, x, y, z = q
    return [float(x), float(y), float(z), float(w)]


def interpret_tracking(tracking_mode: str, confidence: float) -> tuple[str, float]:
    """Map tracking mode string to semantic label and adjust confidence."""
    if tracking_mode == "6DoF":
        return "FullBody", confidence
    elif tracking_mode == "3DoF":
        return "HeadOnly", confidence * 0.75
    else:
        return "Proxy", confidence * 0.40


def cwar_relay_interval(velocity: float,
                         device_weight: float,
                         spatial_priority: float = 1.0) -> float:
    """
    CWAR adaptive relay interval (seconds).
    τ_i = τ_base / (w_i × min(v/v_thresh, 1) × ρ_i)
    Clamped to [1/RELAY_RATE_MAX, 1/RELAY_RATE_MIN].
    """
    v_factor = min(velocity / max(VELOCITY_THRESH, 1e-6), 1.0)
    # Avoid division by zero — use base rate if all factors are near zero
    denom = device_weight * max(v_factor, 0.05) * max(spatial_priority, 0.1)
    tau = (1.0 / RELAY_RATE_BASE) / denom
    return float(np.clip(tau, 1.0 / RELAY_RATE_MAX, 1.0 / RELAY_RATE_MIN))


# ── App setup ─────────────────────────────────────────────────────────────

app = FastAPI(title="PREP Edge Server", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# Singletons
clock_registry  = ClockSyncRegistry()
tcn_predictor   = TCNResidualPredictor()
blend_gate      = BlendGate()
client_states:  dict[str, ClientState] = {}
ws_subscribers: set[WebSocket] = set()   # active Unity EdgePoseReceiver connections


# ── WebSocket endpoint ────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Unity EdgePoseReceiver connects here.
    The server pushes interpreted pose frames as JSON messages.
    One connection per Unity instance (multiple clients can share one WS).
    """
    await websocket.accept()
    ws_subscribers.add(websocket)
    print(f"[WS] New subscriber — total: {len(ws_subscribers)}")
    try:
        while True:
            # Keep connection alive; we push data from /pose handler
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        ws_subscribers.discard(websocket)
        print(f"[WS] Subscriber disconnected — total: {len(ws_subscribers)}")


async def push_to_subscribers(payload: dict) -> None:
    """Broadcast an interpreted pose frame to all connected Unity listeners."""
    if not ws_subscribers:
        return
    msg = json.dumps(payload)
    dead = set()
    for ws in ws_subscribers:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.add(ws)
    ws_subscribers -= dead


# ── Clock sync endpoints ──────────────────────────────────────────────────

@app.post("/sync_clock")
async def sync_clock(req: SyncClockRequest):
    """Step 1: client sends t_c1; server returns its current time t_e."""
    model = clock_registry.get_or_create(req.user_id)
    t_e = model.record_sync_request(req.t_c1)
    return {"t_e": t_e}


@app.post("/sync_ack")
async def sync_ack(req: SyncAckRequest):
    """Step 2: client sends t_c1, t_e, t_c2 → edge computes offset and skew."""
    model = clock_registry.get_or_create(req.user_id)
    model.record_sync_ack(req.t_c1, req.t_e, req.t_c2)
    return {
        "status":   "ok",
        "offset_ms": round(model.offset * 1000, 3),
        "skew":      round(model.skew, 6),
        "n_syncs":   model.n_syncs,
    }


# ── Main pose endpoint ────────────────────────────────────────────────────

@app.post("/pose")
async def receive_pose(pose: RawPoseFrame):
    """
    Core endpoint. Runs the full PREP pipeline per client and relays
    the result to WebSocket subscribers if the relay interval has elapsed.
    """
    t_recv = time.monotonic()   # edge server reception time

    # ── 1. Clock correction ───────────────────────────────────────────
    t_corrected = clock_registry.correct_timestamp(pose.user_id, pose.timestamp)

    # ── 2. Get / create per-client state ──────────────────────────────
    if pose.user_id not in client_states:
        client_states[pose.user_id] = ClientState(pose.user_id, pose.device_type)
    state = client_states[pose.user_id]
    state.n_frames += 1

    # ── 3. Convert quaternions: Unity [x,y,z,w] → internal [w,x,y,z] ─
    head_p  = np.array(pose.head_position,        dtype=np.float64)
    head_q  = unity_to_wxyz(pose.head_rotation)
    lhand_p = np.array(pose.left_hand_position,   dtype=np.float64)
    lhand_q = unity_to_wxyz(pose.left_hand_rotation)
    rhand_p = np.array(pose.right_hand_position,  dtype=np.float64)
    rhand_q = unity_to_wxyz(pose.right_hand_rotation)

    # ── 4. Semantic interpretation ────────────────────────────────────
    interpretation, adj_confidence = interpret_tracking(
        pose.tracking_mode, pose.confidence
    )

    # ── 5. BASELINE mode — skip PREP, relay raw pose directly ─────────
    if OPERATIONAL_MODE == "baseline":
        payload = _build_payload(
            pose.user_id, interpretation, t_corrected, t_recv,
            head_p, head_q, lhand_p, lhand_q, rhand_p, rhand_q,
            adj_confidence,
            uncertainty=np.zeros(6), alpha=0.0,
            source="baseline", mahalanobis=0.0,
        )
        asyncio.create_task(push_to_subscribers(payload))
        return {"status": "ok", "mode": "baseline"}

    # ── 6. PREP pipeline ──────────────────────────────────────────────

    # 6a. Compute 6D prediction error against previous physics estimate
    #     (before updating predictor so we compare new measurement vs old pred)
    error_6d = state.predictor.prediction_error_6d(head_p, head_q, part="head")

    # 6b. Update physics predictor with new measurement
    state.predictor.update(
        head_p, head_q, lhand_p, lhand_q, rhand_p, rhand_q, t_corrected
    )

    # 6c. TCN residual prediction
    #     Build feature vector from physics predictor internal state
    head_pred = state.predictor.head
    v   = head_pred._v   if head_pred._v is not None else np.zeros(3)
    om  = head_pred._omega if head_pred._omega is not None else np.zeros(3)
    feature = TCNResidualPredictor.encode_frame(head_p, head_q, v, om)
    tcn_predictor.push_frame(pose.user_id, feature)
    delta_x, sigma2 = tcn_predictor.predict(pose.user_id)

    # 6d. Physics prediction (current best estimate, h=0 = no extrapolation)
    phys_out = state.predictor.predict(h=0.0)
    p_phys   = np.array(phys_out["head_position"])
    q_phys   = np.array(phys_out["head_rotation"])

    # 6e. Blend gate
    p_blend, q_blend, alpha, source = blend_gate.compute(
        state=state.gate_state,
        error_6d=error_6d,
        delta_x=delta_x,
        sigma2=sigma2,
        p_phys=p_phys,
        q_phys=q_phys,
    )

    # For hands: use physics prediction directly (TCN head-only for now)
    lp_out = np.array(phys_out["left_hand_position"])
    lq_out = np.array(phys_out["left_hand_rotation"])
    rp_out = np.array(phys_out["right_hand_position"])
    rq_out = np.array(phys_out["right_hand_rotation"])

    # ── 7. CWAR relay rate control ────────────────────────────────────
    velocity = float(np.linalg.norm(head_pred._v)) if head_pred._v is not None else 0.0
    w_device = DEVICE_WEIGHTS.get(pose.device_type, 1.0)
    tau_i    = cwar_relay_interval(velocity, w_device)

    elapsed = t_recv - state.last_relay_t
    if elapsed < tau_i:
        return {"status": "throttled", "next_relay_in_ms": round((tau_i - elapsed) * 1000)}

    state.last_relay_t = t_recv

    # ── 8. Build and push interpreted frame ───────────────────────────
    mahal = state.gate_state.last_mahalanobis
    payload = _build_payload(
        pose.user_id, interpretation, t_corrected, t_recv,
        p_blend, q_blend, lp_out, lq_out, rp_out, rq_out,
        adj_confidence, sigma2, alpha, source, mahal,
    )

    asyncio.create_task(push_to_subscribers(payload))

    # Latency instrumentation: L = t_recv - t_corrected (edge processing delay)
    state.latencies.append((t_recv - t_corrected) * 1000)
    if len(state.latencies) > 500:
        state.latencies = state.latencies[-500:]

    return {"status": "ok", "mode": "edge", "alpha": round(alpha, 3)}


def _build_payload(user_id, interpretation, t_corrected, t_edge,
                   head_p, head_q, lhand_p, lhand_q, rhand_p, rhand_q,
                   confidence, uncertainty=None, alpha=0.0,
                   source="physics", mahalanobis=0.0) -> dict:
    """Assemble the InterpretedPoseFrame JSON payload."""
    if uncertainty is None:
        uncertainty = [0.0] * 6
    return {
        "user_id":               user_id,
        "interpretation":        interpretation,
        "corrected_timestamp":   t_corrected,
        "edge_timestamp":        t_edge,
        # Poses — back to Unity [x,y,z,w] convention
        "head_position":         [float(v) for v in head_p],
        "head_rotation":         wxyz_to_unity(head_q),
        "left_hand_position":    [float(v) for v in lhand_p],
        "left_hand_rotation":    wxyz_to_unity(lhand_q),
        "right_hand_position":   [float(v) for v in rhand_p],
        "right_hand_rotation":   wxyz_to_unity(rhand_q),
        # Quality
        "confidence":            float(confidence),
        "uncertainty":           [float(v) for v in uncertainty],
        "blend_alpha":           float(alpha),
        "prediction_source":     source,
        "mahalanobis_distance":  float(mahalanobis),
    }


# ── Diagnostic endpoints ──────────────────────────────────────────────────

@app.get("/session")
async def session_info():
    return {
        "edge_server_id": EDGE_SERVER_ID,
        "mode":           OPERATIONAL_MODE,
        "clients":        [
            {
                "user_id":    uid,
                "device":     s.device_type,
                "n_frames":   s.n_frames,
                "avg_lat_ms": round(np.mean(s.latencies), 2) if s.latencies else None,
            }
            for uid, s in client_states.items()
        ],
        "ws_subscribers": len(ws_subscribers),
    }


@app.get("/telemetry")
async def telemetry():
    out = {}
    for uid, s in client_states.items():
        out[uid] = {
            "n_frames":          s.n_frames,
            "avg_latency_ms":    round(np.mean(s.latencies), 3) if s.latencies else None,
            "p95_latency_ms":    round(np.percentile(s.latencies, 95), 3) if len(s.latencies) > 5 else None,
            "last_alpha":        round(s.gate_state.last_alpha, 3),
            "last_mahalanobis":  round(s.gate_state.last_mahalanobis, 4),
            "clock":             repr(clock_registry.get_or_create(uid)),
        }
    return out
