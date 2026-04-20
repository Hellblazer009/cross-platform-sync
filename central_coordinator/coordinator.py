"""
coordinator.py
──────────────
Central Coordinator — lightweight reference server for the PREP system.

Responsibilities
────────────────
1. Reference clock  — all edge servers and clients sync to this clock,
                      making cross-edge latency measurements comparable.
2. Session registry — tracks which clients are in the session and which
                      edge server each belongs to.
3. Global telemetry — edge servers report per-client stats here;
                      coordinator aggregates session-wide metrics.
4. Model registry   — serves the current TCN weights file path so edge
                      servers stay in sync on the same model version.

Endpoints
─────────
POST /register_edge    Edge server announces itself
POST /register_client  Client (via its edge server) joins session
POST /deregister       Client or edge server leaves
GET  /session          Full session roster
POST /sync_clock       Reference clock sync (same protocol as edge server)
POST /sync_ack         Completes clock sync round
POST /telemetry/report Edge server pushes per-client stats
GET  /telemetry        Aggregated session-wide metrics
GET  /model/version    Current TCN model version + download URL

Runs on port 9000. Edge servers connect to it on startup.
"""

import time
from collections import defaultdict
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


app = FastAPI(title="PREP Central Coordinator", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ── In-memory state ───────────────────────────────────────────────────────

# edge_id → { url, registered_at, last_heartbeat }
_edge_servers: dict[str, dict] = {}

# user_id → { device_type, edge_id, joined_at, last_seen }
_clients: dict[str, dict] = {}

# user_id → { pending_t_c1, pending_t_e } for clock sync rounds
_sync_pending: dict[str, dict] = {}

# Telemetry store: user_id → latest report dict from edge server
_telemetry: dict[str, dict] = {}

# Model registry
_model_version = "0.0.0"
_model_url     = ""


# ── Request models ────────────────────────────────────────────────────────

class EdgeRegistration(BaseModel):
    edge_id:  str
    url:      str          # http://ip:port — so clients can reach it


class ClientRegistration(BaseModel):
    user_id:     str
    device_type: str
    edge_id:     str       # which edge server this client belongs to


class DeregisterRequest(BaseModel):
    entity_id: str         # user_id or edge_id


class SyncClockRequest(BaseModel):
    entity_id: str         # user_id or edge_id performing sync
    t_c1:      float


class SyncAckRequest(BaseModel):
    entity_id: str
    t_c1:      float
    t_e:       float
    t_c2:      float


class TelemetryReport(BaseModel):
    edge_id:   str
    user_id:   str
    n_frames:  int
    avg_lat_ms:  Optional[float] = None
    p95_lat_ms:  Optional[float] = None
    last_alpha:  Optional[float] = None
    last_mahal:  Optional[float] = None


# ── Edge server registration ──────────────────────────────────────────────

@app.post("/register_edge")
async def register_edge(req: EdgeRegistration):
    _edge_servers[req.edge_id] = {
        "url":              req.url,
        "registered_at":    time.monotonic(),
        "last_heartbeat":   time.monotonic(),
    }
    print(f"[Coordinator] Edge registered: {req.edge_id} @ {req.url}")
    return {"status": "ok", "coordinator_time": time.monotonic()}


# ── Client registration ───────────────────────────────────────────────────

@app.post("/register_client")
async def register_client(req: ClientRegistration):
    _clients[req.user_id] = {
        "device_type": req.device_type,
        "edge_id":     req.edge_id,
        "joined_at":   time.monotonic(),
        "last_seen":   time.monotonic(),
    }
    print(f"[Coordinator] Client joined: {req.user_id} "
          f"({req.device_type}) via {req.edge_id}")
    return {"status": "ok", "n_clients": len(_clients)}


@app.post("/deregister")
async def deregister(req: DeregisterRequest):
    removed = False
    if req.entity_id in _clients:
        del _clients[req.entity_id]
        removed = True
    if req.entity_id in _edge_servers:
        del _edge_servers[req.entity_id]
        removed = True
    return {"status": "ok", "removed": removed}


# ── Session info ──────────────────────────────────────────────────────────

@app.get("/session")
async def session():
    return {
        "n_clients":    len(_clients),
        "n_edges":      len(_edge_servers),
        "clients":      _clients,
        "edge_servers": {eid: {k: v for k, v in info.items()
                               if k != "last_heartbeat"}
                         for eid, info in _edge_servers.items()},
    }


# ── Reference clock sync ──────────────────────────────────────────────────
# Same bidirectional exchange protocol as the edge server's /sync_clock,
# but this is the ROOT reference clock that edge servers themselves sync to.

@app.post("/sync_clock")
async def sync_clock(req: SyncClockRequest):
    """
    Step 1: edge server (or client) sends its t_c1.
    We record t_e (coordinator monotonic time) and return it.
    """
    t_e = time.monotonic()
    _sync_pending[req.entity_id] = {"t_c1": req.t_c1, "t_e": t_e}
    return {"t_e": t_e}


@app.post("/sync_ack")
async def sync_ack(req: SyncAckRequest):
    """
    Step 2: edge server sends back t_c1, t_e, t_c2.
    Compute and return offset estimate.
    """
    offset_estimate = req.t_e - (req.t_c1 + req.t_c2) / 2.0
    rtt             = req.t_c2 - req.t_c1
    _sync_pending.pop(req.entity_id, None)
    return {
        "status":     "ok",
        "offset_ms":  round(offset_estimate * 1000, 3),
        "rtt_ms":     round(rtt * 1000, 3),
    }


# ── Telemetry ─────────────────────────────────────────────────────────────

@app.post("/telemetry/report")
async def telemetry_report(report: TelemetryReport):
    """Edge servers push per-client stats here."""
    _telemetry[report.user_id] = {
        "edge_id":     report.edge_id,
        "n_frames":    report.n_frames,
        "avg_lat_ms":  report.avg_lat_ms,
        "p95_lat_ms":  report.p95_lat_ms,
        "last_alpha":  report.last_alpha,
        "last_mahal":  report.last_mahal,
        "reported_at": time.monotonic(),
    }
    return {"status": "ok"}


@app.get("/telemetry")
async def telemetry_summary():
    """Aggregate session-wide latency and quality metrics."""
    if not _telemetry:
        return {"status": "no_data"}

    lats = [v["avg_lat_ms"] for v in _telemetry.values()
            if v.get("avg_lat_ms") is not None]
    alphas = [v["last_alpha"] for v in _telemetry.values()
              if v.get("last_alpha") is not None]

    return {
        "n_clients":          len(_telemetry),
        "session_avg_lat_ms": round(sum(lats) / len(lats), 3) if lats else None,
        "session_max_lat_ms": round(max(lats), 3) if lats else None,
        "session_avg_alpha":  round(sum(alphas) / len(alphas), 3) if alphas else None,
        "per_client":         _telemetry,
    }


# ── Model registry ────────────────────────────────────────────────────────

@app.get("/model/version")
async def model_version():
    return {
        "version": _model_version,
        "url":     _model_url,
    }
