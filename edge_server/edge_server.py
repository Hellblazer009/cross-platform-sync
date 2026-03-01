from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import asyncio
import websockets
import json
import time

app = FastAPI()

# -------- Data Model -------- #

class RawPoseFrame(BaseModel):
    user_id: str
    device_type: str       # VR, AR, Desktop
    tracking_mode: str     # 6DoF, 3DoF, Synthetic
    timestamp: float
    position: list[float]
    rotation: list[float]
    confidence: float

# -------- Pose Processing -------- #

def smooth_position(current, previous, alpha=0.6):
    if previous is None:
        return current
    return (alpha * np.array(current) + (1 - alpha) * np.array(previous)).tolist()

previous_positions = {}

def interpret_pose(pose: RawPoseFrame):
    # --- Interpretation logic ---
    if pose.tracking_mode == "6DoF":
        interpretation = "FullBody"
    elif pose.tracking_mode == "3DoF":
        interpretation = "HeadOnly"
    else:
        interpretation = "Proxy"

    # --- Optional smoothing ---
    prev = previous_positions.get(pose.user_id)
    smoothed_position = smooth_position(pose.position, prev)
    previous_positions[pose.user_id] = smoothed_position

    adjusted_confidence = pose.confidence
    if interpretation == "HeadOnly":
        adjusted_confidence *= 0.75

    return {
        "user_id": pose.user_id,
        "interpretation": interpretation,
        "position": smoothed_position,
        "rotation": pose.rotation,
        "confidence": adjusted_confidence,
        "server_timestamp": time.time()
    }

# -------- WebSocket to Unity Edge Publisher -------- #

UNITY_EDGE_WS = "ws://localhost:8765"

async def publish_to_unity(interpreted_pose):
    try:
        async with websockets.connect(UNITY_EDGE_WS) as websocket:
            await websocket.send(json.dumps(interpreted_pose))
    except:
        print("Unity edge publisher not available")

# -------- API Endpoint -------- #

@app.post("/pose")
async def receive_pose(pose: RawPoseFrame):
    interpreted = interpret_pose(pose)
    asyncio.create_task(publish_to_unity(interpreted))
    return {"status": "ok"}