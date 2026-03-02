# Cross-Platform XR Pose Synchronization Framework

A lightweight distributed framework for cross-platform XR pose synchronization using:

- **Unity** (VR / AR / Desktop clients)
- **Normcore** (central relay server)
- **Python-based edge server** (pose interpretation layer)

This project demonstrates an edge-mediated architecture for decoupling raw pose capture from semantic pose interpretation in multi-user XR systems.

---

## 🧠 System Overview

The framework introduces an edge server between Unity clients and the central Normcore server.

### Data Flow
Unity Clients (Raw Pose)
↓
Python Edge Server (Interpretation + Processing)
↓
Unity Edge Publisher
↓
Normcore Relay Server
↓
All Clients (Interpreted Pose Rendering)


### Raw Pose Frame

Captured directly from XR devices:

- Position  
- Rotation  
- Tracking mode (6DoF / 3DoF / Synthetic)  
- Confidence  

### Interpreted Pose Frame

Processed at the edge:

- Pose type classification  
- Optional smoothing  
- Confidence adaptation  
- Normalized output for rendering  

---

## 📂 Repository Structure
- edge_server/ → Python-based pose interpretation server
- unity_scripts/ → Unity C# scripts (not full Unity project)
- docs/ → 


> Note: The full Unity project is not included to avoid repository bloat.  
> The provided scripts are sufficient to recreate the project inside a new Unity project.

---

## ⚙️ Edge Server Setup

Navigate to the edge server directory:
cd edge_server

Run the server:
uvicorn edge_server:app --host 0.0.0.0 --port 8000


The edge server will:

- Accept raw pose frames via HTTP  
- Interpret and optionally smooth pose data  
- Forward interpreted poses to the Unity Edge Publisher  

---

## 🎮 Unity Setup

WIP

---

## 🧪 Modes of Operation

### Baseline Mode

Direct Normcore pose synchronization without edge processing.

### Edge Mode

Raw pose → edge interpretation → relayed interpreted pose.

This enables controlled experimentation between centralized and distributed architectures.

---

## 📜 License

