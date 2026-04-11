# MWA Planning UI — Setup Guide

## Files required (all in same directory)
```
hs_directional_mwa.py   ← your existing v11 algorithm
mwa_ui_server.py        ← Flask API bridge
mwa_ui.html             ← browser wizard UI
```

## Install dependencies
```
pip install flask flask-cors pillow --break-system-packages
```
All other packages (pyvista, numpy, scipy, tqdm) should already be in your sim_env.

## On Windows (your setup)
Remove the `pv.start_xvfb()` line from mwa_ui_server.py — that is Linux-only.
PyVista on Windows renders offscreen by default when `off_screen=True` is set.

Open mwa_ui_server.py, find line ~50 and remove or comment:
    pv.start_xvfb()

## Start the server
```
cd "C:\Users\z005562w\...\Nunna Algo"
conda activate sim_env
python mwa_ui_server.py
```
Then open http://localhost:5050 in Chrome or Edge.

## Workflow in the browser
1. Upload  — drag and drop all your VTK files; assign tumor/surface/vessels
2. Tumors  — click a tumor card to select it; metrics shown inline
3. Params  — choose histology, consistency, needle entry direction
4. Results — ASI gauge, full prescription table, OAR list
           — click "Launch interactive 3D visualisation" to open PyVista window

## API endpoints (for debugging / scripting)
POST /upload          multipart form: tumor_vtk, surface_vtk, vessel_vtks[]
POST /analyse         loads meshes, extracts tumors, returns metrics + PNG
POST /select_tumor    { tumor_idx: 0 } → runs ray tracing, returns PNG
POST /optimise        { tumor_type_key, consistency_key, entry_axis_key }
POST /visualise       launches PyVista window in background thread
GET  /snapshot        returns current scene as base64 PNG
POST /reset           clears all session data

## Notes
- The server is single-user (one session dict in memory).
- For multi-user or production use, replace SESSION dict with
  Flask-Session or a proper session store keyed by session ID.
- The offscreen PNG rendering requires the Pillow library (pip install pillow).
- If PyVista offscreen fails on your GPU, set the env var before starting:
    set PYVISTA_OFF_SCREEN=true
