#!/usr/bin/env python3
"""
MWA Planning System — UI Server
================================
Run:  python mwa_ui_server.py
Then open: http://localhost:5050  in your browser.

Requires: flask, pyvista, numpy, scipy, tqdm  (same env as hs_directional_mwa.py)
Install:  pip install flask flask-cors

This server imports the algorithm functions from hs_directional_mwa.py
(which must be in the same directory) and exposes them as a JSON API
so the browser wizard can call them step by step without blocking the UI.
"""

import os
import sys
import json
import uuid
import base64
import threading
import subprocess
import tempfile
import traceback

import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ── Make sure the algorithm module is importable ──────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

try:
    import hs1_directional_mwa as alg
    print("  ✔ Algorithm module loaded from hs1_directional_mwa.py")
except ImportError as e:
    print(f"  ✘ Cannot import hs1_directional_mwa: {e}")
    print("  Make sure hs1_directional_mwa.py is in the same directory.")
    sys.exit(1)

import pyvista as pv
# pv.start_xvfb() is Linux/WSL only — removed for Windows compatibility.
# On Windows, PyVista handles offscreen rendering automatically when
# off_screen=True is set on the Plotter (which render_overview_png() does).

app = Flask(__name__, static_folder=".")
CORS(app)

# ── In-memory session store (single-user local app) ───────────────────────
SESSION = {
    "tumor_mesh":      None,
    "surface":         None,
    "vessels":         [],
    "vnames":          [],
    "tumors":          [],
    "metrics":         [],
    "centroids":       None,
    # set after analysis
    "sel_idx":         None,
    "centroid":        None,
    "centroid_dists":  {},
    "antenna_axis":    None,
    "das_angle_deg":   None,
    "opt_result":      None,
    "asi":             None,
    "oar_list":        [],
    "all_losses":      [],
    "ray_results":     [],
    "particle_systems":[],
    # file paths
    "tumor_vtk":       None,
    "surface_vtk":     None,
    "vessel_vtks":     [],
}

UPLOAD_DIR = os.path.join(SCRIPT_DIR, "_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def ndarray_to_list(obj):
    """Recursively convert numpy types to plain Python for JSON serialisation."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: ndarray_to_list(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [ndarray_to_list(i) for i in obj]
    return obj


def render_overview_png(highlight_idx=None):
    """
    Render an offscreen PNG of the current scene.
    Returns base64-encoded PNG string.
    """
    try:
        pl = pv.Plotter(off_screen=True, window_size=[900, 700])
        pl.background_color = "#0d1117"

        s = SESSION
        if s["surface"] is not None:
            pl.add_mesh(s["surface"], color="#8899aa", opacity=0.07)

        for v, vn in zip(s["vessels"], s["vnames"]):
            col = alg.VESSEL_COLOR_MAP.get(vn, "gray")
            pl.add_mesh(v, color=col, opacity=0.65)

        for i, (t, m) in enumerate(zip(s["tumors"], s["metrics"])):
            is_target = (i == highlight_idx)
            col = alg.TUMOR_COLORS[i % len(alg.TUMOR_COLORS)]
            op  = 0.90 if is_target else 0.30
            pl.add_mesh(t, color=col, opacity=op)
            # centroid sphere
            sph = pv.Sphere(radius=0.007, center=m["centroid"])
            pl.add_mesh(sph, color="white", opacity=0.90)

        # If optimisation done, show D-shaped zone
        if s["opt_result"] is not None and s["antenna_axis"] is not None:
            opt = s["opt_result"]
            zone = alg.make_dshaped_zone(
                s["centroid"],
                opt["zone_fwd_cm"] / 100.0,
                opt["zone_diam_fwd_cm"] / 100.0,
                s["antenna_axis"],
                frac=1.0,
            )
            if zone is not None and zone.n_points > 0:
                pl.add_mesh(zone, scalars="Temperature_C", cmap="plasma",
                            clim=[37, 90], opacity=0.60)
            # Antenna axis arrow (cyan)
            ax = np.array(s["antenna_axis"])
            ax /= np.linalg.norm(ax) + 1e-9
            pl.add_mesh(
                pv.Arrow(start=s["centroid"], direction=ax, scale=0.06),
                color="cyan", opacity=0.95)

        pl.reset_camera()
        img = pl.screenshot(None, return_img=True)
        pl.close()

        # Encode to base64 PNG
        from PIL import Image
        import io
        pil_img = Image.fromarray(img)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return b64
    except Exception as exc:
        print(f"  Render error: {exc}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Serve the HTML UI."""
    return send_from_directory(SCRIPT_DIR, "mwa_ui.html")


@app.route("/upload", methods=["POST"])
def upload_files():
    """
    Receive VTK files from the browser drag-drop.
    Expected form fields:
        tumor_vtk   — single file
        surface_vtk — single file
        vessel_vtks — multiple files (up to 5)
    Returns JSON with file paths saved to _uploads/.
    """
    try:
        saved = {}

        def save(field, multi=False):
            files = request.files.getlist(field) if multi else [request.files.get(field)]
            paths = []
            for f in files:
                if f and f.filename:
                    fname = f.filename.replace(" ", "_")
                    dest  = os.path.join(UPLOAD_DIR, fname)
                    f.save(dest)
                    paths.append(dest)
            return paths

        tumor_paths   = save("tumor_vtk")
        surface_paths = save("surface_vtk")
        vessel_paths  = save("vessel_vtks", multi=True)

        if not tumor_paths or not surface_paths:
            return jsonify({"error": "tumor_vtk and surface_vtk are required"}), 400

        SESSION["tumor_vtk"]   = tumor_paths[0]
        SESSION["surface_vtk"] = surface_paths[0]
        SESSION["vessel_vtks"] = vessel_paths

        return jsonify({
            "status":       "uploaded",
            "tumor_vtk":    os.path.basename(tumor_paths[0]),
            "surface_vtk":  os.path.basename(surface_paths[0]),
            "vessel_vtks":  [os.path.basename(p) for p in vessel_paths],
            "n_vessels":    len(vessel_paths),
        })
    except Exception as exc:
        return jsonify({"error": str(exc), "trace": traceback.format_exc()}), 500


@app.route("/analyse", methods=["POST"])
def analyse():
    """
    Load VTK files, extract tumors, compute metrics.
    Returns tumor list + base64 overview PNG.
    """
    try:
        s = SESSION

        if not s["tumor_vtk"] or not s["surface_vtk"]:
            return jsonify({"error": "Upload files first via /upload"}), 400

        # Load meshes
        s["tumor_mesh"] = alg.rescale(alg.load_vtk(s["tumor_vtk"]))
        s["surface"]    = alg.rescale(alg.load_vtk(s["surface_vtk"]))

        s["vessels"] = []
        s["vnames"]  = []
        vessel_name_map = {
            "00001": "portal_vein",
            "00002": "hepatic_vein",
            "00003": "aorta",
            "00004": "ivc",
            "00005": "hepatic_artery",
        }
        for path in s["vessel_vtks"]:
            v = alg.rescale(alg.load_vtk(path))
            if v is not None:
                # Infer vessel name from filename
                bn = os.path.basename(path)
                vn = next(
                    (name for key, name in vessel_name_map.items() if key in bn),
                    os.path.splitext(bn)[0],
                )
                s["vessels"].append(v)
                s["vnames"].append(vn)

        # Extract tumors
        s["tumors"]    = alg.extract_tumors(s["tumor_mesh"])
        s["metrics"]   = alg.tumor_metrics(
            s["tumors"], s["surface"], s["vessels"], s["vnames"])
        s["centroids"] = np.array([m["centroid"] for m in s["metrics"]])

        # Render overview PNG
        png_b64 = render_overview_png()

        # Serialise metrics
        metrics_out = []
        for m in s["metrics"]:
            metrics_out.append({
                "idx":            int(m["idx"]),
                "diameter_cm":    round(float(m["diameter_cm"]), 2),
                "depth_cm":       round(float(m["depth_cm"]), 2),
                "closest_vessel": str(m["closest_vessel"]),
                "min_vessel_mm":  round(float(m["min_vessel_m"]) * 1000, 1),
                "eligible":       bool(m["eligible"]),
                "centroid":       [round(float(v), 4) for v in m["centroid"]],
            })

        return jsonify({
            "status":      "analysed",
            "n_tumors":    len(s["tumors"]),
            "n_vessels":   len(s["vessels"]),
            "vessel_names":s["vnames"],
            "metrics":     metrics_out,
            "preview_png": png_b64,
        })
    except Exception as exc:
        return jsonify({"error": str(exc), "trace": traceback.format_exc()}), 500


@app.route("/select_tumor", methods=["POST"])
def select_tumor():
    """
    Accept tumor index selection.
    Computes centroid distances, runs OAR orientation solver, ray tracing.
    Returns updated preview PNG + antenna axis + DAS angle.
    """
    try:
        s   = SESSION
        req = request.get_json()
        idx = int(req.get("tumor_idx", 0))

        if idx < 0 or idx >= len(s["metrics"]):
            return jsonify({"error": f"Invalid tumor index {idx}"}), 400

        s["sel_idx"]  = idx
        s["centroid"] = s["metrics"][idx]["centroid"]
        centroid      = np.array(s["centroid"])

        # Per-vessel centroid distances
        from scipy.spatial import cKDTree
        s["centroid_dists"] = {
            s["vnames"][i]: float(
                cKDTree(np.array(v.points)).query(centroid, k=1)[0])
            for i, v in enumerate(s["vessels"])
        }

        # OAR orientation solver
        best_axis, top5, _ = alg.find_optimal_antenna_axis(
            centroid, s["centroid_dists"], s["vnames"], needle_insertion_dir=None)

        antenna_axis, das_angle_deg = alg.refine_axis_with_vessel_coords(
            centroid, s["vessels"], s["vnames"], best_axis, s["centroid_dists"])

        s["antenna_axis"]  = antenna_axis.tolist()
        s["das_angle_deg"] = float(das_angle_deg)

        # Quick ray tracing
        rays    = alg.generate_rays(n_theta=12, n_phi=24)  # lighter for UI
        results = []
        v_pts   = [np.array(v.points) for v in s["vessels"]]
        for direction in rays:
            try:
                hits, _ = s["surface"].ray_trace(
                    centroid, centroid + direction * 0.5)
                if len(hits) == 0:
                    continue
                hit    = hits[0]
                path_d = float(np.linalg.norm(hit - centroid))
                seg_d  = {
                    vn: alg.ray_segment_dist(centroid, direction, path_d,
                                             v_pts[vi], s["centroid_dists"][vn])
                    for vi, vn in enumerate(s["vnames"])
                }
                dom_vn = min(seg_d, key=seg_d.get)
                sar_w  = alg.directional_sar_weight(direction, antenna_axis)
                hs     = alg.heat_sink_physics(
                    seg_d[dom_vn], dom_vn, 60.0, 300.0, sar_weight=sar_w)
                hs["ray_direction"] = direction
                hs["path_distance"] = path_d
                hs["sar_weight"]    = sar_w
                results.append(hs)
            except Exception:
                continue

        s["ray_results"] = results
        s["all_losses"]  = [r["loss_pct"] for r in results]

        # Render updated PNG with selected tumor highlighted
        png_b64 = render_overview_png(highlight_idx=idx)

        return jsonify({
            "status":        "tumor_selected",
            "tumor_idx":     idx,
            "diameter_cm":   round(float(s["metrics"][idx]["diameter_cm"]), 2),
            "depth_cm":      round(float(s["metrics"][idx]["depth_cm"]), 2),
            "closest_vessel":str(s["metrics"][idx]["closest_vessel"]),
            "min_vessel_mm": round(float(s["metrics"][idx]["min_vessel_m"])*1000, 1),
            "antenna_axis":  [round(float(v), 4) for v in antenna_axis],
            "das_angle_deg": round(float(das_angle_deg), 1),
            "n_rays":        len(results),
            "loss_range":    [
                round(float(np.min(s["all_losses"])), 2),
                round(float(np.max(s["all_losses"])), 2),
            ] if s["all_losses"] else [0, 0],
            "preview_png":   png_b64,
        })
    except Exception as exc:
        return jsonify({"error": str(exc), "trace": traceback.format_exc()}), 500


@app.route("/optimise", methods=["POST"])
def optimise():
    """
    Run the full directional biophysical optimizer + ASI computation.
    Body JSON: { tumor_type_key, consistency_key, entry_axis_key }
    Returns prescription + ASI scores + updated preview PNG.
    """
    try:
        s   = SESSION
        req = request.get_json()

        if s["sel_idx"] is None:
            return jsonify({"error": "Select a tumor first via /select_tumor"}), 400

        type_key    = req.get("tumor_type_key",   "HCC")
        consist_key = req.get("consistency_key",  "firm")
        entry_key   = req.get("entry_axis_key",   "AUTO")

        # Override antenna axis if user chose a specific entry direction
        entry_axes = {
            "SUPERIOR":  np.array([0., 0., -1.]),
            "ANTERIOR":  np.array([0., -1., 0.]),
            "RIGHT_LAT": np.array([-1., 0., 0.]),
            "LEFT_LAT":  np.array([1.,  0., 0.]),
            "AUTO":      None,
        }
        entry_vec = entry_axes.get(entry_key, None)
        centroid  = np.array(s["centroid"])

        if entry_vec is not None:
            # Re-solve orientation solver constrained to entry cone
            best_axis, _, _ = alg.find_optimal_antenna_axis(
                centroid, s["centroid_dists"], s["vnames"],
                needle_insertion_dir=entry_vec)
            antenna_axis, das_angle_deg = alg.refine_axis_with_vessel_coords(
                centroid, s["vessels"], s["vnames"], best_axis,
                s["centroid_dists"])
            s["antenna_axis"]  = antenna_axis.tolist()
            s["das_angle_deg"] = float(das_angle_deg)
        else:
            antenna_axis  = np.array(s["antenna_axis"])
            das_angle_deg = s["das_angle_deg"]

        sel_diam = float(s["metrics"][s["sel_idx"]]["diameter_cm"])

        # Run optimizer
        opt_result = alg.run_directional_optimizer(
            tumor_diam_cm   = sel_diam,
            tumor_type_key  = type_key,
            consistency_key = consist_key,
            centroid_dists  = s["centroid_dists"],
            vnames          = s["vnames"],
            vessels         = s["vessels"],
            tumor_centroid  = centroid,
            antenna_axis    = antenna_axis,
            margin_cm       = 0.5,
        )
        s["opt_result"] = opt_result

        # OAR identification
        oar_list = alg.identify_oars_directional(
            centroid, s["vessels"], s["vnames"],
            opt_result["zone_diam_fwd_cm"],
            opt_result["zone_diam_rear_cm"],
            opt_result["zone_fwd_cm"],
            antenna_axis)
        s["oar_list"] = oar_list

        # ASI
        asi = alg.compute_asi_v11(
            per_vessel_hs    = opt_result["per_vessel_hs"],
            clearance_report = opt_result["clearance_report"],
            tumor_diam_cm    = sel_diam,
            zone_diam_fwd_cm = opt_result["zone_diam_fwd_cm"],
            ray_losses       = s["all_losses"],
            constrained      = opt_result["constrained"],
            das_angle_deg    = float(das_angle_deg),
            antenna_axis     = antenna_axis,
            centroid_dists   = s["centroid_dists"],
            vnames           = s["vnames"],
        )
        s["asi"] = asi

        # Render updated PNG with zone
        png_b64 = render_overview_png(highlight_idx=s["sel_idx"])

        # Build serialisable OAR list
        oar_out = []
        for o in oar_list:
            oar_out.append({
                "vessel":       o["vessel"],
                "wall_clear_mm":round(float(o["wall_clear_mm"]), 1),
                "risk":         o["risk"],
                "in_rear_lobe": bool(o.get("in_rear_lobe", False)),
                "hemisphere":   o.get("hemisphere", ""),
            })

        # Heat sink summary per vessel
        hs_summary = {}
        for vn, hs in opt_result["per_vessel_hs"].items():
            hs_summary[vn] = {
                "loss_pct":    round(float(hs["loss_pct"]), 3),
                "Q_loss_W":    round(float(hs["Q_loss_W"]), 4),
                "sar_weight":  round(float(hs.get("sar_weight", 1.0)), 3),
                "flow_regime": str(hs["flow_regime"]),
            }

        return jsonify({
            "status":            "optimised",
            "prescription": {
                "power_w":          round(float(opt_result["P_opt"]), 1),
                "time_s":           round(float(opt_result["t_opt"]), 0),
                "time_min":         round(float(opt_result["t_opt"])/60, 1),
                "zone_fwd_diam_cm": round(float(opt_result["zone_diam_fwd_cm"]), 2),
                "zone_rear_diam_cm":round(float(opt_result["zone_diam_rear_cm"]), 2),
                "zone_fwd_cm":      round(float(opt_result["zone_fwd_cm"]), 2),
                "Q_sink_W":         round(float(opt_result["Q_sink_W"]), 3),
                "P_net_W":          round(float(opt_result["P_net_W"]), 3),
                "converged":        bool(opt_result["converged"]),
                "constrained":      bool(opt_result["constrained"]),
                "dose_sf":          round(float(opt_result["dose_sf"]), 3),
                "histology":        str(opt_result["tissue"]["label"]),
                "consistency":      str(opt_result["consistency"]["label"]),
                "antenna_axis":     [round(float(v), 4) for v in antenna_axis],
                "das_angle_deg":    round(float(das_angle_deg), 1),
            },
            "asi": {
                "asi":          asi["asi"],
                "risk_label":   asi["risk_label"],
                "hss_score":    asi["hss_score"],
                "ocm_score":    asi["ocm_score"],
                "cc_score":     asi["cc_score"],
                "dra_score":    asi["dra_score"],
                "das_score":    asi["das_score"],
                "max_loss_pct": asi["max_loss_pct"],
                "min_clear_mm": asi["min_clear_mm"],
                "margin_mm":    asi["margin_mm"],
                "das_angle_deg":asi["das_angle_deg"],
                "interpretation":asi["interpretation"],
            },
            "oar_list":     oar_out,
            "hs_summary":   hs_summary,
            "preview_png":  png_b64,
        })
    except Exception as exc:
        return jsonify({"error": str(exc), "trace": traceback.format_exc()}), 500


@app.route("/visualise", methods=["POST"])
def launch_visualisation():
    """
    Launch the full interactive PyVista window in a background thread.
    Non-blocking — returns immediately with status.
    """
    try:
        s = SESSION
        if s["opt_result"] is None:
            return jsonify({"error": "Run /optimise first"}), 400

        def run_vis():
            try:
                centroids = s["centroids"]
                # Build particle systems
                particle_systems = []
                for v, vn in zip(s["vessels"], s["vnames"]):
                    ps = alg.VesselParticleSystem(v, vn, n_particles=80)
                    particle_systems.append(ps)

                alg.phase3_visualise(
                    surface          = s["surface"],
                    vessels          = s["vessels"],
                    vnames           = s["vnames"],
                    tumors           = s["tumors"],
                    centroids        = centroids,
                    sel_idx          = s["sel_idx"],
                    results          = s["ray_results"],
                    opt_result       = s["opt_result"],
                    asi              = s["asi"],
                    oar_list         = s["oar_list"],
                    particle_systems = particle_systems,
                    centroid_dists   = s["centroid_dists"],
                )
            except Exception as exc:
                print(f"  Visualisation error: {exc}")
                traceback.print_exc()

        t = threading.Thread(target=run_vis, daemon=True)
        t.start()

        return jsonify({
            "status": "visualisation_launched",
            "message": "PyVista window opening — check your taskbar.",
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/snapshot", methods=["GET"])
def snapshot():
    """Return latest offscreen PNG of the current scene."""
    highlight = request.args.get("highlight", None)
    hi_idx    = int(highlight) if highlight is not None else SESSION.get("sel_idx")
    png_b64   = render_overview_png(highlight_idx=hi_idx)
    if png_b64:
        return jsonify({"png": png_b64})
    return jsonify({"error": "Could not render snapshot"}), 500


@app.route("/reset", methods=["POST"])
def reset():
    """Clear all session data."""
    global SESSION
    SESSION = {k: ([] if isinstance(v, list) else (None if v is not None else None))
               for k, v in SESSION.items()}
    SESSION["centroid_dists"] = {}
    return jsonify({"status": "reset"})


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  MWA Planning UI Server")
    print("  Open  http://localhost:5050  in your browser")
    print("  Ctrl+C to stop")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)
