#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   UNIFIED MICROWAVE ABLATION TREATMENT PLANNING SYSTEM  —  v1.0            ║
║   Patient-Specific End-to-End Pipeline                                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Author  : Veda Nunna                                                        ║
║  Version : 1.0  (Unified)                                                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  PIPELINE STAGES                                                             ║
║  ──────────────                                                              ║
║  Stage 0 — DICOM/VTK loader + unit normalisation                            ║
║  Stage 1 — Tumor extraction (connected components) + eligibility filter     ║
║  Stage 2 — Tumor metrics: centroid, diameter, depth, vessel distances       ║
║  Stage 3 — Interactive tumor selection (Phase 1 overview + Phase 2 picker)  ║
║  Stage 4 — Histology + consistency input                                    ║
║  Stage 5 — Ray casting: find safest + shortest route (skin→tumor)           ║
║  Stage 6 — OAR identification + heat-sink physics (Gnielinski/Dittus-       ║
║             Boelter for turbulent; Nu=4.36 for laminar)                     ║
║  Stage 7 — REGIME DECISION (three-tier waterfall):                          ║
║             Tier A → Table-based: does any standard regime cover the        ║
║                      required ablation zone with ≤10% heat-sink loss?       ║
║                      If YES → use table regime (safest, fastest).           ║
║             Tier B → Physics-compensated: compute net power after heat-     ║
║                      sink loss; find table regime or custom (P,t) that      ║
║                      covers the required zone with OAR margin ≥5 mm.        ║
║             Tier C → Directional ablation (OPTIONAL): triggered only if     ║
║                      Tier A and Tier B both fail OAR clearance. Antenna     ║
║                      axis is optimised so the rear null faces the closest   ║
║                      OAR. D-shaped zone covers forward side, protects OAR  ║
║                      on the rear. Uses cos²(θ/2) SAR model (Lee 2023,       ║
║                      Fallahi & Prakash 2018).                               ║
║  Stage 8 — ASI v11: five sub-scores (HSS, OCM, CC, DRA, DAS if Tier C)    ║
║  Stage 9 — Animated 3D visualisation (Phase 3)                             ║
║                                                                             ║
║  THEORETICAL BASIS                                                           ║
║  ─────────────────                                                           ║
║  Heat transfer : Gnielinski (2300≤Re≤10⁶), Dittus-Boelter (Re>10⁴),       ║
║                  Nu=4.36 laminar; wall-layer correction (Sieder-Tate 1936) ║
║  Bioheat       : Pennes 1948 — analytical steady-state radius               ║
║  Directional   : Lee (2023), Fallahi & Prakash (2018) — cos²(θ/2) SAR     ║
║  Cell death    : 60 °C isotherm (Audigier et al. 2020)                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  FILE PATHS  — edit to match your machine
# ══════════════════════════════════════════════════════════════════════════════

DATASET_BASE     = r"C:\Users\z005562w\OneDrive - Siemens Healthineers\Veda\Project\siemens project\3d simulation\Nunna Algo\Nunna Algo\Dataset"
PORTALVENOUS_DIR = os.path.join(DATASET_BASE, "portalvenous")

TUMOR_VTK   = os.path.join(DATASET_BASE, "908ac523data00007_leasion_out.vtk")
SURFACE_VTK = os.path.join(DATASET_BASE, "908ac523data00013_skin_out.vtk")

VESSEL_VTK_LIST = [
    os.path.join(PORTALVENOUS_DIR, "908ac52300001.vtk"),   # portal vein
    os.path.join(PORTALVENOUS_DIR, "908ac52300002.vtk"),   # hepatic vein
    os.path.join(PORTALVENOUS_DIR, "908ac52300003.vtk"),   # aorta
    os.path.join(PORTALVENOUS_DIR, "908ac52300004.vtk"),   # ivc
    os.path.join(PORTALVENOUS_DIR, "908ac52300005.vtk"),   # hepatic artery
]
VESSEL_NAMES = ["portal_vein", "hepatic_vein", "aorta", "ivc", "hepatic_artery"]


# ══════════════════════════════════════════════════════════════════════════════
#  COLOUR MAPS
# ══════════════════════════════════════════════════════════════════════════════

VESSEL_COLOR_MAP = {
    "aorta":          "#FF0000",
    "portal_vein":    "#1565C0",
    "hepatic_vein":   "#1E90FF",
    "ivc":            "#1E90FF",
    "hepatic_artery": "orange",
}
TUMOR_COLORS = ["yellow", "orange", "purple", "pink", "red", "lime",
                "gold", "cyan", "salmon", "chartreuse"]


# ══════════════════════════════════════════════════════════════════════════════
#  PHYSICAL CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

RHO_B   = 1060.0    # blood density  kg/m³
MU_B    = 3.5e-3    # dynamic viscosity  Pa·s
C_B     = 3700.0    # blood specific heat  J/(kg·K)
K_B     = 0.52      # blood thermal conductivity  W/(m·K)
T_BLOOD = 37.0      # °C  body temperature
T_ABL   = 60.0      # °C  cell-death isotherm
T_TISS  = 90.0      # °C  ablation max visualisation

ALPHA_TISSUE    = 70.0    # tissue thermal attenuation  1/m (microwave)
L_SEG           = 0.01    # vessel contact segment length  m
OAR_MIN_CLEAR_M = 5e-3    # 5 mm minimum OAR wall clearance

MIN_DIAMETER_CM = 3.0
MAX_DIAMETER_CM = 5.0
MAX_DEPTH_CM    = 26.0

# Heat-sink loss threshold below which the table regime is accepted
HS_LOSS_ACCEPT_PCT = 10.0   # Tier A: if max vessel loss ≤ 10% → use table

VESSEL_DIAMETERS = {
    "portal_vein":   12e-3,
    "hepatic_vein":   8e-3,
    "aorta":         25e-3,
    "ivc":           20e-3,
    "hepatic_artery": 4.5e-3,
}
VESSEL_VELOCITIES = {
    "portal_vein":   0.15,
    "hepatic_vein":  0.20,
    "aorta":         0.40,
    "ivc":           0.35,
    "hepatic_artery":0.30,
}
VESSEL_RADII = {vn: d / 2.0 for vn, d in VESSEL_DIAMETERS.items()}


# ══════════════════════════════════════════════════════════════════════════════
#  MANUFACTURER ABLATION TABLE  (power_W, time_s, vol_cc, fwd_cm, diam_cm)
# ══════════════════════════════════════════════════════════════════════════════

ABLATION_TABLE = [
    (30,  180, 2.20, 1.9, 2.3),  (30,  300, 2.50, 2.4, 2.7),
    (30,  480, 4.90, 2.9, 3.0),  (30,  600, 5.47, 3.1, 3.1),
    (60,  180, 2.80, 2.5, 2.8),  (60,  300, 4.70, 3.0, 3.3),
    (60,  480, 6.33, 3.8, 3.8),  (60,  600, 5.82, 3.9, 3.9),
    (90,  180, 3.80, 3.1, 3.3),  (90,  300, 5.20, 3.7, 3.8),
    (90,  480, 5.20, 4.2, 4.6),  (90,  600, 6.30, 4.6, 4.9),
    (80,  300, 3.40, 4.2, 3.8),  (80,  600, 8.40, 5.2, 4.4),
    (80,  300, 4.80, 4.5, 3.6),  (80,  600, 9.20, 5.1, 4.6),
    (120, 300, 8.00, 5.1, 4.3),  (120, 600, 9.40, 5.6, 5.0),
    (120, 300, 6.40, 5.2, 3.9),  (140, 600, 8.82, 6.0, 5.0),
    (120, 600, 9.70, 5.9, 5.1),  (160, 300, 6.90, 5.8, 4.2),
    (160, 300, 7.40, 5.4, 4.4),  (160, 300, 6.70, 4.9, 4.5),
    (160, 600, 7.20, 6.3, 5.6),  (160, 600, 10.20, 5.9, 5.8),
    (160, 600, 10.30, 6.1, 5.8),
]


# ══════════════════════════════════════════════════════════════════════════════
#  TUMOR BIOLOGY LIBRARY
# ══════════════════════════════════════════════════════════════════════════════

TUMOR_TYPES = {
    "HCC": {
        "label":       "Hepatocellular Carcinoma (HCC)",
        "k_tissue":    0.52, "rho_cp": 3.6e6, "omega_b": 0.0064,
        "epsilon_r":   43.0, "sigma": 1.69, "k_factor": 1.00,
        "description": "Hypervascular; standard MWA response",
    },
    "COLORECTAL": {
        "label":       "Colorectal Liver Metastasis",
        "k_tissue":    0.48, "rho_cp": 3.8e6, "omega_b": 0.0030,
        "epsilon_r":   39.5, "sigma": 1.55, "k_factor": 1.12,
        "description": "Hypovascular, denser; requires ~12% more energy",
    },
    "NEUROENDOCRINE": {
        "label":       "Neuroendocrine Tumour Metastasis",
        "k_tissue":    0.55, "rho_cp": 3.5e6, "omega_b": 0.0090,
        "epsilon_r":   45.0, "sigma": 1.75, "k_factor": 0.93,
        "description": "Highly vascular; slight dose reduction possible",
    },
    "CHOLANGIO": {
        "label":       "Cholangiocarcinoma / Biliary Origin",
        "k_tissue":    0.44, "rho_cp": 4.0e6, "omega_b": 0.0020,
        "epsilon_r":   37.0, "sigma": 1.40, "k_factor": 1.22,
        "description": "Fibrotic, low conductivity; needs ~22% more energy",
    },
    "FATTY_BACKGROUND": {
        "label":       "Tumour in Fatty/Cirrhotic Liver",
        "k_tissue":    0.38, "rho_cp": 3.2e6, "omega_b": 0.0015,
        "epsilon_r":   34.0, "sigma": 1.20, "k_factor": 1.30,
        "description": "Fatty/cirrhotic liver background; zone spreads differently",
    },
    "UNKNOWN": {
        "label":       "Unknown / Not Biopsied",
        "k_tissue":    0.50, "rho_cp": 3.7e6, "omega_b": 0.0050,
        "epsilon_r":   41.0, "sigma": 1.60, "k_factor": 1.10,
        "description": "Conservative estimate",
    },
}

CONSISTENCY_FACTORS = {
    "soft": {"label": "Soft  (necrotic core, cystic, well-vascularised)",
             "dose_factor": 0.90, "note": "10% dose reduction"},
    "firm": {"label": "Firm  (solid, typical)",
             "dose_factor": 1.00, "note": "Standard dose"},
    "hard": {"label": "Hard  (fibrotic, desmoplastic, calcified)",
             "dose_factor": 1.20, "note": "20% dose increase required"},
}


# ══════════════════════════════════════════════════════════════════════════════
#  DIRECTIONAL ANTENNA CONSTANTS  (Fallahi & Prakash 2018, Lee 2023)
# ══════════════════════════════════════════════════════════════════════════════

G_FORWARD      = 1.80    # forward hemisphere gain factor
G_REAR         = 0.20    # rear hemisphere gain factor (80% suppression)
BEAM_TILT_DEG  = 12.0
BEAM_TILT_RAD  = np.radians(BEAM_TILT_DEG)
N_AZ_SEARCH    = 36
N_EL_SEARCH    = 18

# Optimizer bounds
P_MIN_W  = 20.0
P_MAX_W  = 200.0
T_MIN_S  = 60.0
T_MAX_S  = 900.0
MAX_ITER = 60
CONV_TOL = 0.005

ASI_WEIGHTS_STD = {"hss": 0.35, "ocm": 0.30, "cc": 0.20, "dra": 0.15}
ASI_WEIGHTS_DIR = {"hss": 0.30, "ocm": 0.27, "cc": 0.18, "dra": 0.15, "das": 0.10}


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 0 — I/O HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_vtk(path):
    if not os.path.exists(path):
        print(f"  ❌  Missing: {path}")
        return None
    try:
        mesh = pv.read(path)
        print(f"  ✔  {os.path.basename(path)}  ({mesh.n_points} pts, {mesh.n_cells} cells)")
        return mesh
    except Exception as e:
        print(f"  ❌  Error reading {path}: {e}")
        return None


def rescale(mesh):
    """Convert mm → m if bounding box > 1 m (heuristic)."""
    if mesh is None:
        return None
    pts = np.array(mesh.points)
    if np.max(np.abs(pts)) > 1000:
        mesh.points = pts / 1000.0
        print("     ↳ rescaled mm → m")
    return mesh


def smooth_tumor(mesh, n_iter=80, relax=0.1):
    try:
        return mesh.smooth(n_iter=n_iter, relaxation_factor=relax,
                           feature_smoothing=False, boundary_smoothing=True)
    except Exception:
        return mesh


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 1 — TUMOR EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_tumors(tumor_mesh):
    """Split combined VTK into connected components (individual tumors)."""
    conn = tumor_mesh.connectivity()
    bodies = conn.split_bodies()
    print(f"  → {len(bodies)} lesion(s) detected")
    return bodies


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 2 — TUMOR METRICS
# ══════════════════════════════════════════════════════════════════════════════

def tumor_metrics(tumors, surface, vessels, vnames):
    s_tree  = cKDTree(np.array(surface.points))
    v_trees = [cKDTree(np.array(v.points)) for v in vessels]
    metrics = []
    for i, t in enumerate(tumors):
        c   = np.array(t.center)
        b   = t.bounds
        dm  = max(b[1]-b[0], b[3]-b[2], b[5]-b[4])  # bounding-box max extent
        dep = float(s_tree.query(c, k=1)[0])
        vd  = [float(vt.query(c, k=1)[0]) for vt in v_trees]
        elig = (MIN_DIAMETER_CM <= dm*100 <= MAX_DIAMETER_CM
                and dep*100 <= MAX_DEPTH_CM)
        metrics.append({
            "idx": i, "centroid": c,
            "diameter_cm": dm * 100.0, "depth_cm": dep * 100.0,
            "vessel_dists_m": vd, "min_vessel_m": min(vd),
            "closest_vessel": vnames[int(np.argmin(vd))],
            "centroid_dists": {vn: vd[j] for j, vn in enumerate(vnames)},
            "eligible": elig,
        })
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 3 — PHASE 1 OVERVIEW VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def phase1_overview(surface, vessels, vnames, tumors, metrics):
    print("\n" + "="*70)
    print("  PHASE 1 — OVERVIEW  (close window to continue)")
    print("="*70)
    pl = pv.Plotter(window_size=[1400, 900])
    pl.add_mesh(surface, color="bisque", opacity=0.18, label="Skin")
    for i, v in enumerate(vessels):
        pl.add_mesh(v, color=VESSEL_COLOR_MAP.get(vnames[i], "cyan"),
                    opacity=0.70, label=vnames[i])
    for i, (t, m) in enumerate(zip(tumors, metrics)):
        col = TUMOR_COLORS[i % len(TUMOR_COLORS)]
        pl.add_mesh(smooth_tumor(t), color=col, opacity=0.88,
                    label=f"T{i} {m['diameter_cm']:.1f}cm")
        pl.add_point_labels([m["centroid"]], [f"T{i}"],
                            font_size=14, text_color="white",
                            point_color=col, point_size=10,
                            shape_opacity=0.0)
    pl.add_legend(loc="upper right", size=(0.22, 0.32))
    pl.add_text("MWA Planning — Phase 1 Overview\n"
                "Close window to select tumor", position="upper_left",
                font_size=11)
    pl.show_axes()
    pl.show()


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 3 — PHASE 2 TUMOR PICKER + HISTOLOGY INPUT
# ══════════════════════════════════════════════════════════════════════════════

def phase2_pick_tumor(metrics, vnames):
    print("\n" + "="*70)
    print("  PHASE 2 — TUMOR SELECTION")
    print("="*70)
    print(f"\n  {'#':>3}  {'Diam(cm)':>9}  {'Depth(cm)':>10}  "
          f"{'Closest vessel':>18}  {'Dist(mm)':>9}  {'Eligible':>8}")
    print("  " + "-"*65)
    for m in metrics:
        flag = "✔" if m["eligible"] else "✘"
        print(f"  {m['idx']:>3}  {m['diameter_cm']:>9.2f}  "
              f"{m['depth_cm']:>10.2f}  {m['closest_vessel']:>18}  "
              f"{m['min_vessel_m']*1000:>9.1f}  {flag:>8}")

    eligible = [m for m in metrics if m["eligible"]]
    if not eligible:
        print("\n  ⚠  No tumors meet eligibility criteria.")
        print("  → Defaulting to largest tumor for demonstration.")
        eligible = sorted(metrics, key=lambda x: x["diameter_cm"], reverse=True)[:1]

    while True:
        raw = input(f"\n  Enter tumor number to plan [0–{len(metrics)-1}]: ").strip()
        try:
            sel = metrics[int(raw)]
            break
        except (ValueError, IndexError):
            print("  Invalid input, try again.")

    # Histology
    print("\n  HISTOLOGICAL TYPE:")
    for k, v in TUMOR_TYPES.items():
        print(f"    {k:<20}  {v['label']}")
    type_key = "UNKNOWN"
    raw = input("  Enter type key [default UNKNOWN]: ").strip().upper()
    if raw in TUMOR_TYPES:
        type_key = raw

    # Consistency
    print("\n  CONSISTENCY:")
    for k, v in CONSISTENCY_FACTORS.items():
        print(f"    {k:<10}  {v['label']}")
    consist_key = "firm"
    raw = input("  Enter consistency [default firm]: ").strip().lower()
    if raw in CONSISTENCY_FACTORS:
        consist_key = raw

    return sel, type_key, consist_key


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 5 — RAY CASTING: SAFEST + SHORTEST ACCESS ROUTE
# ══════════════════════════════════════════════════════════════════════════════

def generate_rays(n_theta=20, n_phi=40):
    rays = []
    for t in np.linspace(0, np.pi, n_theta):
        for p in np.linspace(0, 2*np.pi, n_phi):
            rays.append(np.array([np.sin(t)*np.cos(p),
                                   np.sin(t)*np.sin(p),
                                   np.cos(t)]))
    return np.array(rays)


def ray_segment_dist(origin, direction, path_d, vessel_pts, vessel_tree,
                     n_samples=60):
    """Minimum distance from the ray segment to any vessel surface point."""
    ts = np.linspace(0, path_d, n_samples)
    pts = origin[None, :] + np.outer(ts, direction)
    dists, _ = vessel_tree.query(pts, k=1)
    return float(np.min(dists))


def cast_rays(centroid, surface, vessels, vnames, n_theta=20, n_phi=40):
    """
    Cast rays from the tumor centroid toward the skin surface.
    Each ray is scored on:
      - path_length_m  (shorter = better, less trauma)
      - min_vessel_dist_m  (larger = safer, less heat-sink)
      - heat_loss_pct  (lower = better)

    Returns a list of dicts, sorted by composite score.
    The best ray (lowest path length, furthest from vessels) is the
    recommended needle access route.
    """
    print("\n  Ray casting …", end="", flush=True)
    rays    = generate_rays(n_theta, n_phi)
    all_pts = np.vstack([np.array(v.points) for v in vessels])
    v_tree  = cKDTree(all_pts)

    results = []
    for direction in rays:
        try:
            pts_hit, _ = surface.ray_trace(centroid, centroid + direction * 0.30)
            if len(pts_hit) == 0:
                continue
            hit   = pts_hit[0]
            pdist = float(np.linalg.norm(hit - centroid))
            # Closest vessel distance along path
            mvd   = ray_segment_dist(centroid, direction, pdist, all_pts, v_tree)

            # Simple heat-sink loss estimate along path (exponential decay from nearest vessel)
            hl_pct = 100.0 * np.exp(-ALPHA_TISSUE * mvd)

            results.append({
                "direction":       direction,
                "hit_point":       hit,
                "path_length_m":   pdist,
                "min_vessel_dist_m": mvd,
                "heat_loss_pct":   hl_pct,
            })
        except Exception:
            continue

    # Composite score — lower is better:
    #   normalise path length (0–1) + normalise inverse vessel dist (0–1)
    if results:
        lens  = np.array([r["path_length_m"]   for r in results])
        mvds  = np.array([r["min_vessel_dist_m"] for r in results])
        ln    = (lens  - lens.min())  / max(lens.max()  - lens.min(),  1e-9)
        mvdn  = (mvds.max() - mvds)   / max(mvds.max()  - mvds.min(),  1e-9)  # inverted
        scores = 0.45 * ln + 0.55 * mvdn
        for i, r in enumerate(results):
            r["composite_score"] = float(scores[i])
        results.sort(key=lambda x: x["composite_score"])

    print(f" {len(results)} valid rays found.")
    return results


def pick_needle_direction(ray_results, n_candidates=3):
    """Return top-N candidate needle directions with summary."""
    print("\n  TOP NEEDLE ACCESS ROUTES:")
    print(f"  {'#':>3}  {'Path(mm)':>9}  {'VesselClear(mm)':>17}  "
          f"{'HeatLoss%':>11}  {'Score':>7}")
    print("  " + "-"*55)
    for i, r in enumerate(ray_results[:n_candidates]):
        print(f"  {i+1:>3}  {r['path_length_m']*1000:>9.1f}  "
              f"{r['min_vessel_dist_m']*1000:>17.1f}  "
              f"{r['heat_loss_pct']:>11.2f}  "
              f"{r['composite_score']:>7.4f}")
    best = ray_results[0]
    print(f"\n  ✔ Recommended: path={best['path_length_m']*1000:.1f} mm, "
          f"vessel_clear={best['min_vessel_dist_m']*1000:.1f} mm")
    return best["direction"], best


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 6 — HEAT-SINK PHYSICS
# ══════════════════════════════════════════════════════════════════════════════

def nusselt_full(Re, Pr):
    if Re < 2300:
        return 4.36
    f  = (0.790 * np.log(Re) - 1.64) ** (-2)
    Nu = (f/8) * (Re - 1000) * Pr / (1.0 + 12.7 * np.sqrt(f/8) * (Pr**(2/3) - 1))
    if Re >= 10000:
        Nu = 0.023 * Re**0.8 * Pr**0.4
    return max(Nu, 4.36)


def wall_layer_correction(Re, D):
    eta = 1.0 + 0.0015 * np.sqrt(max(Re, 0)) * 0.5 / max(D * 1000, 1.0)
    return float(np.clip(eta, 0.90, 1.20))


def heat_sink_physics(distance_m, vessel_name, power_w, time_s,
                       sar_weight=1.0):
    """
    Full Gnielinski/Dittus-Boelter heat-sink model.
    sar_weight < 1 reduces effective heat removal (directional mode).
    """
    D  = VESSEL_DIAMETERS[vessel_name]
    u  = VESSEL_VELOCITIES[vessel_name]
    Re = (RHO_B * u * D) / MU_B
    Pr = (C_B * MU_B) / K_B
    Nu = nusselt_full(Re, Pr)
    eta = wall_layer_correction(Re, D)
    hb  = (Nu * K_B) / D
    hw  = hb * eta
    Ac  = (D/2) * (np.pi/3) * L_SEG
    Af  = np.pi * D * L_SEG
    dTw = max(T_TISS - T_BLOOD, 0.1)
    dTb = max((T_TISS + T_BLOOD)/2 - T_BLOOD, 0.1)
    turb_mix = 0.30 if Re >= 2300 else 0.05
    Q_wall  = hw * Ac * dTw
    Q_bulk  = turb_mix * hb * Af * dTb
    Qv = min((Q_wall + Q_bulk) * sar_weight, power_w)
    d  = max(distance_m, 1e-4)
    Ql = min(Qv * np.exp(-ALPHA_TISSUE * d), power_w)
    Ei = power_w * time_s
    El = min(Ql * time_s, Ei)
    flow_regime = "Turbulent" if Re >= 2300 else "Laminar"
    return {
        "vessel": vessel_name, "dist_mm": d * 1000,
        "Re": Re, "Pr": Pr, "Nu": Nu, "flow_regime": flow_regime,
        "Q_loss_W": Ql, "E_loss_J": El,
        "loss_pct": 100.0 * El / max(Ei, 1e-9),
        "Q_wall_W": Q_wall, "Q_bulk_W": Q_bulk,
        "sar_weight": sar_weight,
    }


def total_heat_sink(centroid_dists, vnames, power_w, time_s, sar_weights=None):
    """Sum heat-sink across all vessels. Returns (Q_total_W, per_vessel_dict)."""
    if sar_weights is None:
        sar_weights = {vn: 1.0 for vn in vnames}
    per = {}
    for vn in vnames:
        per[vn] = heat_sink_physics(centroid_dists[vn], vn, power_w, time_s,
                                     sar_weights.get(vn, 1.0))
    return sum(hs["Q_loss_W"] for hs in per.values()), per


# ══════════════════════════════════════════════════════════════════════════════
#  DIRECTIONAL ANTENNA HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def directional_sar_weight(direction, antenna_axis):
    """cos²(θ/2) for forward, sin²(θ/2) for rear  (Lee 2023 Eq.1)."""
    a_norm = np.linalg.norm(antenna_axis)
    d_norm = np.linalg.norm(direction)
    if a_norm < 1e-9 or d_norm < 1e-9:
        return 1.0
    cos_th = np.clip(np.dot(direction/d_norm, antenna_axis/a_norm), -1, 1)
    theta  = np.arccos(cos_th)
    if theta <= np.pi/2:
        return float(G_FORWARD * np.cos(theta/2)**2)
    else:
        return float(G_REAR * np.sin(theta/2)**2)


def find_optimal_antenna_axis(centroid, centroid_dists, vnames, vessels):
    """
    Grid search over (azimuth × elevation) to find the antenna axis that
    maximises OAR protection (null toward nearest OAR) while maintaining
    forward coverage toward the ablation zone.
    """
    oar_name = min(centroid_dists, key=centroid_dists.get)
    oar_vtx  = np.array(vessels[vnames.index(oar_name)].points)
    oar_dir  = oar_vtx.mean(axis=0) - centroid
    d        = np.linalg.norm(oar_dir)
    oar_unit = oar_dir / max(d, 1e-9)

    best_axis  = np.array([0.0, 0.0, 1.0])
    best_score = -np.inf

    for az in np.linspace(0, 2*np.pi, N_AZ_SEARCH, endpoint=False):
        for el in np.linspace(-np.pi/2, np.pi/2, N_EL_SEARCH):
            axis = np.array([np.cos(el)*np.cos(az),
                              np.cos(el)*np.sin(az),
                              np.sin(el)])
            # null = -axis; score = alignment of null with OAR direction
            null_align  = float(np.dot(-axis, oar_unit))
            fwd_penalty = float(max(0.0, np.dot(axis, oar_unit)))
            score = null_align - 2.0 * fwd_penalty
            if score > best_score:
                best_score = score
                best_axis  = axis.copy()

    return best_axis, oar_name


def sar_weights_directional(centroid, vessels, vnames, antenna_axis):
    """Compute per-vessel SAR weight based on directional antenna model."""
    weights = {}
    for vn in vnames:
        pts = np.array(vessels[vnames.index(vn)].points)
        d   = pts.mean(axis=0) - centroid
        weights[vn] = directional_sar_weight(d, antenna_axis)
    return weights


# ══════════════════════════════════════════════════════════════════════════════
#  BIOPHYSICAL RADIUS (Pennes 1948, analytical steady-state)
# ══════════════════════════════════════════════════════════════════════════════

def pennes_radius(P_net_w, time_s, tissue, sar_w=1.0):
    kt    = tissue["k_tissue"]
    omega = tissue["omega_b"]
    gamma = np.sqrt(omega * RHO_B * C_B / max(kt, 1e-9))
    tau   = tissue["rho_cp"] / max(omega * RHO_B * C_B, 1e-6)
    eff   = 1.0 - np.exp(-time_s / max(tau, 1e-3))
    Peff  = max(P_net_w * eff * sar_w, 0.1)
    denom = 4 * np.pi * kt * (T_ABL - T_BLOOD) * max(gamma, 1e-3)
    return float(np.clip(np.sqrt(max(Peff / denom, 1e-6)), 0.005, 0.100))


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 7 — THREE-TIER REGIME DECISION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _oar_clearance(zone_r_m, centroid_dists, vnames):
    """Symmetric zone wall clearance."""
    return {vn: centroid_dists[vn] - VESSEL_RADII.get(vn, 0) - zone_r_m
            for vn in vnames}


def tier_a_table(tumor_diam_cm, centroid_dists, vnames, margin_cm=0.5):
    """
    Tier A — Pure manufacturer table lookup.
    Accepted only if the regime covers the required zone AND the
    maximum heat-sink loss from all vessels is ≤ HS_LOSS_ACCEPT_PCT.

    Returns (result_dict, accepted: bool)
    """
    req = tumor_diam_cm + margin_cm
    # Find smallest diameter in table that covers requirement
    cands = sorted(
        [(P,t,vol,fwd,diam) for P,t,vol,fwd,diam in ABLATION_TABLE if diam >= req],
        key=lambda r: (r[4], r[0], r[1])
    )
    if not cands:
        return None, False
    P_rec, t_rec, vol_rec, fwd_rec, diam_rec = cands[0]

    per_hs = {vn: heat_sink_physics(centroid_dists[vn], vn, P_rec, t_rec)
              for vn in vnames}
    max_loss = max(hs["loss_pct"] for hs in per_hs.values())
    zone_r   = (diam_rec / 2.0) / 100.0
    clr      = _oar_clearance(zone_r, centroid_dists, vnames)
    min_cl   = min(clr.values())
    accepted = (max_loss <= HS_LOSS_ACCEPT_PCT) and (min_cl >= OAR_MIN_CLEAR_M)

    result = {
        "tier": "A", "tier_label": "Table-Based (No Heat-Sink Compensation)",
        "P_opt": P_rec, "t_opt": t_rec,
        "zone_diam_cm": diam_rec, "zone_fwd_cm": fwd_rec,
        "per_vessel_hs": per_hs, "clearance_report":
            [{"vessel": vn, "wall_clear_mm": v*1000} for vn, v in clr.items()],
        "min_clear_mm": min_cl * 1000,
        "max_loss_pct": max_loss,
        "constrained": not accepted, "converged": True,
        "Q_sink_W": sum(hs["Q_loss_W"] for hs in per_hs.values()),
        "P_net_W": P_rec,
        "margin_cm": margin_cm, "dose_sf": 1.0,
        "directional": False,
    }
    return result, accepted


def tier_b_physics(tumor_diam_cm, tissue_key, consist_key,
                    centroid_dists, vnames, margin_cm=0.5):
    """
    Tier B — Physics-compensated regime.
    Iterates over table entries; for each, subtracts heat-sink loss to get
    P_net, computes Pennes radius, and checks OAR clearance.
    Selects the regime with best (zone coverage + OAR safety) at minimum energy.
    If no table entry works, constructs a custom (P, t) by iterative escalation.

    Returns (result_dict, accepted: bool)
    """
    tissue   = TUMOR_TYPES[tissue_key]
    consist  = CONSISTENCY_FACTORS[consist_key]
    dose_sf  = tissue["k_factor"] * consist["dose_factor"]
    req_r_m  = ((tumor_diam_cm + margin_cm) / 2.0) / 100.0

    # ── Pass 1: scan table entries
    best = None; best_cost = np.inf
    for P, t, vol, fwd, diam_tab in ABLATION_TABLE:
        Q_total, per_hs = total_heat_sink(centroid_dists, vnames, P, t)
        P_net    = max(P - Q_total, 0.5)
        r_abl    = pennes_radius(P_net, t, tissue) * dose_sf
        zone_diam = r_abl * 2.0 * 100.0
        zone_r    = r_abl
        clr       = _oar_clearance(zone_r, centroid_dists, vnames)
        min_cl    = min(clr.values())
        oar_ok    = min_cl >= OAR_MIN_CLEAR_M
        covers    = zone_diam >= (tumor_diam_cm + margin_cm)

        if covers and oar_ok:
            cost = (P * t) / (160 * 900)
        elif covers:
            cost = 0.5 + abs(min_cl) / OAR_MIN_CLEAR_M
        else:
            cost = 1.0 + (tumor_diam_cm + margin_cm - zone_diam)

        if cost < best_cost:
            best_cost = cost
            best = (P, t, vol, fwd, zone_diam, P_net, Q_total,
                    per_hs, clr, min_cl, oar_ok, covers)

    if best is None:
        return None, False

    P_rec, t_rec, vol_rec, fwd_rec, diam_rec, Pnet_rec, Q_rec, \
        per_hs_rec, clr_rec, min_cl_rec, oar_ok_rec, covers_rec = best

    # ── Pass 2: if best table entry doesn't satisfy both, try iterative escalation
    if not (covers_rec and oar_ok_rec):
        P_cur = P_MIN_W; t_cur = 300.0
        for _ in range(MAX_ITER):
            Q_total, per_hs = total_heat_sink(centroid_dists, vnames, P_cur, t_cur)
            P_net   = max(P_cur - Q_total, 0.5)
            r_abl   = pennes_radius(P_net, t_cur, tissue) * dose_sf
            zone_diam = r_abl * 2.0 * 100.0
            clr     = _oar_clearance(r_abl, centroid_dists, vnames)
            min_cl  = min(clr.values())
            covers  = zone_diam >= (tumor_diam_cm + margin_cm)
            oar_ok  = min_cl >= OAR_MIN_CLEAR_M
            if covers and oar_ok:
                P_rec, t_rec, diam_rec, fwd_rec = P_cur, t_cur, zone_diam, zone_diam*0.55
                Pnet_rec, Q_rec = P_net, Q_total
                per_hs_rec, clr_rec, min_cl_rec = per_hs, clr, min_cl
                oar_ok_rec = True; covers_rec = True
                break
            if not covers:
                if P_cur < P_MAX_W:
                    P_cur = min(P_cur + 5.0, P_MAX_W)
                else:
                    t_cur = min(t_cur + 30.0, T_MAX_S)
                    if t_cur >= T_MAX_S:
                        break
            elif not oar_ok:
                break   # coverage OK but OAR issue → must go to Tier C

    accepted = covers_rec and oar_ok_rec
    result = {
        "tier": "B", "tier_label": "Physics-Compensated (Heat-Sink Corrected)",
        "P_opt": P_rec, "t_opt": t_rec,
        "zone_diam_cm": diam_rec, "zone_fwd_cm": fwd_rec,
        "per_vessel_hs": per_hs_rec,
        "clearance_report":
            [{"vessel": vn, "wall_clear_mm": v*1000} for vn, v in clr_rec.items()],
        "min_clear_mm": min_cl_rec * 1000,
        "max_loss_pct": max(hs["loss_pct"] for hs in per_hs_rec.values()),
        "constrained": not accepted, "converged": covers_rec,
        "Q_sink_W": Q_rec, "P_net_W": Pnet_rec,
        "margin_cm": margin_cm, "dose_sf": dose_sf,
        "directional": False,
    }
    return result, accepted


def tier_c_directional(tumor_diam_cm, tissue_key, consist_key,
                        centroid_dists, vnames, vessels, centroid,
                        margin_cm=0.5):
    """
    Tier C — Directional MWA (OPTIONAL, only if Tier A and Tier B fail).
    Finds optimal antenna axis (null toward nearest OAR), then runs the
    directional biophysical optimizer with cos²(θ/2) SAR weighting.

    Returns result_dict.
    """
    tissue   = TUMOR_TYPES[tissue_key]
    consist  = CONSISTENCY_FACTORS[consist_key]
    dose_sf  = tissue["k_factor"] * consist["dose_factor"]
    req_r_m  = ((tumor_diam_cm + margin_cm) / 2.0) / 100.0

    # Find optimal antenna axis
    antenna_axis, oar_name = find_optimal_antenna_axis(
        centroid, centroid_dists, vnames, vessels)
    sar_w = sar_weights_directional(centroid, vessels, vnames, antenna_axis)

    print(f"\n  Directional mode: null pointed toward [{oar_name}]")
    print(f"  Antenna axis: ({antenna_axis[0]:.3f}, {antenna_axis[1]:.3f}, {antenna_axis[2]:.3f})")

    # Iterative optimizer with directional weighting
    P_cur = P_MIN_W; t_cur = 300.0
    converged = False; constrained = False
    per_hs_final = {}

    print(f"\n  {'Iter':>4}  {'P(W)':>7}  {'t(s)':>6}  {'Q_sink(W)':>10}  "
          f"{'P_net(W)':>9}  {'r_fwd(cm)':>10}  {'Status'}")
    print("  " + "-"*70)

    for it in range(1, MAX_ITER + 1):
        Q_total, per_hs = total_heat_sink(centroid_dists, vnames, P_cur, t_cur, sar_w)
        P_net   = max(P_cur - Q_total, 0.5)
        r_fwd   = pennes_radius(P_net, t_cur, tissue, sar_w=G_FORWARD) * dose_sf
        r_rear  = pennes_radius(P_net, t_cur, tissue, sar_w=G_REAR)    * dose_sf

        # D-shaped OAR clearance: forward zone vs rear zone per vessel
        clr = {}
        for vn in vnames:
            vpts = np.array(vessels[vnames.index(vn)].points)
            v_dir = vpts.mean(axis=0) - centroid
            w = directional_sar_weight(v_dir, antenna_axis)
            zone_r = r_fwd if w >= 0.5 else r_rear
            clr[vn] = centroid_dists[vn] - VESSEL_RADII.get(vn, 0) - zone_r

        min_cl  = min(clr.values())
        oar_ok  = min_cl >= OAR_MIN_CLEAR_M
        covers  = r_fwd >= req_r_m

        status = ""
        if covers and oar_ok:
            status = "✔ CONVERGED"; converged = True; per_hs_final = per_hs; break
        elif covers and not oar_ok:
            status = "⚠ OAR ENCROACH"; constrained = True; per_hs_final = per_hs; break
        elif P_cur >= P_MAX_W:
            t_cur = min(t_cur + 30.0, T_MAX_S)
            if t_cur >= T_MAX_S:
                constrained = True; per_hs_final = per_hs; break
            status = f"↑ time"
        else:
            P_cur = min(P_cur + 5.0, P_MAX_W)
            status = "↑ power"
        per_hs_final = per_hs
        if it % 5 == 0 or it == 1:
            print(f"  {it:>4}  {P_cur:>7.1f}  {t_cur:>6.0f}  "
                  f"{Q_total:>10.3f}  {P_net:>9.3f}  "
                  f"{r_fwd*100:>10.3f}  {status}")

    Q_f, per_hs_final = total_heat_sink(centroid_dists, vnames, P_cur, t_cur, sar_w)
    P_net_f = max(P_cur - Q_f, 0.5)
    r_fwd_f = pennes_radius(P_net_f, t_cur, tissue, sar_w=G_FORWARD) * dose_sf
    r_rear_f= pennes_radius(P_net_f, t_cur, tissue, sar_w=G_REAR)    * dose_sf
    clr_f   = {}
    for vn in vnames:
        vpts = np.array(vessels[vnames.index(vn)].points)
        v_dir = vpts.mean(axis=0) - centroid
        w = directional_sar_weight(v_dir, antenna_axis)
        zone_r = r_fwd_f if w >= 0.5 else r_rear_f
        clr_f[vn] = centroid_dists[vn] - VESSEL_RADII.get(vn, 0) - zone_r

    return {
        "tier": "C", "tier_label": "Directional MWA (OAR-Protected, Antenna Null Aligned)",
        "P_opt": P_cur, "t_opt": t_cur,
        "zone_diam_cm": r_fwd_f * 2.0 * 100.0,
        "zone_diam_fwd_cm": r_fwd_f * 2.0 * 100.0,
        "zone_diam_rear_cm": r_rear_f * 2.0 * 100.0,
        "zone_fwd_cm": r_fwd_f * 2.0 * 100.0 * 1.25,
        "per_vessel_hs": per_hs_final,
        "clearance_report":
            [{"vessel": vn, "wall_clear_mm": v*1000} for vn, v in clr_f.items()],
        "min_clear_mm": min(clr_f.values()) * 1000,
        "max_loss_pct": max(hs["loss_pct"] for hs in per_hs_final.values()),
        "constrained": constrained, "converged": converged,
        "Q_sink_W": Q_f, "P_net_W": P_net_f,
        "margin_cm": margin_cm, "dose_sf": dose_sf,
        "directional": True,
        "antenna_axis": antenna_axis,
        "oar_name": oar_name,
        "G_forward": G_FORWARD, "G_rear": G_REAR,
        "sar_weights": sar_w,
        "r_fwd_m": r_fwd_f, "r_rear_m": r_rear_f,
    }


def run_regime_decision(tumor_diam_cm, tissue_key, consist_key,
                         centroid_dists, vnames, vessels, centroid,
                         margin_cm=0.5):
    """
    Three-tier waterfall:
      Tier A → Table (if low heat-sink loss)
      Tier B → Physics-compensated (if Tier A fails)
      Tier C → Directional (only if both Tier A and B fail OAR clearance)

    Returns final result dict and which tier was selected.
    """
    print("\n" + "╔" + "═"*68 + "╗")
    print("║  STAGE 7 — REGIME DECISION ENGINE                                   ║")
    print("╚" + "═"*68 + "╝")
    print(f"\n  Tumor: {tumor_diam_cm:.2f} cm  |  Required zone: {tumor_diam_cm+margin_cm:.2f} cm")
    print(f"  Histology: {TUMOR_TYPES[tissue_key]['label']}")
    print(f"  Consistency: {CONSISTENCY_FACTORS[consist_key]['label']}")

    # ── Tier A
    print("\n  ── TIER A: Table-Based Regime ──")
    res_a, ok_a = tier_a_table(tumor_diam_cm, centroid_dists, vnames, margin_cm)
    if ok_a:
        print(f"  ✔ TIER A ACCEPTED: {res_a['P_opt']:.0f} W × {res_a['t_opt']:.0f} s  "
              f"| zone={res_a['zone_diam_cm']:.2f} cm  "
              f"| max_loss={res_a['max_loss_pct']:.1f}%  "
              f"| min_clear={res_a['min_clear_mm']:.1f} mm")
        return res_a, "A"
    else:
        reason = []
        if res_a:
            if res_a["max_loss_pct"] > HS_LOSS_ACCEPT_PCT:
                reason.append(f"heat-sink loss {res_a['max_loss_pct']:.1f}% > {HS_LOSS_ACCEPT_PCT}%")
            if res_a["min_clear_mm"] < OAR_MIN_CLEAR_M * 1000:
                reason.append(f"OAR clearance {res_a['min_clear_mm']:.1f} mm < 5 mm")
        print(f"  ✘ Tier A rejected: {'; '.join(reason) if reason else 'no table entry sufficient'}")

    # ── Tier B
    print("\n  ── TIER B: Physics-Compensated Regime ──")
    res_b, ok_b = tier_b_physics(tumor_diam_cm, tissue_key, consist_key,
                                   centroid_dists, vnames, margin_cm)
    if ok_b:
        print(f"  ✔ TIER B ACCEPTED: {res_b['P_opt']:.0f} W × {res_b['t_opt']:.0f} s  "
              f"| zone={res_b['zone_diam_cm']:.2f} cm  "
              f"| P_net={res_b['P_net_W']:.1f} W  "
              f"| min_clear={res_b['min_clear_mm']:.1f} mm")
        return res_b, "B"
    else:
        print(f"  ✘ Tier B rejected: OAR clearance insufficient "
              f"({res_b['min_clear_mm']:.1f} mm < 5 mm)" if res_b else
              "  ✘ Tier B rejected: optimizer failed to converge")

    # ── Tier C (directional — optional, triggered only if A and B both fail)
    print("\n  ── TIER C: Directional MWA (OAR-Protected) ──")
    print("  ⚡ Classic omnidirectional ablation cannot safely clear OAR.")
    print("     Initiating directional antenna optimisation …")
    res_c = tier_c_directional(tumor_diam_cm, tissue_key, consist_key,
                                centroid_dists, vnames, vessels, centroid,
                                margin_cm)
    tier_label = "C"
    if res_c["converged"]:
        print(f"  ✔ TIER C CONVERGED: {res_c['P_opt']:.0f} W × {res_c['t_opt']:.0f} s  "
              f"| fwd_zone={res_c['zone_diam_fwd_cm']:.2f} cm  "
              f"| rear_zone={res_c['zone_diam_rear_cm']:.2f} cm  "
              f"| min_clear={res_c['min_clear_mm']:.1f} mm")
    else:
        print(f"  ⚠ TIER C CONSTRAINED (best effort): "
              f"{res_c['P_opt']:.0f} W × {res_c['t_opt']:.0f} s  "
              f"| min_clear={res_c['min_clear_mm']:.1f} mm")

    return res_c, tier_label


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 8 — ASI v11
# ══════════════════════════════════════════════════════════════════════════════

def compute_asi(opt_result, tumor_diam_cm, ray_losses=None):
    per_hs  = opt_result["per_vessel_hs"]
    cr      = opt_result["clearance_report"]
    zone    = opt_result["zone_diam_cm"]
    const   = opt_result["constrained"]
    directional = opt_result.get("directional", False)
    weights = ASI_WEIGHTS_DIR if directional else ASI_WEIGHTS_STD

    max_loss  = max(hs["loss_pct"] for hs in per_hs.values())
    hss_score = float(np.clip(100.0 * (1.0 - max_loss / 50.0), 0, 100))

    min_cl_mm = min(c["wall_clear_mm"] for c in cr) if cr else 20.0
    ocm_score = float(np.clip(100.0 * min_cl_mm / 20.0, 0, 100))

    margin_mm = (zone - tumor_diam_cm) * 10.0
    cc_score  = float(np.clip(100.0 * margin_mm / 10.0, 0, 100))
    if const:
        cc_score *= 0.60

    if ray_losses and len(ray_losses) > 1:
        spread    = float(np.max(ray_losses) - np.min(ray_losses))
        dra_score = float(np.clip(100.0 * (1.0 - spread / 30.0), 0, 100))
    else:
        dra_score = 50.0

    components = {
        "hss": hss_score * weights["hss"],
        "ocm": ocm_score * weights["ocm"],
        "cc":  cc_score  * weights["cc"],
        "dra": dra_score * weights["dra"],
    }

    das_score = 0.0
    if directional and "antenna_axis" in opt_result:
        # DAS = how well null aligns with nearest OAR direction
        oar_name = opt_result.get("oar_name", "")
        if oar_name:
            das_score = float(np.clip(
                100.0 * (1.0 - np.dot(-opt_result["antenna_axis"],
                                       np.array([0, 0, 1]))**2),
                0, 100))
        components["das"] = das_score * weights.get("das", 0.0)

    asi = sum(components.values())
    risk = ("LOW" if asi >= 75 else "MODERATE" if asi >= 50
            else "HIGH" if asi >= 30 else "CRITICAL")

    return {
        "asi": round(asi, 1), "risk_label": risk,
        "hss_score": round(hss_score, 1),
        "ocm_score": round(ocm_score, 1),
        "cc_score":  round(cc_score, 1),
        "dra_score": round(dra_score, 1),
        "das_score": round(das_score, 1),
        "max_loss_pct": round(max_loss, 2),
        "min_clear_mm": round(min_cl_mm, 1),
        "margin_mm":    round(margin_mm, 1),
        "spread_pct":   round(float(np.max(ray_losses)-np.min(ray_losses))
                              if ray_losses and len(ray_losses) > 1 else 0.0, 2),
        "directional": directional,
        "tier": opt_result.get("tier", "?"),
    }


def print_asi(asi, opt_result):
    print("\n" + "╔" + "═"*68 + "╗")
    print("║  ABLATION SAFETY INDEX (ASI)                                        ║")
    print("╠" + "═"*68 + "╣")
    bar_n = int(asi["asi"] / 2)
    bar   = "█" * bar_n + "░" * (50 - bar_n)
    print(f"║  ASI = {asi['asi']:5.1f}/100  [{asi['risk_label']:<8}]  {bar} ║")
    print("╠" + "═"*68 + "╣")
    print(f"║  Heat Sink Severity     (HSS): {asi['hss_score']:5.1f}    max loss {asi['max_loss_pct']:.1f}%          ║")
    print(f"║  OAR Clearance Margin   (OCM): {asi['ocm_score']:5.1f}    min clear {asi['min_clear_mm']:.1f} mm        ║")
    print(f"║  Coverage Confidence    (CC):  {asi['cc_score']:5.1f}    margin {asi['margin_mm']:.1f} mm            ║")
    print(f"║  Directional Risk Asym. (DRA): {asi['dra_score']:5.1f}    spread {asi['spread_pct']:.1f}%              ║")
    if asi["directional"]:
        print(f"║  Directional Antenna    (DAS): {asi['das_score']:5.1f}    null aligned to OAR              ║")
    print("╠" + "═"*68 + "╣")
    print(f"║  Tier: {opt_result['tier']}  — {opt_result['tier_label']:<55} ║")
    print(f"║  Regime: {opt_result['P_opt']:.0f} W × {opt_result['t_opt']:.0f} s  "
          f"| zone {opt_result['zone_diam_cm']:.2f} cm  "
          f"| P_net {opt_result['P_net_W']:.1f} W{' '*18}║")
    print("╚" + "═"*68 + "╝")


# ══════════════════════════════════════════════════════════════════════════════
#  ABLATION ZONE MESH BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def make_ellipsoid(centroid, fwd_m, diam_m, needle_dir=None):
    a = max(fwd_m, 0.005)
    b = max(diam_m / 2.0, 0.003)
    sph = pv.Sphere(radius=1.0, theta_resolution=24, phi_resolution=24)
    pts = np.array(sph.points)
    pts[:, 0] *= a
    pts[:, 1] *= b
    pts[:, 2] *= b
    if needle_dir is not None:
        nd = np.array(needle_dir, dtype=float)
        nd /= max(np.linalg.norm(nd), 1e-9)
        if abs(nd[0]) < 0.9:
            up = np.array([1.0, 0.0, 0.0])
        else:
            up = np.array([0.0, 1.0, 0.0])
        right  = np.cross(nd, up); right /= max(np.linalg.norm(right), 1e-9)
        up_new = np.cross(right, nd)
        R = np.column_stack([nd, right, up_new])
        pts = pts @ R.T
    sph.points = pts + centroid
    return sph


def make_dshaped_zone(centroid, r_fwd_m, r_rear_m, antenna_axis, frac=1.0):
    """D-shaped zone: two half-ellipsoids joined at equatorial plane."""
    ax = np.array(antenna_axis, dtype=float)
    ax /= max(np.linalg.norm(ax), 1e-9)
    if abs(ax[0]) < 0.9:
        up = np.array([1.0, 0.0, 0.0])
    else:
        up = np.array([0.0, 1.0, 0.0])
    right  = np.cross(ax, up); right /= max(np.linalg.norm(right), 1e-9)
    up_new = np.cross(right, ax)
    R = np.column_stack([ax, right, up_new])

    meshes = []
    for half, r_ax in [(1, r_fwd_m * frac), (-1, r_rear_m * frac)]:
        sph = pv.Sphere(radius=1.0, theta_resolution=20, phi_resolution=20)
        pts = np.array(sph.points)
        mask = (pts[:, 0] * half) >= 0
        pts[~mask, 0] = 0.0
        pts[:, 0] *= r_ax
        pts[:, 1] *= max(r_fwd_m * frac, 0.003)
        pts[:, 2] *= max(r_fwd_m * frac, 0.003)
        pts = pts @ R.T
        sph.points = pts + centroid
        meshes.append(sph)
    return meshes


# ══════════════════════════════════════════════════════════════════════════════
#  BLOOD PARTICLE SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

class VesselParticleSystem:
    def __init__(self, vessel, vessel_name, n_particles=80):
        pts    = np.array(vessel.points)
        D      = VESSEL_DIAMETERS[vessel_name]
        R      = D / 2.0
        u_mean = VESSEL_VELOCITIES[vessel_name]
        Re     = (RHO_B * u_mean * D) / MU_B
        cen    = pts - pts.mean(axis=0)
        _, _, vt = np.linalg.svd(cen[:min(5000, len(cen))], full_matrices=False)
        self.flow_dir = vt[0]
        self.origin   = pts.mean(axis=0)
        proj   = cen.dot(self.flow_dir)
        self.L = max(float(proj.max() - proj.min()), 0.02)
        idx    = np.random.choice(len(pts), min(n_particles, len(pts)),
                                  replace=False)
        spts   = pts[idx]
        rel    = spts - self.origin
        axc    = np.outer(rel.dot(self.flow_dir), self.flow_dir)
        perp   = rel - axc
        rv     = np.linalg.norm(perp, axis=1)
        self.r_norm  = np.clip(rv / (R + 1e-9), 0, 1)
        if Re < 2300:
            self.u_local = u_mean * 2.0 * (1.0 - self.r_norm**2)
        else:
            self.u_local = u_mean * (8/7) * (9/8) * (1.0 - self.r_norm)**(1/7)
        self.phase    = np.random.uniform(0, self.L, len(idx))
        self.base_pts = spts - np.outer(rel.dot(self.flow_dir), self.flow_dir)
        self.vessel_name = vessel_name
        self.Re          = Re
        self.n           = len(idx)
        self.speed_scale = u_mean / 500.0

    def update(self, t):
        axial = (self.phase + self.speed_scale * t) % self.L
        return self.base_pts + np.outer(axial, self.flow_dir), self.u_local


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 9 — PHASE 3 ANIMATED VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def phase3_visualise(surface, vessels, vnames, tumors, centroids,
                      sel_idx, opt_result, asi, needle_dir, ray_results,
                      tissue_key, consist_key):

    sel     = opt_result
    centroid = centroids[sel_idx]
    directional = opt_result.get("directional", False)

    pl = pv.Plotter(window_size=[1600, 950])

    # ── Static anatomy
    pl.add_mesh(surface, color="bisque", opacity=0.12, label="Skin")
    for i, v in enumerate(vessels):
        pl.add_mesh(v, color=VESSEL_COLOR_MAP.get(vnames[i], "cyan"),
                    opacity=0.65, label=vnames[i])
    for i, t in enumerate(tumors):
        alpha = 0.90 if i == sel_idx else 0.20
        pl.add_mesh(smooth_tumor(t), color=TUMOR_COLORS[i % len(TUMOR_COLORS)],
                    opacity=alpha)

    # ── Needle path (from skin hit toward centroid)
    if ray_results:
        hit = ray_results[0]["hit_point"]
        needle_pts = np.array([hit, centroid])
        needle_line = pv.Spline(needle_pts, 30)
        pl.add_mesh(needle_line, color="silver", line_width=4,
                    opacity=0.90, label="Needle path")
        entry_sphere = pv.Sphere(radius=0.005, center=hit)
        pl.add_mesh(entry_sphere, color="white", opacity=1.0, label="Skin entry")

    # ── Heat-sink arrows
    per_hs = sel["per_vessel_hs"]
    losses = [hs["loss_pct"] for hs in per_hs.values()]
    mx, mn = max(losses), min(losses)
    def col(p):
        t = float(np.clip((p - mn) / max(mx - mn, 0.01), 0, 1))
        return ([2*t, 1.0, 0.0] if t < 0.5 else [1.0, 2*(1-t), 0.0])
    for vn, hs in per_hs.items():
        if vn not in vnames:
            continue
        pts  = np.array(vessels[vnames.index(vn)].points)
        raw  = pts[cKDTree(pts).query(centroid, k=1)[1]] - centroid
        dist = np.linalg.norm(raw)
        if dist < 1e-6:
            continue
        arr_len = max(0.04 * hs["loss_pct"] / max(mx, 1.), 0.005)
        sar_w   = float(hs.get("sar_weight", 1.0))
        pl.add_mesh(
            pv.Arrow(start=centroid, direction=raw/dist, scale=arr_len,
                     tip_length=0.3, tip_radius=0.05, shaft_radius=0.02),
            color=col(hs["loss_pct"]),
            opacity=float(np.clip(0.30 + 0.65 * sar_w, 0, 1)))

    # ── Directional antenna axis arrows
    if directional and "antenna_axis" in opt_result:
        ax = opt_result["antenna_axis"]
        ax_u = ax / max(np.linalg.norm(ax), 1e-9)
        pl.add_mesh(pv.Arrow(start=centroid, direction=ax_u, scale=0.07,
                             tip_length=0.25, tip_radius=0.07, shaft_radius=0.025),
                    color="cyan", opacity=0.95, label="Antenna fwd")
        pl.add_mesh(pv.Arrow(start=centroid, direction=-ax_u, scale=0.045,
                             tip_length=0.25, tip_radius=0.06, shaft_radius=0.018),
                    color="magenta", opacity=0.80, label="Antenna null (OAR)")

    # ── Animated ablation zone
    zone_actors = []
    t_val = [0.0]
    playing = [True]

    def clear_zone():
        for a in zone_actors:
            pl.remove_actor(a)
        zone_actors.clear()

    def draw_zone(frac):
        clear_zone()
        frac = float(np.clip(frac, 0.01, 1.0))
        if directional:
            r_fwd  = opt_result.get("r_fwd_m", 0.02)
            r_rear = opt_result.get("r_rear_m", 0.01)
            meshes = make_dshaped_zone(centroid, r_fwd, r_rear,
                                        opt_result["antenna_axis"], frac)
            for m in meshes:
                zone_actors.append(
                    pl.add_mesh(m, color="orangered", opacity=0.45 * frac,
                                show_edges=False))
        else:
            fwd_m  = (opt_result["zone_fwd_cm"] / 100.0) * frac
            diam_m = (opt_result["zone_diam_cm"] / 100.0) * frac
            zone_actors.append(
                pl.add_mesh(make_ellipsoid(centroid, fwd_m, diam_m, needle_dir),
                            color="orangered", opacity=0.45 * frac,
                            show_edges=False))

    # ── Particle systems
    particles_sys = [VesselParticleSystem(v, vnames[i])
                     for i, v in enumerate(vessels)]
    particle_actors = []

    def update_particles(t_now):
        for a in particle_actors:
            pl.remove_actor(a)
        particle_actors.clear()
        for ps in particles_sys:
            pts_now, u_loc = ps.update(t_now)
            pc = pv.PolyData(pts_now)
            pc["speed"] = u_loc
            actor = pl.add_mesh(
                pc, scalars="speed", cmap="Blues",
                point_size=4, render_points_as_spheres=True,
                show_scalar_bar=False, opacity=0.70)
            particle_actors.append(actor)

    draw_zone(0.01)

    def update_cb(t_slider):
        t_val[0] = float(t_slider)
        frac = t_val[0] / max(opt_result["t_opt"], 1.0)
        draw_zone(frac)
        update_particles(t_val[0])

    pl.add_slider_widget(update_cb, rng=[0, opt_result["t_opt"]],
                         value=0, title="Time (s)",
                         pointa=(0.10, 0.08), pointb=(0.55, 0.08),
                         style="modern")

    # ── HUD text
    tier_info = f"Tier {opt_result['tier']} | {opt_result['tier_label']}"
    abl_info  = (f"{opt_result['P_opt']:.0f} W × {opt_result['t_opt']:.0f} s  "
                 f"| zone {opt_result['zone_diam_cm']:.2f} cm  "
                 f"| P_net {opt_result['P_net_W']:.1f} W")
    hud = (f"ASI = {asi['asi']:.1f}/100  [{asi['risk_label']}]\n"
           f"{tier_info}\n"
           f"{abl_info}\n"
           f"OCM {asi['ocm_score']:.0f}  HSS {asi['hss_score']:.0f}  "
           f"CC {asi['cc_score']:.0f}  DRA {asi['dra_score']:.0f}"
           + (f"  DAS {asi['das_score']:.0f}" if directional else ""))
    pl.add_text(hud, position="upper_left", font_size=10)

    pl.add_legend(loc="upper right", size=(0.22, 0.28))
    pl.show_axes()
    pl.show()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "╔" + "═"*70 + "╗")
    print("║   UNIFIED MWA TREATMENT PLANNING SYSTEM  —  v1.0                    ║")
    print("╚" + "═"*70 + "╝\n")

    # ── Stage 0: Load data
    print("STAGE 0 — Loading VTK files …")
    tumor_raw = load_vtk(TUMOR_VTK)
    surface   = load_vtk(SURFACE_VTK)
    if tumor_raw is None or surface is None:
        print("❌ Cannot continue without tumor and surface meshes.")
        sys.exit(1)
    tumor_raw = rescale(tumor_raw)
    surface   = rescale(surface)

    vessels = []
    for i, path in enumerate(VESSEL_VTK_LIST):
        v = rescale(load_vtk(path))
        if v is not None:
            vessels.append(v)
        else:
            # Placeholder so indices stay consistent
            print(f"  ⚠  Vessel {VESSEL_NAMES[i]} not loaded — excluded from analysis")
    vnames = VESSEL_NAMES[:len(vessels)]

    if not vessels:
        print("❌ No vessel meshes loaded.")
        sys.exit(1)

    # ── Stage 1: Extract tumors
    print("\nSTAGE 1 — Extracting tumors …")
    tumors = extract_tumors(tumor_raw)

    # ── Stage 2: Metrics
    print("\nSTAGE 2 — Computing tumor metrics …")
    metrics = tumor_metrics(tumors, surface, vessels, vnames)
    for m in metrics:
        flag = "✔ eligible" if m["eligible"] else "✘ ineligible"
        print(f"  T{m['idx']}: diam={m['diameter_cm']:.2f}cm  "
              f"depth={m['depth_cm']:.2f}cm  "
              f"closest={m['closest_vessel']} @{m['min_vessel_m']*1000:.1f}mm  {flag}")

    # ── Stage 3: Phase 1 + Phase 2
    phase1_overview(surface, vessels, vnames, tumors, metrics)
    sel, type_key, consist_key = phase2_pick_tumor(metrics, vnames)

    sel_idx       = sel["idx"]
    centroid      = sel["centroid"]
    centroid_dists = sel["centroid_dists"]

    print(f"\n  Selected: Tumor {sel_idx}  |  "
          f"{TUMOR_TYPES[type_key]['label']}  |  "
          f"{CONSISTENCY_FACTORS[consist_key]['label']}")

    # ── Stage 5: Ray casting
    print("\nSTAGE 5 — Ray casting (needle route) …")
    ray_results = cast_rays(centroid, surface, vessels, vnames,
                             n_theta=20, n_phi=40)
    needle_dir, best_ray = pick_needle_direction(ray_results)
    ray_losses = [r["heat_loss_pct"] for r in ray_results]

    # ── Stage 7: Regime decision
    opt_result, tier = run_regime_decision(
        tumor_diam_cm  = sel["diameter_cm"],
        tissue_key     = type_key,
        consist_key    = consist_key,
        centroid_dists = centroid_dists,
        vnames         = vnames,
        vessels        = vessels,
        centroid       = centroid,
        margin_cm      = 0.5,
    )

    # ── Stage 8: ASI
    print("\nSTAGE 8 — Computing Ablation Safety Index …")
    asi = compute_asi(opt_result, sel["diameter_cm"], ray_losses)
    print_asi(asi, opt_result)

    # ── Stage 9: Visualisation
    print("\nSTAGE 9 — Launching treatment planning visualisation …")
    centroids = [m["centroid"] for m in metrics]
    phase3_visualise(
        surface       = surface,
        vessels       = vessels,
        vnames        = vnames,
        tumors        = tumors,
        centroids     = centroids,
        sel_idx       = sel_idx,
        opt_result    = opt_result,
        asi           = asi,
        needle_dir    = needle_dir,
        ray_results   = ray_results,
        tissue_key    = type_key,
        consist_key   = consist_key,
    )

    print("\n✔  Pipeline complete.")
    return opt_result, asi


if __name__ == "__main__":
    main()
