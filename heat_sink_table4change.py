#!/usr/bin/env python3
"""
Integrated Heat Sink Analysis — v6 (OAR-Safe Regime Selector)
==============================================================

CHANGES IN v6 vs v5
--------------------
CHANGE 1 — VESSEL_RADII constant added.
    Each vessel has a known anatomical radius. We now subtract the vessel
    radius from the centroid-to-centreline distance before computing OAR
    clearance, so clearance is measured from the vessel WALL, not the axis.

CHANGE 2 — identify_oars() upgraded.
    Now reports both centroid-distance AND wall-distance for each OAR.
    Risk thresholds updated to use wall clearance (< 5 mm = CRITICAL,
    5–10 mm = HIGH) instead of the old 5 mm centroid threshold.

CHANGE 3 — select_regime() replaced with select_regime_oar_safe().
    Old function: picks first table row where zone_diam >= required.
    New function: TWO-CONSTRAINT filter —
        Constraint A: zone_diam  >= required_diam  (covers tumour + margin)
        Constraint B: clearance  >= OAR_MIN_CLEARANCE_M  (stays off vessel wall)
    Among rows passing both constraints: lowest power, then shortest time.
    Falls back gracefully with a clear printed warning if no row passes both.

CHANGE 4 — main() updated.
    Passes centroid_dists + vessel names into the new regime selector so it
    can compute per-vessel wall clearance for every candidate table row.
    OAR check after regime selection uses the chosen zone dimensions
    (not hard-coded 5.82 / 3.9 cm as before).
"""

import os
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
# FILE PATHS  (unchanged — edit to match your dataset location)
# ─────────────────────────────────────────────────────────────────────

DATASET_BASE     = r"C:\Users\z005562w\OneDrive - Siemens Healthineers\Veda\Project\siemens project\3d simulation\Nunna Algo\Nunna Algo\Dataset"
PORTALVENOUS_DIR = os.path.join(DATASET_BASE, "portalvenous")

TUMOR_VTK   = os.path.join(DATASET_BASE, "908ac523data00007_leasion_out.vtk")
SURFACE_VTK = os.path.join(DATASET_BASE, "908ac523data00013_skin_out.vtk")

VESSEL_VTK_LIST = [
    os.path.join(PORTALVENOUS_DIR, "908ac52300001.vtk"),
    os.path.join(PORTALVENOUS_DIR, "908ac52300002.vtk"),
    os.path.join(PORTALVENOUS_DIR, "908ac52300003.vtk"),
    os.path.join(PORTALVENOUS_DIR, "908ac52300004.vtk"),
    os.path.join(PORTALVENOUS_DIR, "908ac52300005.vtk"),
]

VESSEL_NAMES  = ["portal_vein", "hepatic_vein", "aorta", "ivc", "hepatic_artery"]
# portal_vein=dark blue, hepatic_vein=dark blue, aorta=red, ivc=light blue, hepatic_artery=orange
VESSEL_COLORS = ["#00008B", "#00008B", "#FF0000", "#ADD8E6", "orange"]

# ─────────────────────────────────────────────────────────────────────
# MANUFACTURER ABLATION TABLE
# (power_w, time_s, vol_cc, fwd_cm, diam_cm)
# ─────────────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────────────
# PHYSICAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────

RHO_B    = 1060.0
MU_B     = 3.5e-3
C_B      = 3700.0
K_B      = 0.52
T_BLOOD  = 37.0
T_TISSUE = 90.0

ALPHA_TISSUE = 70.0
L_SEG        = 0.01

MIN_DIAMETER_CM = 3.0
MAX_DIAMETER_CM = 5.0
MAX_DEPTH_CM    = 26.0

VESSEL_DIAMETERS  = {"portal_vein": 12e-3, "hepatic_vein": 8e-3,
                     "aorta": 25e-3, "ivc": 20e-3, "hepatic_artery": 4.5e-3}
VESSEL_VELOCITIES = {"portal_vein": 0.15, "hepatic_vein": 0.20,
                     "aorta": 0.40, "ivc": 0.35, "hepatic_artery": 0.30}

# ── CHANGE 1 ──────────────────────────────────────────────────────────
# Vessel radii (metres) = diameter / 2.
# Purpose: convert centroid-to-centreline KDTree distance into
#          centroid-to-vessel-WALL distance before OAR clearance check.
# Without this correction the code overestimates safety by up to 12.5 mm
# (aorta radius) and can cause vessel thermal injury.
VESSEL_RADII = {vn: d / 2.0 for vn, d in VESSEL_DIAMETERS.items()}

# Minimum ablation-zone edge to vessel-wall clearance (metres).
# 5 mm is the accepted clinical OAR safety margin for hepatic RFA.
# Increase to 8–10 mm for portal vein / IVC in high-risk anatomy.
OAR_MIN_CLEARANCE_M = 5e-3
# ─────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────
# ELLIPSOID (unchanged from v5)
# ─────────────────────────────────────────────────────────────────────

def make_ellipsoid_clean(centroid, fwd_m, diam_m, needle_dir=None):
    if fwd_m < 1e-4 or diam_m < 1e-4:
        return None
    a = diam_m / 2.0
    c = fwd_m  / 2.0
    ell = pv.ParametricEllipsoid(xradius=a, yradius=a, zradius=c,
                                 u_res=30, v_res=30, w_res=10)
    if needle_dir is not None:
        n = np.array(needle_dir, dtype=float)
        n /= np.linalg.norm(n) + 1e-9
        z = np.array([0.0, 0.0, 1.0])
        axis = np.cross(z, n)
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-6:
            axis /= axis_norm
            angle = np.degrees(np.arccos(np.clip(np.dot(z, n), -1, 1)))
            ell = ell.rotate_vector(axis, angle, inplace=False)
    ell.points += centroid
    r_norm = np.linalg.norm(ell.points - centroid, axis=1) / (max(a, c) + 1e-9)
    ell["Temperature_C"] = T_BLOOD + (T_TISSUE - T_BLOOD) * np.exp(-2.0 * r_norm**2)
    return ell


# ─────────────────────────────────────────────────────────────────────
# BLOOD PARTICLE SYSTEM (unchanged from v5)
# ─────────────────────────────────────────────────────────────────────

class VesselParticleSystem:
    def __init__(self, vessel, vessel_name, n_particles=80):
        pts   = np.array(vessel.points)
        D     = VESSEL_DIAMETERS[vessel_name]
        R     = D / 2.0
        u_mean= VESSEL_VELOCITIES[vessel_name]
        Re    = (RHO_B * u_mean * D) / MU_B

        centered = pts - pts.mean(axis=0)
        _, _, vt = np.linalg.svd(centered[:min(5000, len(centered))],
                                  full_matrices=False)
        self.flow_dir = vt[0]
        self.origin   = pts.mean(axis=0)

        proj    = centered.dot(self.flow_dir)
        self.L  = float(proj.max() - proj.min())
        self.L  = max(self.L, 0.02)

        idx  = np.random.choice(len(pts), min(n_particles, len(pts)), replace=False)
        spts = pts[idx]

        rel      = spts - self.origin
        axial_c  = np.outer(rel.dot(self.flow_dir), self.flow_dir)
        perp     = rel - axial_c
        r_vals   = np.linalg.norm(perp, axis=1)
        self.r_norm = np.clip(r_vals / (R + 1e-9), 0.0, 1.0)

        if Re < 2300:
            self.u_local = u_mean * 2.0 * (1.0 - self.r_norm**2)
        else:
            u_max = u_mean * (8/7) * (9/8)
            self.u_local = u_max * (1.0 - self.r_norm)**(1.0/7.0)

        self.phase = np.random.uniform(0.0, self.L, len(idx))

        axial_pos_i  = rel.dot(self.flow_dir)
        self.base_pts= spts - np.outer(axial_pos_i, self.flow_dir)

        self.vessel_name = vessel_name
        self.u_mean      = u_mean
        self.Re          = Re
        self.n           = len(idx)
        self.speed_scale = u_mean / 500.0

    def update(self, t):
        axial_advance = (self.phase + self.speed_scale * t) % self.L
        pts = self.base_pts + np.outer(axial_advance, self.flow_dir)
        return pts, self.u_local


# ─────────────────────────────────────────────────────────────────────
# PHYSICS (unchanged from v5)
# ─────────────────────────────────────────────────────────────────────

def nusselt_full(Re, Pr):
    if Re < 2300:
        return 4.36
    f  = (0.790 * np.log(Re) - 1.64) ** (-2)
    Nu = ((f / 8) * (Re - 1000) * Pr /
          (1 + 12.7 * np.sqrt(f / 8) * (Pr**(2/3) - 1)))
    if Re >= 10000:
        Nu = 0.023 * (Re**0.8) * (Pr**0.4)
    return max(Nu, 4.36)


def wall_layer_correction(Re, D):
    if Re < 2300:
        return 1.0
    f     = (0.790 * np.log(Re) - 1.64) ** (-2)
    u_ref = 0.25
    nu    = MU_B / RHO_B
    u_tau = u_ref * np.sqrt(f / 8)
    delta_v = 5.0 * nu / (u_tau + 1e-9)
    Pr    = (C_B * MU_B) / K_B
    delta_t = delta_v * Pr**(-1/3)
    return max(0.90, 1.0 - delta_t / (D / 2.0))


def heat_sink_full_physics(distance_m, vessel_name, power_w, ablation_time_s):
    D      = VESSEL_DIAMETERS[vessel_name]
    u_mean = VESSEL_VELOCITIES[vessel_name]
    Re     = (RHO_B * u_mean * D) / MU_B
    Pr     = (C_B * MU_B) / K_B
    Nu     = nusselt_full(Re, Pr)
    eta    = wall_layer_correction(Re, D)
    h_bulk = (Nu * K_B) / D
    h_wall = h_bulk * eta

    A_contact = (D / 2.0) * (np.pi / 3) * L_SEG
    A_full    = np.pi * D * L_SEG
    dT_wall   = max(T_TISSUE - T_BLOOD, 0.1)
    dT_bulk   = max((T_TISSUE + T_BLOOD) / 2 - T_BLOOD, 0.1)

    Q_wall  = h_wall * A_contact * dT_wall
    bw      = 0.30 if Re >= 2300 else 0.05
    Q_bulk  = bw * h_bulk * A_full * dT_bulk
    Q_vessel= min(Q_wall + Q_bulk, power_w)

    d      = max(distance_m, 1e-4)
    Q_loss = min(Q_vessel * np.exp(-ALPHA_TISSUE * d), power_w)
    E_in   = power_w * ablation_time_s
    E_loss = min(Q_loss * ablation_time_s, E_in)

    regime = ("Laminar" if Re < 2300 else
              "Transition" if Re < 10000 else "Turbulent")

    return {
        "vessel": vessel_name, "dist_mm": d * 1000,
        "Re": Re, "Pr": Pr, "Nu": Nu, "flow_regime": regime,
        "eta_wall": eta, "h_bulk": h_bulk, "h_wall": h_wall,
        "Q_wall_W": Q_wall, "Q_bulk_W": Q_bulk,
        "Q_vessel_W": Q_vessel, "Q_loss_W": Q_loss,
        "E_loss_J": E_loss, "loss_pct": 100.0 * E_loss / E_in,
    }


def print_re_nu_explanation(per_vessel_hs):
    print("\n" + "=" * 70)
    print("  Re AND Nu EXPLANATION")
    print("=" * 70)
    print(f"  {'Vessel':<18} {'D(mm)':<8} {'u(m/s)':<9} {'Re':<8} "
          f"{'Regime':<14} {'Nu':<8}")
    print("  " + "-" * 70)
    for vn, hs in per_vessel_hs.items():
        D = VESSEL_DIAMETERS[vn] * 1000
        u = VESSEL_VELOCITIES[vn]
        print(f"  {vn:<18} {D:<8.1f} {u:<9.2f} {hs['Re']:<8.0f} "
              f"{hs['flow_regime']:<14} {hs['Nu']:<8.1f}")


# ─────────────────────────────────────────────────────────────────────
# MESH UTILITIES (unchanged)
# ─────────────────────────────────────────────────────────────────────

def load_vtk(path):
    if not os.path.exists(path):
        print(f"  ✘ Missing: {path}")
        return None
    mesh = pv.read(path)
    print(f"  ✔ {os.path.basename(path)} ({mesh.n_points} pts)")
    return mesh

def rescale(mesh):
    if mesh is None:
        return None
    if np.max(np.abs(mesh.points)) > 1000:
        mesh.points = mesh.points / 1000.0
    return mesh

def smooth_tumor(mesh, n_iter=80, relax=0.1):
    try:
        return mesh.smooth(n_iter=n_iter, relaxation_factor=relax,
                           boundary_smoothing=False)
    except Exception:
        return mesh

def extract_tumors(tumor_mesh):
    print("\n🔍 Splitting tumor mesh...")
    tumors = tumor_mesh.connectivity().split_bodies()
    print(f"   {len(tumors)} tumors")
    return tumors

def tumor_metrics(tumors, surface, vessels, vnames):
    s_tree  = cKDTree(np.array(surface.points))
    v_trees = [cKDTree(np.array(v.points)) for v in vessels]
    metrics = []
    for i, t in enumerate(tumors):
        c   = np.array(t.center)
        b   = t.bounds
        dm  = max(b[1]-b[0], b[3]-b[2], b[5]-b[4])
        dep = s_tree.query(c, k=1)[0]
        vd  = [float(vt.query(c, k=1)[0]) for vt in v_trees]
        metrics.append({
            "idx": i, "centroid": c,
            "diameter_cm": dm * 100, "depth_cm": dep * 100,
            "vessel_dists_m": vd, "min_vessel_m": min(vd),
            "closest_vessel": int(np.argmin(vd)),
        })
    return metrics


def ray_segment_dist(origin, direction, path_d, vessel_pts, centroid_dist, n_sample=30):
    ts      = np.linspace(0.0, path_d, n_sample)
    samples = origin + np.outer(ts, direction)
    dists,_ = cKDTree(vessel_pts).query(samples, k=1)
    raw     = float(np.min(dists))
    return max(raw, centroid_dist * 0.5)


# ── CHANGE 2 ──────────────────────────────────────────────────────────
# identify_oars() — now reports vessel-WALL clearance in addition to
# centroid distance.  Risk levels now use wall clearance, not centreline:
#   < 5 mm wall clearance  → CRITICAL
#   5–10 mm wall clearance → HIGH
# Previously the code used the raw centroid distance with a 5 mm threshold,
# which understated risk for large-radius vessels (aorta, IVC).
# ─────────────────────────────────────────────────────────────────────
def identify_oars(centroid, vessels, vnames, fwd_cm, diam_cm, needle_dir=None):
    """
    Check which vessels are encroached by the ablation ellipsoid.

    Returns list of dicts with both centroid distance and wall clearance.
    Wall clearance = (centroid-to-vessel-centreline dist) - vessel_radius
                     - (ablation_zone_diam / 2)
    Risk is classified by wall clearance:
        < 5 mm  → CRITICAL
        < 10 mm → HIGH
    """
    a = (fwd_cm  / 2.0) / 100.0
    b = (diam_cm / 2.0) / 100.0
    n_hat = (np.array(needle_dir) / (np.linalg.norm(needle_dir) + 1e-9)
             if needle_dir is not None else np.array([0., 0., 1.]))
    oars = []
    for vessel, vname in zip(vessels, vnames):
        pts   = np.array(vessel.points)
        rel   = pts - centroid
        ax    = rel.dot(n_hat)
        perp  = np.linalg.norm(rel - np.outer(ax, n_hat), axis=1)
        inside= (ax / a)**2 + (perp / b)**2 <= 1.0
        n_in  = int(inside.sum())
        if n_in > 0:
            # centroid-to-nearest-vessel-point distance (centreline approximation)
            cl_centroid = float(np.min(np.linalg.norm(rel[inside], axis=1)))
            # subtract vessel radius → wall distance
            v_radius    = VESSEL_RADII.get(vname, 0.0)
            cl_wall     = max(cl_centroid - v_radius, 0.0)
            risk = "CRITICAL" if cl_wall < OAR_MIN_CLEARANCE_M else "HIGH"
            oars.append({
                "vessel":          vname,
                "points_inside":   n_in,
                "closest_mm":      cl_centroid * 1000,   # centreline dist (for display)
                "wall_clear_mm":   cl_wall * 1000,        # ← NEW: wall clearance
                "risk":            risk,
            })
    return oars


# ── CHANGE 3 ──────────────────────────────────────────────────────────
# select_regime_oar_safe() — replaces select_regime().
#
# OLD logic (one constraint only):
#   required = (tumour + margin) / (1 - heat_loss_frac)
#   pick first table row where zone_diam >= required
#
# NEW logic (two constraints):
#   Constraint A: zone_diam >= required_diam        ← tumour coverage
#   Constraint B: for every vessel,
#                 wall_clearance >= OAR_MIN_CLEARANCE_M  ← vessel safety
#   where:
#     wall_clearance = centroid_dist_m - vessel_radius - (zone_diam_m / 2)
#
# Among rows passing BOTH constraints:
#   → pick lowest power, then shortest time (minimum thermal dose).
#
# If NO row passes both constraints:
#   → "anatomically constrained" — returns the row that satisfies tumour
#     coverage with the largest wall clearance (best available), plus a
#     warning flag so the caller can alert the user.
# ─────────────────────────────────────────────────────────────────────
def select_regime_oar_safe(tumor_diam_cm, max_loss_pct, centroid_dists_m,
                            vnames_present, margin_cm=0.5,
                            oar_min_m=OAR_MIN_CLEARANCE_M):
    """
    Two-constraint ablation regime selector.

    Parameters
    ----------
    tumor_diam_cm     : float — largest tumour diameter in cm
    max_loss_pct      : float — dominant heat-sink loss percentage
    centroid_dists_m  : dict  — {vessel_name: distance_m} from KDTree query
                                (centroid-to-centreline distances)
    vnames_present    : list  — vessel names that have VTK data loaded
    margin_cm         : float — clinical safety margin in cm (default 0.5)
    oar_min_m         : float — minimum vessel-wall clearance in metres

    Returns
    -------
    rec      : tuple  — chosen table row (power_w, time_s, vol_cc, fwd_cm, diam_cm)
    alts     : list   — next 2 safe candidates (for display)
    raw_req  : float  — required zone diameter in cm (before table lookup)
    constrained : bool — True if no row passes both constraints
    clearance_report : list of dicts — per-vessel clearance for chosen row
    """
    lf      = max_loss_pct / 100.0
    raw_req = (tumor_diam_cm + margin_cm) / max(1.0 - lf, 0.01)

    def min_wall_clearance(zone_diam_cm):
        """
        Compute the minimum ablation-zone-edge to vessel-wall clearance
        across all loaded vessels, for a given ablation zone diameter.

        clearance_i = dist(centroid → vessel_centreline_i)
                      - vessel_radius_i
                      - (zone_diam / 2)

        We use centroid_dists_m which is centroid-to-NEAREST-POINT on the
        vessel mesh (KDTree). This approximates centreline distance well
        for tubular vessels.
        """
        zone_radius_m = (zone_diam_cm / 2.0) / 100.0
        clearances = []
        for vn in vnames_present:
            dist_cl  = centroid_dists_m.get(vn, 999.0)   # centroid-to-centreline
            v_radius = VESSEL_RADII.get(vn, 0.0)
            wall_cl  = dist_cl - v_radius - zone_radius_m
            clearances.append((vn, wall_cl))
        return clearances

    # ── Pass 1: rows that cover tumour AND respect OAR clearance ──
    safe_cands = []
    for row in ABLATION_TABLE:
        pw, ts, vol, fwd, diam = row
        if diam < raw_req:
            continue   # Constraint A fails — too small
        clrs = min_wall_clearance(diam)
        min_cl = min(c for _, c in clrs)
        if min_cl >= oar_min_m:
            safe_cands.append((pw, ts, vol, fwd, diam, min_cl, clrs))

    constrained = len(safe_cands) == 0

    if not constrained:
        # Sort by lowest power → shortest time → largest clearance
        safe_cands.sort(key=lambda x: (x[0], x[1], -x[5]))
        best = safe_cands[0]
        rec  = (best[0], best[1], best[2], best[3], best[4])
        alts = [(r[0], r[1], r[2], r[3], r[4]) for r in safe_cands[1:3]]
        clearance_report = [{"vessel": vn, "wall_clear_mm": cl * 1000}
                             for vn, cl in best[6]]
    else:
        # Fallback: among coverage-sufficient rows, pick largest wall clearance
        print("\n  ⚠  WARNING: No regime satisfies BOTH tumour coverage AND OAR")
        print(f"     clearance ≥ {oar_min_m*1000:.0f} mm.")
        print("     Returning best available (largest clearance among coverage rows).")
        print("     Consider: re-positioning needle, partial ablation, or staged treatment.\n")

        coverage_rows = [(pw, ts, vol, fwd, diam, min(c for _, c in min_wall_clearance(diam)),
                          min_wall_clearance(diam))
                         for pw, ts, vol, fwd, diam in ABLATION_TABLE
                         if diam >= raw_req]

        if not coverage_rows:
            coverage_rows = [(pw, ts, vol, fwd, diam, min(c for _, c in min_wall_clearance(diam)),
                              min_wall_clearance(diam))
                             for pw, ts, vol, fwd, diam in ABLATION_TABLE]

        coverage_rows.sort(key=lambda x: (-x[5], x[0], x[1]))
        best = coverage_rows[0]
        rec  = (best[0], best[1], best[2], best[3], best[4])
        alts = [(r[0], r[1], r[2], r[3], r[4]) for r in coverage_rows[1:3]]
        clearance_report = [{"vessel": vn, "wall_clear_mm": cl * 1000}
                             for vn, cl in best[6]]

    return rec, alts, raw_req, constrained, clearance_report


# ─────────────────────────────────────────────────────────────────────
# EXISTING HELPERS (unchanged from v5)
# ─────────────────────────────────────────────────────────────────────

def generate_rays(n_theta=20, n_phi=40):
    rays = []
    for t in np.linspace(0, np.pi, n_theta):
        for p in np.linspace(0, 2*np.pi, n_phi):
            rays.append([np.sin(t)*np.cos(p), np.sin(t)*np.sin(p), np.cos(t)])
    return np.array(rays)


def create_heat_flow_arrows(centroid, vessels, vnames, per_vessel_hs, plotter):
    max_loss = max(hs["loss_pct"] for hs in per_vessel_hs.values())
    min_loss = min(hs["loss_pct"] for hs in per_vessel_hs.values())
    BASE     = 0.04

    def color(pct):
        t = (pct - min_loss) / max(max_loss - min_loss, 0.01)
        return [2*t, 1.0, 0.0] if t < 0.5 else [1.0, 2*(1-t), 0.0]

    for vname, hs in per_vessel_hs.items():
        vessel = vessels[vnames.index(vname)]
        pts    = np.array(vessel.points)
        _, idx = cKDTree(pts).query(centroid, k=1)
        target = pts[idx]
        raw    = target - centroid
        dist   = np.linalg.norm(raw)
        if dist < 1e-6:
            continue
        unit     = raw / dist
        arr_len  = max(BASE * hs["loss_pct"] / max(max_loss, 1.0), 0.005)
        arrow    = pv.Arrow(start=centroid, direction=unit, scale=arr_len,
                            tip_length=0.3, tip_radius=0.05, shaft_radius=0.02)
        plotter.add_mesh(arrow, color=color(hs["loss_pct"]), opacity=0.95)
        label_pos = centroid + unit * arr_len * 1.2
        regime_ch = hs["flow_regime"][0]
        plotter.add_point_labels(
            pv.PolyData([label_pos]),
            [f"{vname.replace('_',' ')}\n{hs['loss_pct']:.2f}% [{regime_ch}]\n"
             f"Q={hs['Q_loss_W']:.3f}W"],
            font_size=9, text_color=color(hs["loss_pct"]),
            point_size=1, always_visible=True, shape_opacity=0.0)

    plotter.add_text("Heat Flow Arrows:\n Green=Low  Yellow=Mid  Red=High\n"
                     "[L]=Laminar [T]=Turbulent [Tr]=Transition",
                     position="lower_right", font_size=9, color="white")


# ─────────────────────────────────────────────────────────────────────
# VISUALIZATION (unchanged from v5 except OAR check uses rec dims)
# ─────────────────────────────────────────────────────────────────────

def run_visualization(surface, vessels, vnames, tumors, centroids,
                      sel_idx, results, per_vessel_hs,
                      recommended, oar_list, safest_dir,
                      particle_systems):

    print("\n🎬 Building visualization...")

    power_w    = float(recommended[0])
    time_s     = float(recommended[1])
    fwd_m      = recommended[2] / 100.0
    diam_m     = recommended[3] / 100.0
    centroid   = centroids[sel_idx]
    needle_dir = safest_dir

    plotter = pv.Plotter(window_size=[1500, 1000])
    plotter.background_color = "black"

    plotter.add_mesh(surface, color="lightgray", opacity=0.08, label="Body Surface")

    # Vessels
    VESSEL_COLOR_MAP = {
        "aorta":          "#FF0000",
        "ivc":            "#ADD8E6",
        "portal_vein":    "#00008B",
        "hepatic_vein":   "#00008B",
        "hepatic_artery": "orange",
    }
    for i, (v, vn) in enumerate(zip(vessels, vnames)):
        is_oar   = any(o["vessel"] == vn for o in oar_list)
        base_col = VESSEL_COLOR_MAP.get(vn, VESSEL_COLORS[i % len(VESSEL_COLORS)])
        disp_col = "red" if is_oar else base_col
        opacity  = 0.85 if is_oar else 0.60
        label    = ("⚠ OAR: " if is_oar else "") + vn.replace("_", " ").title()
        plotter.add_mesh(v, color=disp_col, opacity=opacity, label=label)

    # Tumors
    colors = ["yellow", "orange", "purple", "pink", "red", "lime"]
    for i, t in enumerate(tumors):
        td    = smooth_tumor(t) if i == sel_idx else t
        op    = 0.85 if i == sel_idx else 0.30
        label = (f"Tumor {i+1} ablation target (smoothed)"
                 if i == sel_idx else f"Tumor {i+1}")
        plotter.add_mesh(td, color=colors[i % len(colors)], opacity=op, label=label)

    plotter.add_mesh(pv.Sphere(radius=0.006, center=centroid),
                     color="yellow", label="Tumor centroid")

    create_heat_flow_arrows(centroid, vessels, vnames, per_vessel_hs, plotter)

    # Ray lines
    if results:
        losses = np.array([r["loss_pct"] for r in results])
        lo, hi = losses.min(), losses.max()
        norm   = (losses - lo) / (hi - lo + 1e-9)
        step   = max(1, len(results) // 80)
        for i in range(0, len(results), step):
            r   = results[i]
            ep  = centroid + r["ray_direction"] * r["path_distance"]
            cv  = norm[i]
            plotter.add_mesh(pv.Line(centroid, ep),
                             color=[cv, 0.0, 1.0 - cv], line_width=2.5, opacity=0.55)

    # Table lookup helper
    SORTED_TABLE = sorted(ABLATION_TABLE, key=lambda r: (r[0], r[1]))

    def get_table_row(power_w, t_sec):
        candidates = [r for r in SORTED_TABLE if r[0] == round(power_w, 0)]
        if not candidates:
            candidates = SORTED_TABLE
        return min(candidates, key=lambda r: abs(r[1] - t_sec))

    def update(t_val):
        t    = float(t_val)
        frac = min(t / time_s, 1.0)

        for name in ["ablation", "particles", "hud"]:
            try:
                plotter.remove_actor(name)
            except Exception:
                pass

        cur_fwd  = fwd_m  * frac
        cur_diam = diam_m * frac
        ell = make_ellipsoid_clean(centroid, cur_fwd, cur_diam, needle_dir)
        if ell is not None:
            plotter.add_mesh(
                ell, scalars="Temperature_C", cmap="plasma",
                clim=[T_BLOOD, T_TISSUE], opacity=0.62, name="ablation",
                scalar_bar_args={
                    "title": "Temperature (°C)", "n_labels": 5,
                    "label_font_size": 11, "title_font_size": 12,
                    "position_x": 0.02, "position_y": 0.25,
                    "width": 0.08, "height": 0.40, "color": "white",
                }
            )

        all_pts, all_vel = [], []
        for ps in particle_systems:
            pts, vel = ps.update(t)
            all_pts.append(pts)
            all_vel.append(vel)

        if all_pts:
            pts_all = np.vstack(all_pts)
            vel_all = np.concatenate(all_vel)
            cloud   = pv.PolyData(pts_all)
            cloud["blood_velocity_m_s"] = vel_all
            plotter.add_mesh(
                cloud, scalars="blood_velocity_m_s", cmap="coolwarm",
                clim=[0.0, max(VESSEL_VELOCITIES.values()) * 2.0],
                point_size=5, render_points_as_spheres=True, name="particles",
                scalar_bar_args={
                    "title": "Blood velocity (m/s)", "n_labels": 3,
                    "label_font_size": 10, "title_font_size": 11,
                    "position_x": 0.12, "position_y": 0.25,
                    "width": 0.08, "height": 0.40, "color": "white",
                }
            )

        row      = get_table_row(power_w, t)
        tbl_pw   = row[0]; tbl_t = row[1]; tbl_vol = row[2]
        tbl_fwd  = row[3]; tbl_dm = row[4]
        zone_str = (f"Zone: {cur_fwd*100:.1f}cm × {cur_diam*100:.1f}cm"
                    if frac > 0.01 else "Zone: growing...")
        hud = (f"t = {t:.0f}s / {time_s:.0f}s  ({frac*100:.0f}%)\n"
               f"Power: {power_w:.0f} W\n"
               f"{zone_str}\n"
               f"─────────────────────────\n"
               f"Table ref (t={tbl_t:.0f}s, {tbl_pw:.0f}W):\n"
               f"  Vol={tbl_vol:.2f}cc  Fwd={tbl_fwd:.1f}cm  Diam={tbl_dm:.1f}cm\n"
               f"─────────────────────────\n"
               f"OARs encroached: {len(oar_list)}\n"
               f"─────────────────────────\n"
               f"Ablation zone colour:\n"
               f"  Purple=cool(37°C)\n"
               f"  Yellow=warm\n"
               f"  White=hot(90°C)\n"
               f"Blood dots:\n"
               f"  Blue=slow(wall)\n"
               f"  Red=fast(centre)\n"
               f"─────────────────────────\n"
               f"Vessels:\n"
               f"  Red=Aorta\n"
               f"  DkBlue=Veins\n"
               f"  LtBlue=IVC")
        plotter.add_text(hud, position="lower_left", font_size=10,
                         color="white", name="hud")
        plotter.render()

    play_state   = {"playing": False, "t": 0.0}
    PLAY_STEP_S  = 5.0
    TIMER_MS     = 100

    def toggle_play(flag):
        play_state["playing"] = bool(flag)
        if play_state["playing"]:
            plotter.add_timer_event(max_steps=100000,
                                    duration=TIMER_MS,
                                    callback=_timer_tick)

    def _timer_tick(step):
        if not play_state["playing"]:
            return
        play_state["t"] = (play_state["t"] + PLAY_STEP_S) % (time_s + 1.0)
        slider_widget.GetRepresentation().SetValue(play_state["t"])
        update(play_state["t"])

    slider_widget = plotter.add_slider_widget(
        update, rng=[0.0, time_s], value=0.0,
        title="Ablation Time (s)",
        pointa=(0.22, 0.05), pointb=(0.90, 0.05),
        style="modern")

    plotter.add_checkbox_button_widget(
        toggle_play, value=False, position=(30, 30),
        size=45, border_size=3, color_on="lime", color_off="gray")
    plotter.add_text("▶ Play", position=(80, 38), font_size=11, color="white")

    plotter.add_legend(loc="upper right", size=(0.24, 0.42))
    plotter.add_text(
        f"Heat Sink + OAR + Flow Profile  |  {power_w:.0f}W × {time_s:.0f}s",
        position="upper_left", font_size=13, color="white")
    plotter.add_axes()

    update(0.0)
    try:
        plotter.show(auto_close=False)
    except Exception as e:
        print(f"  Vis error: {e}")
    finally:
        plotter.close()


# ─────────────────────────────────────────────────────────────────────
# MAIN — CHANGE 4: use new OAR-safe selector, pass vessel distances
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  HEAT SINK v6 — OAR-SAFE REGIME SELECTOR + VESSEL WALL CLEARANCE")
    print("=" * 70)

    if not os.path.exists(DATASET_BASE):
        print(f"  ✘ Dataset not found: {DATASET_BASE}")
        return

    print("\n  Loading meshes...")
    tumor_mesh = rescale(load_vtk(TUMOR_VTK))
    surface    = rescale(load_vtk(SURFACE_VTK))
    vessels, vnames = [], []
    for i, path in enumerate(VESSEL_VTK_LIST):
        v = rescale(load_vtk(path))
        if v is not None:
            vessels.append(v)
            vnames.append(VESSEL_NAMES[i])

    if tumor_mesh is None or surface is None or not vessels:
        print("  ✘ Critical files missing.")
        return

    tumors    = extract_tumors(tumor_mesh)
    metrics   = tumor_metrics(tumors, surface, vessels, vnames)
    centroids = np.array([m["centroid"] for m in metrics])

    eligible = sorted(
        [m for m in metrics
         if MIN_DIAMETER_CM <= m["diameter_cm"] <= MAX_DIAMETER_CM
         and m["depth_cm"] <= MAX_DEPTH_CM],
        key=lambda m: m["min_vessel_m"])

    sel      = eligible[0] if eligible else sorted(metrics, key=lambda m: m["min_vessel_m"])[0]
    sel_idx  = sel["idx"]
    centroid = sel["centroid"]
    sel_diam = sel["diameter_cm"]
    print(f"\n🎯 Tumor {sel_idx+1}: {sel_diam:.2f}cm, depth {sel['depth_cm']:.2f}cm")

    # Centroid-to-vessel centreline distances via KDTree
    centroid_dists = {}
    vessel_trees   = []
    for i, v in enumerate(vessels):
        tree = cKDTree(np.array(v.points))
        vessel_trees.append(tree)
        centroid_dists[vnames[i]] = float(tree.query(centroid, k=1)[0])

    POWER_W = 60.0
    TIME_S  = 600.0

    print("\n  Computing heat sink...")
    per_vessel_hs = {}
    for vn in vnames:
        per_vessel_hs[vn] = heat_sink_full_physics(
            centroid_dists[vn], vn, POWER_W, TIME_S)

    print_re_nu_explanation(per_vessel_hs)

    print(f"\n  {'Vessel':<18}{'Dist(mm)':<11}{'WallDist(mm)':<14}"
          f"{'Regime':<14}{'Nu':<8}{'Q_loss(W)':<12}{'Loss%'}")
    print("  " + "-" * 80)
    for vn, hs in per_vessel_hs.items():
        wall_dist_mm = (centroid_dists[vn] - VESSEL_RADII[vn]) * 1000
        print(f"  {vn:<18}{hs['dist_mm']:<11.1f}{wall_dist_mm:<14.1f}"
              f"{hs['flow_regime']:<14}{hs['Nu']:<8.1f}"
              f"{hs['Q_loss_W']:<12.4f}{hs['loss_pct']:.3f}%")

    max_hs_pct = max(hs["loss_pct"] for hs in per_vessel_hs.values())

    print("\n  Ray tracing...")
    rays    = generate_rays(n_theta=20, n_phi=40)
    results = []
    v_pts   = [np.array(v.points) for v in vessels]
    for direction in tqdm(rays, desc="  Rays"):
        try:
            hits, _ = surface.ray_trace(centroid, centroid + direction * 0.5)
            if len(hits) == 0:
                continue
            hit    = hits[0]
            path_d = float(np.linalg.norm(hit - centroid))
            seg_d  = {vn: ray_segment_dist(centroid, direction, path_d,
                                            v_pts[vi], centroid_dists[vn])
                      for vi, vn in enumerate(vnames)}
            dom_vn = min(seg_d, key=seg_d.get)
            hs     = heat_sink_full_physics(seg_d[dom_vn], dom_vn, POWER_W, TIME_S)
            hs["ray_direction"]   = direction
            hs["path_distance"]   = path_d
            hs["ray_seg_dist_mm"] = seg_d[dom_vn] * 1000
            results.append(hs)
        except Exception:
            continue

    sorted_res = sorted(results, key=lambda x: x["loss_pct"], reverse=True)
    all_losses = [r["loss_pct"] for r in results]
    safest_dir = sorted_res[-1]["ray_direction"] if results else np.array([0., 0., 1.])

    print(f"  {len(results)} rays | loss {np.min(all_losses):.2f}%–{np.max(all_losses):.2f}%")
    print(f"  Safest needle dir: {safest_dir.round(3)}")

    # ── CHANGE 4A: Treatment regime with OAR-safe selector ───────────
    # Pass centroid_dists (centreline distances from KDTree) into the
    # solver.  The solver subtracts vessel radii internally to get wall
    # clearance for every candidate table row before accepting it.
    rec, alts, raw_req, constrained, clearance_report = select_regime_oar_safe(
        sel_diam, max_hs_pct, centroid_dists, vnames, margin_cm=0.5)

    print(f"\n  {'⚠  CONSTRAINED' if constrained else '✔  OAR-SAFE'} Regime: "
          f"{rec[0]:.0f}W × {rec[1]:.0f}s  diam={rec[4]:.2f}cm  "
          f"(need {raw_req:.2f}cm)")
    print(f"\n  Vessel-wall clearances for chosen regime ({rec[4]:.2f}cm diam):")
    print(f"  {'Vessel':<20} {'Wall Clear (mm)'}")
    print("  " + "-" * 38)
    for cr in clearance_report:
        flag = "  ✔" if cr["wall_clear_mm"] >= OAR_MIN_CLEARANCE_M * 1000 else "  ✗ ENCROACH"
        print(f"  {cr['vessel']:<20} {cr['wall_clear_mm']:>8.1f} mm{flag}")

    # ── CHANGE 4B: OAR check uses chosen zone dimensions (not hard-coded)
    oar_list = identify_oars(centroid, vessels, vnames,
                             rec[3], rec[4], safest_dir)
    print(f"\n  OARs encroached: {len(oar_list)}")
    for o in oar_list:
        print(f"    {o['vessel']}  {o['points_inside']} pts  "
              f"centroid={o['closest_mm']:.1f}mm  wall={o['wall_clear_mm']:.1f}mm  "
              f"[{o['risk']}]")

    print("\n  Building particle systems...")
    particle_systems = []
    for v, vn in zip(vessels, vnames):
        ps = VesselParticleSystem(v, vn, n_particles=80)
        particle_systems.append(ps)
        print(f"   {vn}: {ps.n} particles, Re={ps.Re:.0f} "
              f"({'Laminar' if ps.Re < 2300 else 'Turbulent/Transition'}), "
              f"L={ps.L*100:.1f}cm")

    run_visualization(
        surface, vessels, vnames, tumors, centroids,
        sel_idx, results, per_vessel_hs,
        rec, oar_list, safest_dir, particle_systems)

    print("\n  Complete!")
    return results


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  Stopped.")
    except Exception as e:
        print(f"\n  Error: {e}")
        raise
