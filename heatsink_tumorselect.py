#!/usr/bin/env python3
"""
Integrated Heat Sink Analysis — v9 (INTEGRATED + ASI RISK INDEX)
=================================================================
Authors : Veda Nunna (Algorithm) — integrated build
Version : 9.0

WORKFLOW
--------
  PHASE 1  — Overview window opens showing ALL tumors + vessels.
             User inspects the 3D scene, closes the window.

  PHASE 2  — Terminal lists every tumor with its key metrics.
             User types the tumor number they want to analyse.

  PHASE 3  — Full heat-sink physics, OAR check, regime selection,
             Ablation Safety Index (ASI) computation, and animated
             treatment-planning visualisation.

NEW in v9
---------
  * Merged heatsink_tumorspecific.py  +  heat_sink_table5_toggle.py
  * Phase-1 overview visualisation (all tumors) before selection
  * Interactive terminal tumor picker
  * Ablation Safety Index (ASI) — composite 0-100 score with four
    sub-components:
      - Heat Sink Severity     (HSS)
      - OAR Clearance Margin   (OCM)
      - Coverage Confidence    (CC)
      - Directional Risk Asymmetry (DRA)
  * Full ASI breakdown printed in terminal + shown in HUD
"""

import os
import sys
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# FILE PATHS  — edit these to match your machine
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# COLOUR MAPS
# ─────────────────────────────────────────────────────────────────────────────

VESSEL_COLOR_MAP = {
    "aorta":           "#FF0000",   # arterial red
    "portal_vein":     "#1565C0",   # medium blue
    "hepatic_vein":    "#1E90FF",   # dodger blue
    "ivc":             "#1E90FF",   # dodger blue (venous)
    "hepatic_artery":  "orange",
}
TUMOR_COLORS = ["yellow", "orange", "purple", "pink", "red", "lime",
                "gold", "cyan", "salmon", "chartreuse"]


# ─────────────────────────────────────────────────────────────────────────────
# MANUFACTURER ABLATION TABLE  (power_W, time_s, vol_cc, fwd_cm, diam_cm)
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# PHYSICAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

RHO_B    = 1060.0   # blood density  kg/m³
MU_B     = 3.5e-3   # dynamic viscosity  Pa·s
C_B      = 3700.0   # specific heat  J/(kg·K)
K_B      = 0.52     # thermal conductivity  W/(m·K)
T_BLOOD  = 37.0     # °C
T_TISSUE = 90.0     # °C  (target ablation isotherm)

ALPHA_TISSUE    = 70.0   # tissue attenuation constant  1/m
L_SEG           = 0.01   # vessel segment length  m

MIN_DIAMETER_CM = 3.0
MAX_DIAMETER_CM = 5.0
MAX_DEPTH_CM    = 26.0

OAR_MIN_CLEARANCE_M = 5e-3   # 5 mm minimum wall clearance

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

# ASI weights (must sum to 1.0)
ASI_WEIGHTS = {
    "hss": 0.35,   # heat sink severity
    "ocm": 0.30,   # OAR clearance margin
    "cc":  0.20,   # coverage confidence
    "dra": 0.15,   # directional risk asymmetry
}


# ─────────────────────────────────────────────────────────────────────────────
# MESH UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def load_vtk(path):
    if not os.path.exists(path):
        print(f"  ✘ Missing: {path}")
        return None
    mesh = pv.read(path)
    print(f"  ✔ {os.path.basename(path)}  ({mesh.n_points} pts, {mesh.n_cells} cells)")
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
    print("\n🔍 Extracting individual tumors from combined mesh...")
    connected = tumor_mesh.connectivity()
    tumors    = connected.split_bodies()
    print(f"   Detected {len(tumors)} separate tumor(s)")
    return tumors


# ─────────────────────────────────────────────────────────────────────────────
# TUMOR METRICS
# ─────────────────────────────────────────────────────────────────────────────

def tumor_metrics(tumors, surface, vessels, vnames):
    s_tree  = cKDTree(np.array(surface.points))
    v_trees = [cKDTree(np.array(v.points)) for v in vessels]
    metrics = []
    for i, t in enumerate(tumors):
        c  = np.array(t.center)
        b  = t.bounds
        dm = max(b[1]-b[0], b[3]-b[2], b[5]-b[4])
        dep = float(s_tree.query(c, k=1)[0])
        vd  = [float(vt.query(c, k=1)[0]) for vt in v_trees]
        eligible = (MIN_DIAMETER_CM <= dm*100 <= MAX_DIAMETER_CM
                    and dep*100 <= MAX_DEPTH_CM)
        metrics.append({
            "idx":            i,
            "centroid":       c,
            "diameter_cm":    dm * 100.0,
            "depth_cm":       dep * 100.0,
            "vessel_dists_m": vd,
            "min_vessel_m":   min(vd),
            "closest_vessel": vnames[int(np.argmin(vd))],
            "eligible":       eligible,
        })
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# PHYSICS — heat-sink full model
# ─────────────────────────────────────────────────────────────────────────────

def nusselt_full(Re, Pr):
    if Re < 2300:
        return 4.36
    f  = (0.790 * np.log(Re) - 1.64) ** (-2)
    Nu = (f/8) * (Re - 1000) * Pr / (1.0 + 12.7 * np.sqrt(f/8) * (Pr**(2/3) - 1))
    if Re >= 10000:
        Nu = 0.023 * (Re**0.8) * (Pr**0.4)
    return max(Nu, 4.36)

def wall_layer_correction(Re, D):
    if Re < 2300:
        return 1.0
    f      = (0.790 * np.log(Re) - 1.64) ** (-2)
    nu     = MU_B / RHO_B
    u_tau  = 0.25 * np.sqrt(f / 8)
    dv     = 5.0 * nu / (u_tau + 1e-9)
    Pr     = (C_B * MU_B) / K_B
    dt     = dv * Pr**(-1/3)
    return max(0.90, 1.0 - dt / (D / 2.0))

def heat_sink_full_physics(distance_m, vessel_name, power_w, ablation_time_s):
    D      = VESSEL_DIAMETERS[vessel_name]
    u_mean = VESSEL_VELOCITIES[vessel_name]
    Re     = (RHO_B * u_mean * D) / MU_B
    Pr     = (C_B * MU_B) / K_B
    Nu     = nusselt_full(Re, Pr)
    eta    = wall_layer_correction(Re, D)
    h_bulk = (Nu * K_B) / D
    h_wall = h_bulk * eta

    A_contact = (D / 2.0) * (np.pi / 3.0) * L_SEG
    A_full    = np.pi * D * L_SEG
    dT_wall   = max(T_TISSUE - T_BLOOD, 0.1)
    dT_bulk   = max((T_TISSUE + T_BLOOD) / 2.0 - T_BLOOD, 0.1)
    Q_wall    = h_wall * A_contact * dT_wall
    bw        = 0.30 if Re >= 2300 else 0.05
    Q_bulk    = bw * h_bulk * A_full * dT_bulk
    Q_vessel  = min(Q_wall + Q_bulk, power_w)

    d      = max(distance_m, 1e-4)
    Q_loss = min(Q_vessel * np.exp(-ALPHA_TISSUE * d), power_w)
    E_in   = power_w * ablation_time_s
    E_loss = min(Q_loss * ablation_time_s, E_in)
    regime = ("Laminar"     if Re < 2300
              else "Transition" if Re < 10000
              else "Turbulent")

    return {
        "vessel":      vessel_name,
        "dist_mm":     d * 1000,
        "Re":          Re, "Pr": Pr, "Nu": Nu,
        "flow_regime": regime,
        "eta_wall":    eta,
        "h_bulk":      h_bulk, "h_wall": h_wall,
        "Q_wall_W":    Q_wall, "Q_bulk_W": Q_bulk,
        "Q_vessel_W":  Q_vessel, "Q_loss_W": Q_loss,
        "E_loss_J":    E_loss,
        "loss_pct":    100.0 * E_loss / E_in,
    }


# ─────────────────────────────────────────────────────────────────────────────
# RAY UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def generate_rays(n_theta=20, n_phi=40):
    rays = []
    for t in np.linspace(0, np.pi, n_theta):
        for p in np.linspace(0, 2 * np.pi, n_phi):
            rays.append([np.sin(t)*np.cos(p),
                         np.sin(t)*np.sin(p),
                         np.cos(t)])
    return np.array(rays)

def ray_segment_dist(origin, direction, path_d, vessel_pts,
                     centroid_dist, n_sample=30):
    ts      = np.linspace(0.0, path_d, n_sample)
    samples = origin + np.outer(ts, direction)
    dists, _ = cKDTree(vessel_pts).query(samples, k=1)
    return max(float(np.min(dists)), centroid_dist * 0.5)


# ─────────────────────────────────────────────────────────────────────────────
# OAR IDENTIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def identify_oars(centroid, vessels, vnames, fwd_cm, diam_cm,
                  needle_dir=None):
    a     = (fwd_cm  / 2.0) / 100.0
    b     = (diam_cm / 2.0) / 100.0
    n_hat = (np.array(needle_dir) / (np.linalg.norm(needle_dir) + 1e-9)
             if needle_dir is not None else np.array([0., 0., 1.]))
    oars  = []
    for vessel, vname in zip(vessels, vnames):
        pts  = np.array(vessel.points)
        rel  = pts - centroid
        ax   = rel.dot(n_hat)
        perp = np.linalg.norm(rel - np.outer(ax, n_hat), axis=1)
        inside = (ax / a)**2 + (perp / b)**2 <= 1.0
        n_in   = int(inside.sum())
        if n_in > 0:
            cl_c    = float(np.min(np.linalg.norm(rel[inside], axis=1)))
            v_r     = VESSEL_RADII.get(vname, 0.0)
            cl_wall = max(cl_c - v_r, 0.0)
            risk    = "CRITICAL" if cl_wall < OAR_MIN_CLEARANCE_M else "HIGH"
            nr_idx  = int(np.argmin(np.linalg.norm(rel, axis=1)))
            oars.append({
                "vessel":        vname,
                "points_inside": n_in,
                "closest_mm":    cl_c * 1000,
                "wall_clear_mm": cl_wall * 1000,
                "risk":          risk,
                "nearest_pt":    pts[nr_idx],
            })
    return oars


# ─────────────────────────────────────────────────────────────────────────────
# OAR-SAFE REGIME SELECTOR
# ─────────────────────────────────────────────────────────────────────────────

def select_regime_oar_safe(tumor_diam_cm, max_loss_pct,
                            centroid_dists_m, vnames_present,
                            margin_cm=0.5,
                            oar_min_m=OAR_MIN_CLEARANCE_M):
    lf      = max_loss_pct / 100.0
    raw_req = (tumor_diam_cm + margin_cm) / max(1.0 - lf, 0.01)

    def min_wall_clearance(zone_diam_cm):
        zone_r = (zone_diam_cm / 2.0) / 100.0
        return [(vn,
                 centroid_dists_m.get(vn, 999.) - VESSEL_RADII.get(vn, 0.) - zone_r)
                for vn in vnames_present]

    safe_cands = []
    for pw, ts, vol, fwd, diam in ABLATION_TABLE:
        if diam < raw_req:
            continue
        clrs   = min_wall_clearance(diam)
        min_cl = min(c for _, c in clrs)
        if min_cl >= oar_min_m:
            safe_cands.append((pw, ts, vol, fwd, diam, min_cl, clrs))

    constrained = len(safe_cands) == 0

    if not constrained:
        safe_cands.sort(key=lambda x: (x[0], x[1], -x[5]))
        best = safe_cands[0]
        rec  = best[:5]
        alts = [r[:5] for r in safe_cands[1:3]]
        cr   = [{"vessel": vn, "wall_clear_mm": cl * 1000}
                for vn, cl in best[6]]
    else:
        print("\n  ⚠  No regime satisfies BOTH coverage AND OAR clearance.")
        print("     Returning best-available. Consider repositioning or staged treatment.\n")
        coverage = [
            (pw, ts, vol, fwd, diam,
             min(c for _, c in min_wall_clearance(diam)),
             min_wall_clearance(diam))
            for pw, ts, vol, fwd, diam in ABLATION_TABLE
            if diam >= raw_req
        ]
        if not coverage:
            coverage = [
                (pw, ts, vol, fwd, diam,
                 min(c for _, c in min_wall_clearance(diam)),
                 min_wall_clearance(diam))
                for pw, ts, vol, fwd, diam in ABLATION_TABLE
            ]
        coverage.sort(key=lambda x: (-x[5], x[0], x[1]))
        best = coverage[0]
        rec  = best[:5]
        alts = [r[:5] for r in coverage[1:3]]
        cr   = [{"vessel": vn, "wall_clear_mm": cl * 1000}
                for vn, cl in best[6]]

    return rec, alts, raw_req, constrained, cr


# ─────────────────────────────────────────────────────────────────────────────
#  ██████╗  ABLATION SAFETY INDEX (ASI)
#  ██╔══██╗
#  ███████║  Composite risk score  0–100
#  ██╔══██╝  100 = perfectly safe,  0 = extremely high risk
#  ██║  ██║
# ─────────────────────────────────────────────────────────────────────────────

def compute_asi(per_vessel_hs, clearance_report, tumor_diam_cm,
                rec_diam_cm, ray_losses, constrained):
    """
    Returns a dict with:
      asi          – overall score 0-100  (higher = safer)
      hss_score    – heat sink severity sub-score 0-100
      ocm_score    – OAR clearance margin sub-score 0-100
      cc_score     – coverage confidence sub-score 0-100
      dra_score    – directional risk asymmetry sub-score 0-100
      risk_label   – "LOW" / "MODERATE" / "HIGH" / "CRITICAL"
      interpretation – plain-English one-liner
    """

    # ── 1. Heat Sink Severity (HSS) ──────────────────────────────────
    # Worst-case energy loss across all vessels.
    # 0% loss → HSS=100, 50%+ loss → HSS=0
    max_loss_pct = max(hs["loss_pct"] for hs in per_vessel_hs.values())
    hss_score    = float(np.clip(100.0 * (1.0 - max_loss_pct / 50.0), 0, 100))

    # ── 2. OAR Clearance Margin (OCM) ────────────────────────────────
    # Best clearance out of all vessels vs the 5 mm safety threshold.
    # ≥20 mm clearance → OCM=100, <0 mm → OCM=0
    if clearance_report:
        min_clear_mm = min(cr["wall_clear_mm"] for cr in clearance_report)
    else:
        min_clear_mm = 20.0   # no vessels close by → full score
    ocm_score = float(np.clip(100.0 * min_clear_mm / 20.0, 0, 100))

    # ── 3. Coverage Confidence (CC) ──────────────────────────────────
    # How much margin does the recommended zone give beyond raw tumour size?
    # margin ≥ 10 mm → CC=100,  margin ≤ 0 → CC=0
    margin_cm = rec_diam_cm - tumor_diam_cm        # cm
    margin_mm = margin_cm * 10.0                   # mm
    cc_score  = float(np.clip(100.0 * margin_mm / 10.0, 0, 100))

    # Penalise constrained cases (had to sacrifice coverage or OAR)
    if constrained:
        cc_score *= 0.60

    # ── 4. Directional Risk Asymmetry (DRA) ──────────────────────────
    # Spread between max and min ray heat-loss.
    # Low spread → low directional risk → high score.
    # Spread ≥ 30% → DRA=0,  spread = 0% → DRA=100
    if len(ray_losses) > 1:
        spread    = float(np.max(ray_losses) - np.min(ray_losses))
        dra_score = float(np.clip(100.0 * (1.0 - spread / 30.0), 0, 100))
    else:
        dra_score = 50.0

    # ── Weighted composite ────────────────────────────────────────────
    w   = ASI_WEIGHTS
    asi = (w["hss"] * hss_score +
           w["ocm"] * ocm_score +
           w["cc"]  * cc_score  +
           w["dra"] * dra_score)

    # ── Risk label ────────────────────────────────────────────────────
    if   asi >= 75: risk_label = "LOW"
    elif asi >= 50: risk_label = "MODERATE"
    elif asi >= 30: risk_label = "HIGH"
    else:           risk_label = "CRITICAL"

    # ── Interpretation ────────────────────────────────────────────────
    interpretation = {
        "LOW":      "Ablation expected to achieve complete coverage with low risk.",
        "MODERATE": "Vessel proximity may reduce zone size; monitor margins.",
        "HIGH":     "Significant heat sink detected; consider power escalation or repositioning.",
        "CRITICAL": "Ablation zone compromised — staged treatment or repositioning strongly advised.",
    }[risk_label]

    return {
        "asi":            round(asi, 1),
        "hss_score":      round(hss_score, 1),
        "ocm_score":      round(ocm_score, 1),
        "cc_score":       round(cc_score, 1),
        "dra_score":      round(dra_score, 1),
        "risk_label":     risk_label,
        "max_loss_pct":   round(max_loss_pct, 2),
        "min_clear_mm":   round(min_clear_mm, 1),
        "margin_mm":      round(margin_mm, 1),
        "spread_pct":     round(float(np.max(ray_losses) - np.min(ray_losses))
                                if len(ray_losses) > 1 else 0.0, 2),
        "interpretation": interpretation,
    }

def print_asi(asi):
    bar_len = 40
    filled  = int(round(asi["asi"] / 100.0 * bar_len))
    bar_col = {"LOW":"🟢","MODERATE":"🟡","HIGH":"🟠","CRITICAL":"🔴"}[asi["risk_label"]]
    bar     = bar_col * filled + "⬜" * (bar_len - filled)

    print("\n" + "═"*70)
    print("  ABLATION SAFETY INDEX (ASI)")
    print("═"*70)
    print(f"  Overall ASI : {asi['asi']:>5.1f} / 100   [{asi['risk_label']}]")
    print(f"  {bar}")
    print(f"\n  Sub-scores  (weights in parentheses):")
    print(f"  {'Heat Sink Severity':<32} HSS = {asi['hss_score']:>5.1f}  (w={ASI_WEIGHTS['hss']:.2f})"
          f"   [max loss {asi['max_loss_pct']:.1f}%]")
    print(f"  {'OAR Clearance Margin':<32} OCM = {asi['ocm_score']:>5.1f}  (w={ASI_WEIGHTS['ocm']:.2f})"
          f"   [min wall clear {asi['min_clear_mm']:.1f} mm]")
    print(f"  {'Coverage Confidence':<32}  CC = {asi['cc_score']:>5.1f}  (w={ASI_WEIGHTS['cc']:.2f})"
          f"   [margin {asi['margin_mm']:.1f} mm]")
    print(f"  {'Directional Risk Asymmetry':<32} DRA = {asi['dra_score']:>5.1f}  (w={ASI_WEIGHTS['dra']:.2f})"
          f"   [spread {asi['spread_pct']:.1f}%]")
    print(f"\n  ▶  {asi['interpretation']}")
    print("═"*70)


# ─────────────────────────────────────────────────────────────────────────────
# BLOOD PARTICLE SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

class VesselParticleSystem:
    def __init__(self, vessel, vessel_name, n_particles=80):
        pts    = np.array(vessel.points)
        D      = VESSEL_DIAMETERS[vessel_name]
        R      = D / 2.0
        u_mean = VESSEL_VELOCITIES[vessel_name]
        Re     = (RHO_B * u_mean * D) / MU_B
        centred = pts - pts.mean(axis=0)
        _, _, vt = np.linalg.svd(
            centred[:min(5000, len(centred))], full_matrices=False)
        self.flow_dir = vt[0]
        self.origin   = pts.mean(axis=0)
        proj   = centred.dot(self.flow_dir)
        self.L = max(float(proj.max() - proj.min()), 0.02)
        idx    = np.random.choice(len(pts), min(n_particles, len(pts)),
                                  replace=False)
        spts   = pts[idx]
        rel    = spts - self.origin
        axial_c  = np.outer(rel.dot(self.flow_dir), self.flow_dir)
        perp     = rel - axial_c
        r_vals   = np.linalg.norm(perp, axis=1)
        self.r_norm = np.clip(r_vals / (R + 1e-9), 0.0, 1.0)
        if Re < 2300:
            self.u_local = u_mean * 2.0 * (1.0 - self.r_norm**2)
        else:
            u_max = u_mean * (8/7) * (9/8)
            self.u_local = u_max * (1.0 - self.r_norm)**(1.0/7.0)
        self.phase    = np.random.uniform(0.0, self.L, len(idx))
        axial_pos_i   = rel.dot(self.flow_dir)
        self.base_pts = spts - np.outer(axial_pos_i, self.flow_dir)
        self.vessel_name = vessel_name
        self.u_mean      = u_mean
        self.Re          = Re
        self.n           = len(idx)
        self.speed_scale = u_mean / 500.0

    def update(self, t):
        axial = (self.phase + self.speed_scale * t) % self.L
        return self.base_pts + np.outer(axial, self.flow_dir), self.u_local


# ─────────────────────────────────────────────────────────────────────────────
# ELLIPSOID (ablation zone)
# ─────────────────────────────────────────────────────────────────────────────

def make_ellipsoid(centroid, fwd_m, diam_m, needle_dir=None):
    if fwd_m < 1e-4 or diam_m < 1e-4:
        return None
    a   = diam_m / 2.0
    c   = fwd_m  / 2.0
    ell = pv.ParametricEllipsoid(xradius=a, yradius=a, zradius=c,
                                  u_res=30, v_res=30, w_res=10)
    if needle_dir is not None:
        n = np.array(needle_dir, dtype=float)
        n /= np.linalg.norm(n) + 1e-9
        z    = np.array([0., 0., 1.])
        axis = np.cross(z, n)
        an   = np.linalg.norm(axis)
        if an > 1e-6:
            axis  /= an
            angle  = np.degrees(np.arccos(np.clip(np.dot(z, n), -1, 1)))
            ell    = ell.rotate_vector(axis, angle, inplace=False)
    ell.points += centroid
    r_norm = np.linalg.norm(ell.points - centroid, axis=1) / (max(a, c) + 1e-9)
    ell["Temperature_C"] = T_BLOOD + (T_TISSUE - T_BLOOD) * np.exp(-2.0 * r_norm**2)
    return ell


# ─────────────────────────────────────────────────────────────────────────────
# HEAT FLOW ARROWS
# ─────────────────────────────────────────────────────────────────────────────

def create_heat_flow_arrows(centroid, vessels, vnames, per_vessel_hs, plotter):
    losses  = [hs["loss_pct"] for hs in per_vessel_hs.values()]
    mx, mn  = max(losses), min(losses)
    BASE    = 0.04

    def col(pct):
        t = (pct - mn) / max(mx - mn, 0.01)
        return [2*t, 1.0, 0.0] if t < 0.5 else [1.0, 2*(1-t), 0.0]

    for vname, hs in per_vessel_hs.items():
        vessel  = vessels[vnames.index(vname)]
        pts     = np.array(vessel.points)
        _, idx  = cKDTree(pts).query(centroid, k=1)
        raw     = pts[idx] - centroid
        dist    = np.linalg.norm(raw)
        if dist < 1e-6:
            continue
        unit    = raw / dist
        arr_len = max(BASE * hs["loss_pct"] / max(mx, 1.), 0.005)
        plotter.add_mesh(
            pv.Arrow(start=centroid, direction=unit, scale=arr_len,
                     tip_length=0.3, tip_radius=0.05, shaft_radius=0.02),
            color=col(hs["loss_pct"]), opacity=0.95)
        lp = centroid + unit * arr_len * 1.2
        plotter.add_point_labels(
            pv.PolyData([lp]),
            [f"{vname.replace('_',' ')}\n{hs['loss_pct']:.2f}% "
             f"[{hs['flow_regime'][0]}]\nQ={hs['Q_loss_W']:.3f}W"],
            font_size=9, text_color=col(hs["loss_pct"]),
            point_size=1, always_visible=True, shape_opacity=0.0)
    plotter.add_text("Heat Flow:  Green=Low  Yellow=Mid  Red=High",
                     position="lower_right", font_size=9, color="white")


# ─────────────────────────────────────────────────────────────────────────────
# STAGED PLAN
# ─────────────────────────────────────────────────────────────────────────────

def compute_staged_plan(centroid, needle_dir, centroid_dists, oar_list):
    if oar_list:
        vn_oar = min(oar_list, key=lambda o: o["wall_clear_mm"])["vessel"]
    else:
        vn_oar = min(centroid_dists, key=centroid_dists.get)
    wall_dist_m   = centroid_dists[vn_oar] - VESSEL_RADII.get(vn_oar, 0.0)
    max_safe_r_m  = max(wall_dist_m - OAR_MIN_CLEARANCE_M - 2e-3, 0.005)
    max_safe_d_cm = max_safe_r_m * 2 * 100
    valid = sorted([r for r in ABLATION_TABLE if r[4] <= max_safe_d_cm],
                   key=lambda r: (-r[4], r[0], r[1]))
    if not valid:
        valid = sorted(ABLATION_TABLE, key=lambda r: r[4])
    sub      = valid[0]
    fwd_m    = sub[3] / 100.0
    diam_m   = sub[4] / 100.0
    nd       = np.array(needle_dir, dtype=float)
    nd      /= np.linalg.norm(nd) + 1e-9
    overlap  = 0.30 * fwd_m
    offset   = fwd_m - overlap
    return [
        {"centre": centroid - nd * offset / 2.0,
         "fwd_m": fwd_m, "diam_m": diam_m,
         "label": f"Stage 1: {sub[0]:.0f}W×{sub[1]:.0f}s"},
        {"centre": centroid + nd * offset / 2.0,
         "fwd_m": fwd_m, "diam_m": diam_m,
         "label": f"Stage 2: {sub[0]:.0f}W×{sub[1]:.0f}s"},
    ], sub


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — OVERVIEW VISUALISATION (all tumors, no analysis yet)
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

def phase1_overview(surface, vessels, vnames, tumors, metrics):
    """
    Open an interactive 3D window showing ALL tumors labelled with their
    index number, all vessels, and the body surface.
    User closes the window when ready to choose.
    """
    print("\n" + "═"*70)
    print("  PHASE 1 — OVERVIEW  (close window to proceed to tumor selection)")
    print("═"*70)

    plotter = pv.Plotter(window_size=[1400, 900],
                         title="OVERVIEW — All Tumors  |  Close to continue")
    plotter.background_color = "black"

    # Body surface
    plotter.add_mesh(surface, color="lightgray", opacity=0.07,
                     label="Body Surface")

    # Vessels
    for v, vn in zip(vessels, vnames):
        col = VESSEL_COLOR_MAP.get(vn, "gray")
        plotter.add_mesh(v, color=col, opacity=0.60,
                         label=vn.replace("_", " ").title())

    # All tumors — numbered
    for i, (t, m) in enumerate(zip(tumors, metrics)):
        tc    = TUMOR_COLORS[i % len(TUMOR_COLORS)]
        elig  = "✔ ELIGIBLE" if m["eligible"] else "✗ ineligible"
        label = f"T{i+1} ({m['diameter_cm']:.1f}cm {elig})"
        plotter.add_mesh(t, color=tc, opacity=0.80, label=label)
        # centroid sphere + label
        sph = pv.Sphere(radius=0.007, center=m["centroid"])
        plotter.add_mesh(sph, color="white", opacity=0.95)
        plotter.add_point_labels(
            pv.PolyData([m["centroid"] + np.array([0, 0, 0.012])]),
            [f"T{i+1}"],
            font_size=14, text_color=tc,
            point_size=1, always_visible=True, shape_opacity=0.0)

    plotter.add_axes()
    plotter.add_legend(loc="upper right", size=(0.26, 0.50))
    plotter.add_text(
        "PHASE 1 — All Tumors Overview\n"
        "White sphere = centroid  |  Yellow label = tumor number\n"
        "Close this window to select a tumor for analysis",
        position="upper_left", font_size=11, color="white")

    # Colour key for vessels
    plotter.add_text(
        "Vessels:\n"
        "  Red        = Aorta\n"
        "  Med Blue   = Portal vein\n"
        "  Dodger Blue= Hepatic v + IVC\n"
        "  Orange     = Hepatic artery",
        position="lower_left", font_size=10, color="lightgray")

    try:
        plotter.show(auto_close=False)
    except Exception as e:
        print(f"  Visualisation error (non-fatal): {e}")
    finally:
        plotter.close()


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — INTERACTIVE TUMOR SELECTION
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

def phase2_pick_tumor(metrics, vnames):
    """
    Print per-tumor metrics table in the terminal.
    Prompt user to type a tumor number.
    Returns the chosen metric dict.
    """
    print("\n" + "═"*70)
    print("  PHASE 2 — TUMOR SELECTION")
    print("═"*70)
    print(f"\n  {'#':<5} {'Diam(cm)':<11} {'Depth(cm)':<11} "
          f"{'Closest vessel':<18} {'Dist(mm)':<11} {'Eligible?'}")
    print("  " + "─"*70)
    for m in metrics:
        elig = "✔ YES" if m["eligible"] else "✗ NO "
        print(f"  {m['idx']+1:<5} {m['diameter_cm']:<11.2f} "
              f"{m['depth_cm']:<11.2f} "
              f"{m['closest_vessel']:<18} "
              f"{m['min_vessel_m']*1000:<11.1f} {elig}")

    eligible_ids = [m["idx"] + 1 for m in metrics if m["eligible"]]
    if eligible_ids:
        print(f"\n  ✔ Eligible tumors (MWA criteria): {eligible_ids}")
    else:
        print("\n  ⚠  No tumors meet the standard MWA criteria (3–5 cm, depth ≤26 cm).")
        print("     You may still select any tumor for analysis.")

    while True:
        try:
            raw = input(f"\n  ▶  Enter tumor number to analyse [1–{len(metrics)}]: ").strip()
            n   = int(raw)
            if 1 <= n <= len(metrics):
                chosen = metrics[n - 1]
                print(f"\n  ✔ Selected: Tumor {n}  "
                      f"({chosen['diameter_cm']:.2f} cm, "
                      f"depth {chosen['depth_cm']:.2f} cm, "
                      f"closest vessel: {chosen['closest_vessel']} "
                      f"@ {chosen['min_vessel_m']*1000:.1f} mm)")
                return chosen
            else:
                print(f"  ✘ Please enter a number between 1 and {len(metrics)}.")
        except ValueError:
            print("  ✘ Invalid input — enter an integer.")
        except (EOFError, KeyboardInterrupt):
            print("\n  Exiting.")
            sys.exit(0)


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 3 — TREATMENT PLANNING VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

def phase3_visualise(surface, vessels, vnames, tumors, centroids,
                     sel_idx, results, per_vessel_hs,
                     recommended, oar_list, safest_dir,
                     particle_systems, constrained,
                     centroid_dists, asi):

    print("\n🎬 Building treatment-planning visualisation...")

    power_w    = float(recommended[0])
    time_s     = float(recommended[1])
    fwd_m      = recommended[2] / 100.0
    diam_m     = recommended[3] / 100.0
    centroid   = centroids[sel_idx]
    needle_dir = safest_dir

    plotter = pv.Plotter(window_size=[1500, 1000],
                         title=f"Treatment Plan — Tumor {sel_idx+1} | ASI {asi['asi']:.1f} [{asi['risk_label']}]")
    plotter.background_color = "black"

    # Body surface
    plotter.add_mesh(surface, color="lightgray", opacity=0.07,
                     label="Body Surface")

    # Vessels
    for v, vn in zip(vessels, vnames):
        is_oar = any(o["vessel"] == vn for o in oar_list)
        col    = VESSEL_COLOR_MAP.get(vn, "gray")
        if is_oar:
            plotter.add_mesh(v, color="red", opacity=0.90,
                             label=f"⚠ OAR: {vn.replace('_',' ').title()}")
        else:
            plotter.add_mesh(v, color=col, opacity=0.62,
                             label=vn.replace("_", " ").title())

    # Tumors
    for i, t in enumerate(tumors):
        td    = smooth_tumor(t) if i == sel_idx else t
        op    = 0.85 if i == sel_idx else 0.25
        label = (f"Tumor {i+1} [TARGET]" if i == sel_idx else f"Tumor {i+1}")
        plotter.add_mesh(td, color=TUMOR_COLORS[i % len(TUMOR_COLORS)],
                         opacity=op, label=label)

    plotter.add_mesh(pv.Sphere(radius=0.006, center=centroid),
                     color="white", label="Tumour centroid")

    # Heat flow arrows
    create_heat_flow_arrows(centroid, vessels, vnames, per_vessel_hs, plotter)

    # Ray lines — toggled by checkbox
    ray_actor_names = []
    ray_meshes      = []
    if results:
        losses = np.array([r["loss_pct"] for r in results])
        lo, hi = losses.min(), losses.max()
        norm   = (losses - lo) / (hi - lo + 1e-9)
        step   = max(1, len(results) // 80)
        for i in range(0, len(results), step):
            r    = results[i]
            ep   = centroid + r["ray_direction"] * r["path_distance"]
            cv   = norm[i]
            col  = [cv, 0.0, 1.0 - cv]
            name = f"ray_{i}"
            ray_actor_names.append(name)
            ray_meshes.append((pv.Line(centroid, ep), col))
            plotter.add_mesh(pv.Line(centroid, ep), color=col,
                             line_width=2.5, opacity=0.55, name=name)

    # OAR exclusion spheres
    for oar in oar_list:
        if oar["risk"] == "CRITICAL":
            vn      = oar["vessel"]
            excl_r  = VESSEL_RADII.get(vn, 0.005) + OAR_MIN_CLEARANCE_M
            ctr     = oar.get("nearest_pt", centroid)
            sph     = pv.Sphere(radius=excl_r, center=ctr,
                                theta_resolution=20, phi_resolution=20)
            plotter.add_mesh(sph, color="red", opacity=0.18,
                             label=f"OAR exclusion: {vn.replace('_',' ')}")
            plotter.add_mesh(sph, color="red", opacity=0.55,
                             style="wireframe", line_width=1.2)

    # Needle reposition arrow (constrained case)
    if constrained and oar_list:
        closest_oar = min(oar_list, key=lambda o: o["wall_clear_mm"])
        oar_pt      = closest_oar.get("nearest_pt", centroid)
        away        = centroid - oar_pt
        an          = np.linalg.norm(away)
        away        = away / an if an > 1e-6 else np.array([0., 1., 0.])
        plotter.add_mesh(
            pv.Arrow(start=centroid, direction=away, scale=0.04,
                     tip_length=0.25, tip_radius=0.08, shaft_radius=0.03),
            color="cyan", opacity=0.95, label="Suggested needle shift")
        lp       = centroid + away * 0.04 * 1.15
        need_mm  = OAR_MIN_CLEARANCE_M * 1000 - closest_oar["wall_clear_mm"]
        plotter.add_point_labels(
            pv.PolyData([lp]),
            [f"↑ Shift needle\nAway from: {closest_oar['vessel'].replace('_',' ')}\n"
             f"Need: {need_mm:.1f}mm more"],
            font_size=10, text_color="cyan",
            point_size=1, always_visible=True, shape_opacity=0.0)

    # Staged plan
    staged_plan = None
    if constrained:
        staged_plan, sub_row = compute_staged_plan(
            centroid, needle_dir, centroid_dists, oar_list)
        print(f"\n  Staged plan: 2× {sub_row[0]:.0f}W×{sub_row[1]:.0f}s "
              f"({sub_row[4]:.1f}cm diam each)")

    # ── animation helpers ─────────────────────────────────────────────
    SORTED_TABLE = sorted(ABLATION_TABLE, key=lambda r: (r[0], r[1]))
    def get_table_row(pw, t_sec):
        cands = [r for r in SORTED_TABLE if r[0] == round(pw, 0)] or SORTED_TABLE
        return min(cands, key=lambda r: abs(r[1] - t_sec))

    mode_state = {"staged": False, "rays_on": True}

    def clear_dynamic():
        for nm in ["ablation", "ablation_s1", "ablation_s2", "particles", "hud"]:
            try: plotter.remove_actor(nm)
            except: pass

    def draw_single(frac):
        ell = make_ellipsoid(centroid, fwd_m * frac, diam_m * frac, needle_dir)
        if ell is not None:
            plotter.add_mesh(ell, scalars="Temperature_C", cmap="plasma",
                             clim=[T_BLOOD, T_TISSUE], opacity=0.62,
                             name="ablation",
                             scalar_bar_args={
                                 "title": "Temperature (°C)", "n_labels": 5,
                                 "label_font_size": 11, "title_font_size": 12,
                                 "position_x": 0.02, "position_y": 0.25,
                                 "width": 0.08, "height": 0.40, "color": "white"})
        return fwd_m * frac, diam_m * frac

    def draw_staged(frac):
        if staged_plan is None: return 0, 0
        for idx_s, stage in enumerate(staged_plan):
            ell = make_ellipsoid(stage["centre"],
                                 stage["fwd_m"] * frac,
                                 stage["diam_m"] * frac,
                                 needle_dir)
            if ell is not None:
                plotter.add_mesh(ell, scalars="Temperature_C", cmap="plasma",
                                 clim=[T_BLOOD, T_TISSUE], opacity=0.55,
                                 name=f"ablation_s{idx_s+1}",
                                 scalar_bar_args={
                                     "title": "Temperature (°C)", "n_labels": 4,
                                     "label_font_size": 10, "title_font_size": 11,
                                     "position_x": 0.02, "position_y": 0.25,
                                     "width": 0.07, "height": 0.35, "color": "white"})
        s = staged_plan[0]
        return s["fwd_m"], s["diam_m"]

    # ASI colour
    asi_col = {"LOW": "lime", "MODERATE": "yellow",
               "HIGH": "orange", "CRITICAL": "tomato"}[asi["risk_label"]]

    def update(t_val):
        t    = float(t_val)
        frac = min(t / time_s, 1.0)
        clear_dynamic()

        if mode_state["staged"] and staged_plan:
            cur_fwd, cur_diam = draw_staged(frac)
            mode_label = "⚡ STAGED (2× partial)"
            stage_note = (f"  {staged_plan[0]['label']}\n"
                          f"  {staged_plan[1]['label']}\n"
                          f"  each diam={cur_diam*100:.1f}cm")
        else:
            cur_fwd, cur_diam = draw_single(frac)
            mode_label = "● Single zone"
            stage_note = ""

        # Blood particles
        all_pts, all_vel = [], []
        for ps in particle_systems:
            pts, vel = ps.update(t)
            all_pts.append(pts); all_vel.append(vel)
        if all_pts:
            cloud = pv.PolyData(np.vstack(all_pts))
            cloud["blood_velocity_m_s"] = np.concatenate(all_vel)
            plotter.add_mesh(cloud, scalars="blood_velocity_m_s", cmap="coolwarm",
                             clim=[0.0, max(VESSEL_VELOCITIES.values()) * 2.0],
                             point_size=5, render_points_as_spheres=True,
                             name="particles",
                             scalar_bar_args={
                                 "title": "Blood velocity (m/s)", "n_labels": 3,
                                 "label_font_size": 10, "title_font_size": 11,
                                 "position_x": 0.12, "position_y": 0.25,
                                 "width": 0.08, "height": 0.40, "color": "white"})

        row      = get_table_row(power_w, t)
        zone_str = (f"Zone: {cur_fwd*100:.1f}cm × {cur_diam*100:.1f}cm"
                    if frac > 0.01 else "Zone: growing...")

        if constrained:
            worst    = (min(oar_list, key=lambda o: o["wall_clear_mm"])
                        if oar_list else None)
            warn_line = (f"⚠  CONSTRAINED — encroachment\n"
                         f"  {worst['vessel'].replace('_',' ')} wall: "
                         f"{worst['wall_clear_mm']:.1f}mm\n"
                         if worst else "⚠  CONSTRAINED\n")
        else:
            warn_line = "✔  OAR-SAFE regime\n"

        hud = (
            f"{'─'*30}\n"
            f"  ABLATION SAFETY INDEX\n"
            f"  ASI = {asi['asi']:.1f} / 100  [{asi['risk_label']}]\n"
            f"  HSS={asi['hss_score']:.0f}  OCM={asi['ocm_score']:.0f}  "
            f"CC={asi['cc_score']:.0f}  DRA={asi['dra_score']:.0f}\n"
            f"{'─'*30}\n"
            f"{warn_line}"
            f"Mode: {mode_label}\n"
            f"t = {t:.0f}s / {time_s:.0f}s  ({frac*100:.0f}%)\n"
            f"Power: {power_w:.0f} W\n"
            f"{zone_str}\n"
            f"{'─'*30}\n"
            f"Table ({row[1]:.0f}s, {row[0]:.0f}W):\n"
            f"  Vol={row[2]:.2f}cc Fwd={row[3]:.1f}cm D={row[4]:.1f}cm\n"
            + (f"{'─'*30}\nStaged:\n{stage_note}\n" if stage_note else "")
            + f"{'─'*30}\n"
            f"OARs encroached: {len(oar_list)}\n"
            f"{'─'*30}\n"
            f"Ablation: Purple→White=37→90°C\n"
            f"Blood:    Blue=slow  Red=fast\n"
        )

        plotter.add_text(hud, position="lower_left", font_size=9,
                         color=asi_col, name="hud")
        plotter.render()

    # ── play controls ─────────────────────────────────────────────────
    play_state  = {"playing": False, "t": 0.0}
    PLAY_STEP_S = 5.0
    TIMER_MS    = 100

    def toggle_play(flag):
        play_state["playing"] = bool(flag)
        if play_state["playing"]:
            plotter.add_timer_event(max_steps=100000, duration=TIMER_MS,
                                    callback=_timer_tick)

    def _timer_tick(step):
        if not play_state["playing"]: return
        play_state["t"] = (play_state["t"] + PLAY_STEP_S) % (time_s + 1.0)
        slider_widget.GetRepresentation().SetValue(play_state["t"])
        update(play_state["t"])

    slider_widget = plotter.add_slider_widget(
        update, rng=[0.0, time_s], value=0.0,
        title="Ablation Time (s)",
        pointa=(0.22, 0.05), pointb=(0.90, 0.05), style="modern")

    # ── button column ─────────────────────────────────────────────────
    # Play
    plotter.add_checkbox_button_widget(
        toggle_play, value=False, position=(30, 30),
        size=45, border_size=3, color_on="lime", color_off="gray")
    plotter.add_text("▶ Play", position=(80, 38), font_size=11, color="white")

    # Ray toggle
    def toggle_rays(flag):
        mode_state["rays_on"] = bool(flag)
        if flag:
            for name, (lm, col) in zip(ray_actor_names, ray_meshes):
                plotter.add_mesh(lm, color=col, line_width=2.5,
                                 opacity=0.55, name=name)
        else:
            for name in ray_actor_names:
                try: plotter.remove_actor(name)
                except: pass
        plotter.render()

    plotter.add_checkbox_button_widget(
        toggle_rays, value=True, position=(30, 100),
        size=45, border_size=3, color_on="yellow", color_off="dimgray")
    plotter.add_text("◉ Ray lines", position=(80, 108),
                     font_size=11, color="yellow")

    # Staged toggle (constrained only)
    if constrained and staged_plan:
        def toggle_staged(flag):
            mode_state["staged"] = bool(flag)
            update(play_state["t"])

        plotter.add_checkbox_button_widget(
            toggle_staged, value=False, position=(30, 170),
            size=45, border_size=3, color_on="cyan", color_off="dimgray")
        plotter.add_text("⚡ Staged 2×", position=(80, 178),
                         font_size=11, color="cyan")

    plotter.add_legend(loc="upper right", size=(0.26, 0.46))
    title_col = asi_col
    plotter.add_text(
        f"Heat Sink + OAR + Flow  |  {power_w:.0f}W × {time_s:.0f}s"
        + (f"  ⚠ CONSTRAINED" if constrained else "  ✔ OAR-SAFE")
        + f"  |  ASI {asi['asi']:.1f} [{asi['risk_label']}]",
        position="upper_left", font_size=13, color=title_col)
    plotter.add_axes()

    update(0.0)
    try:
        plotter.show(auto_close=False)
    except Exception as e:
        print(f"  Vis error: {e}")
    finally:
        plotter.close()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  HEAT SINK ANALYSIS v9  —  Integrated + ASI Risk Index")
    print("=" * 70)

    if not os.path.exists(DATASET_BASE):
        print(f"  ✘ Dataset directory not found:\n    {DATASET_BASE}")
        return

    # ── Load meshes ────────────────────────────────────────────────────
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
        print("  ✘ Critical files missing — aborting.")
        return

    # ── Extract tumors ─────────────────────────────────────────────────
    tumors  = extract_tumors(tumor_mesh)
    metrics = tumor_metrics(tumors, surface, vessels, vnames)
    centroids = np.array([m["centroid"] for m in metrics])

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1 — Overview window
    # ═══════════════════════════════════════════════════════════════════
    phase1_overview(surface, vessels, vnames, tumors, metrics)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 2 — User picks tumor
    # ═══════════════════════════════════════════════════════════════════
    sel  = phase2_pick_tumor(metrics, vnames)
    sel_idx  = sel["idx"]
    centroid = sel["centroid"]
    sel_diam = sel["diameter_cm"]

    # ── Per-vessel distances from chosen centroid ──────────────────────
    centroid_dists = {}
    for i, v in enumerate(vessels):
        centroid_dists[vnames[i]] = float(
            cKDTree(np.array(v.points)).query(centroid, k=1)[0])

    # ── Heat sink physics ──────────────────────────────────────────────
    POWER_W = 60.0
    TIME_S  = 600.0

    print("\n  Computing per-vessel heat-sink physics...")
    per_vessel_hs = {
        vn: heat_sink_full_physics(centroid_dists[vn], vn, POWER_W, TIME_S)
        for vn in vnames
    }

    print(f"\n  {'Vessel':<18} {'Dist(mm)':<10} {'Regime':<14} "
          f"{'Nu':<8} {'Q_loss(W)':<12} {'Loss%'}")
    print("  " + "─" * 72)
    for vn, hs in per_vessel_hs.items():
        wall_mm = (centroid_dists[vn] - VESSEL_RADII[vn]) * 1000
        print(f"  {vn:<18} {hs['dist_mm']:<10.1f} {hs['flow_regime']:<14} "
              f"{hs['Nu']:<8.1f} {hs['Q_loss_W']:<12.4f} {hs['loss_pct']:.3f}%")

    max_hs_pct = max(hs["loss_pct"] for hs in per_vessel_hs.values())

    # ── Ray tracing ────────────────────────────────────────────────────
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
            seg_d  = {
                vn: ray_segment_dist(centroid, direction, path_d,
                                     v_pts[vi], centroid_dists[vn])
                for vi, vn in enumerate(vnames)
            }
            dom_vn = min(seg_d, key=seg_d.get)
            hs     = heat_sink_full_physics(seg_d[dom_vn], dom_vn, POWER_W, TIME_S)
            hs["ray_direction"] = direction
            hs["path_distance"] = path_d
            results.append(hs)
        except Exception:
            continue

    all_losses = [r["loss_pct"] for r in results]
    sorted_res = sorted(results, key=lambda x: x["loss_pct"], reverse=True)
    safest_dir = (sorted_res[-1]["ray_direction"]
                  if results else np.array([0., 0., 1.]))
    print(f"  {len(results)} rays | loss range "
          f"{np.min(all_losses):.2f}% – {np.max(all_losses):.2f}%")

    # ── Regime selection ───────────────────────────────────────────────
    rec, alts, raw_req, constrained, clearance_report = select_regime_oar_safe(
        sel_diam, max_hs_pct, centroid_dists, vnames, margin_cm=0.5)

    print(f"\n  {'⚠  CONSTRAINED' if constrained else '✔  OAR-SAFE'} Regime: "
          f"{rec[0]:.0f}W × {rec[1]:.0f}s  "
          f"diam={rec[4]:.2f}cm  (need {raw_req:.2f}cm)")

    print(f"\n  Vessel-wall clearances for chosen regime ({rec[4]:.2f}cm diam):")
    print(f"  {'Vessel':<20} {'Wall Clear (mm)'}")
    print("  " + "─" * 38)
    for cr in clearance_report:
        flag = "  ✔" if cr["wall_clear_mm"] >= OAR_MIN_CLEARANCE_M * 1000 else "  ✗ ENCROACH"
        print(f"  {cr['vessel']:<20} {cr['wall_clear_mm']:>8.1f} mm{flag}")

    # ── OAR identification ─────────────────────────────────────────────
    oar_list = identify_oars(centroid, vessels, vnames,
                             rec[3], rec[4], safest_dir)
    print(f"\n  OARs encroached: {len(oar_list)}")
    for o in oar_list:
        print(f"    {o['vessel']}  {o['points_inside']} pts  "
              f"wall={o['wall_clear_mm']:.1f}mm  [{o['risk']}]")

    # ═══════════════════════════════════════════════════════════════════
    # ABLATION SAFETY INDEX
    # ═══════════════════════════════════════════════════════════════════
    asi = compute_asi(
        per_vessel_hs   = per_vessel_hs,
        clearance_report= clearance_report,
        tumor_diam_cm   = sel_diam,
        rec_diam_cm     = rec[4],
        ray_losses      = all_losses,
        constrained     = constrained,
    )
    print_asi(asi)

    # ── Particle systems ───────────────────────────────────────────────
    print("\n  Building blood particle systems...")
    particle_systems = []
    for v, vn in zip(vessels, vnames):
        ps = VesselParticleSystem(v, vn, n_particles=80)
        particle_systems.append(ps)
        print(f"   {vn}: {ps.n} particles, Re={ps.Re:.0f} "
              f"({'Laminar' if ps.Re < 2300 else 'Turbulent/Transition'}), "
              f"L={ps.L*100:.1f}cm")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 3 — Treatment planning visualisation
    # ═══════════════════════════════════════════════════════════════════
    phase3_visualise(
        surface, vessels, vnames, tumors, centroids,
        sel_idx, results, per_vessel_hs,
        rec, oar_list, safest_dir, particle_systems,
        constrained, centroid_dists, asi)

    print("\n  ✔  Complete!")
    return results, asi


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  Stopped.")
    except Exception as e:
        print(f"\n  Error: {e}")
        raise
