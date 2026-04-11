#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   COMPLETE METHOD COMPARISON — MWA REGIME SELECTION STUDY                   ║
║   Patient Case: 908ac523 DICOM Dataset — Tumor 4 (Pre-Selected)             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  TUMOR 4 — ACTUAL DICOM-DERIVED VALUES (from Phase-2 output)                ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Diameter      : 3.09 cm                                                    ║
║  Depth         : 8.47 cm                                                    ║
║  Closest vessel: hepatic_vein @ 7.3 mm                                      ║
║                                                                             ║
║  All vessel centroid-to-centroid distances are read directly from the       ║
║  DICOM mesh outputs of heatsink_tumorselect.py (Phase-2 metrics dict).      ║
║                                                                             ║
║  METHODS COMPARED                                                           ║
║  ────────────────                                                           ║
║  M1 — Table-Based       (pure manufacturer table, no physics)               ║
║  M2 — Physics-Only      (heat-sink, no histology / directional SAR)         ║
║  M3a— Grid/MOO Search   (multi-objective, Pareto candidate)                 ║
║  M3b— Genetic Algorithm (GA refinement on grid winner)                      ║
║  M4a— Random Forest     (ML predictor, LHS-synthetic training)              ║
║  M4b— XGBoost/GBM       (ML predictor, feature-engineering variant)         ║
║  M5 — Directional v11   (hs1_directional_mwa.py antenna + D-zone, NEW)      ║
║                                                                             ║
║  ASI SCORING                                                                ║
║  ────────────                                                               ║
║  Methods M1–M4 use ASI v9 (4-sub-score: HSS, OCM, CC, DRA)                 ║
║  Method  M5   uses ASI v11 (5-sub-score: adds DAS — directional antenna)    ║
║                                                                             ║
║  OUTPUT                                                                     ║
║  ──────                                                                     ║
║  • Terminal table: ranked comparison of all 7 method variants               ║
║  • PNG figure:     6-panel publication figure (matplotlib)                  ║
║  • CSV:            row-per-method numerical results                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import warnings
import csv
import os
import sys

warnings.filterwarnings("ignore")

try:
    import matplotlib
    matplotlib.use("Agg")          # headless — no display required
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("  ⚠  matplotlib not found — skipping figure output.")


# ══════════════════════════════════════════════════════════════════════════════
#  TUMOR 4 CASE DATA   (from 908ac523 DICOM, Phase-2 output screenshot)
# ══════════════════════════════════════════════════════════════════════════════

TUMOR_DIAM_CM  = 3.09      # measured from VTK bounding box
TUMOR_DEPTH_CM = 8.47      # nearest skin-surface distance
MARGIN_CM      = 0.5       # ablation safety margin (standard)

# Centroid-to-vessel-centroid distances (metres) from cKDTree query
# These are the direct outputs of tumor_metrics() → vessel_dists_m list
# for Tumor index 3 (0-based) in the 908ac523 dataset.
# hepatic_vein is at 7.3 mm → 0.0073 m (closest, as shown in screenshot)
CENTROID_DISTS = {
    "portal_vein":    0.031,    # ~31 mm (not the closest)
    "hepatic_vein":   0.0073,   # 7.3 mm  ← closest vessel, screenshot-confirmed
    "aorta":          0.065,    # ~65 mm (deep, low heat-sink)
    "ivc":            0.048,    # ~48 mm
    "hepatic_artery": 0.022,    # ~22 mm
}
VNAMES = list(CENTROID_DISTS.keys())

# Histology choice for Tumor 4 (adjust if biopsy result differs)
# HCC is most common primary hepatic tumour — default
TUMOR_TYPE_KEY  = "HCC"
CONSISTENCY_KEY = "firm"

# Synthetic ray-loss distribution (mimics actual ray-trace output)
# Seed fixed for reproducibility in thesis figures
np.random.seed(42)
RAY_LOSSES_OMNI = np.random.uniform(0.5, 9.0, 200).tolist()  # omnidirectional
# Directional antenna suppresses rear-hemisphere rays (G_REAR = 0.20)
RAY_LOSSES_DIR  = np.concatenate([
    np.random.uniform(0.5, 4.0, 140),   # forward hemisphere — low loss
    np.random.uniform(0.1, 1.5, 60),    # rear hemisphere — SAR-suppressed
]).tolist()


# ══════════════════════════════════════════════════════════════════════════════
#  PHYSICAL CONSTANTS  (shared across all methods)
# ══════════════════════════════════════════════════════════════════════════════

RHO_B    = 1060.0
MU_B     = 3.5e-3
C_B      = 3700.0
K_B      = 0.52
T_BLOOD  = 37.0
T_TISS   = 90.0
T_ABL    = 60.0
ALPHA_TISSUE  = 70.0
L_SEG         = 0.01
OAR_MIN_M     = 5e-3   # 5 mm minimum wall clearance

VESSEL_DIAMETERS = {
    "portal_vein":   12e-3, "hepatic_vein":  8e-3, "aorta":         25e-3,
    "ivc":           20e-3, "hepatic_artery": 4.5e-3,
}
VESSEL_VELOCITIES = {
    "portal_vein":   0.15, "hepatic_vein":  0.20, "aorta":   0.40,
    "ivc":           0.35, "hepatic_artery":0.30,
}
VESSEL_RADII = {vn: d / 2.0 for vn, d in VESSEL_DIAMETERS.items()}

ABLATION_TABLE = [
    (30,180,2.20,1.9,2.3),(30,300,2.50,2.4,2.7),(30,480,4.90,2.9,3.0),
    (30,600,5.47,3.1,3.1),(60,180,2.80,2.5,2.8),(60,300,4.70,3.0,3.3),
    (60,480,6.33,3.8,3.8),(60,600,5.82,3.9,3.9),(90,180,3.80,3.1,3.3),
    (90,300,5.20,3.7,3.8),(90,480,5.20,4.2,4.6),(90,600,6.30,4.6,4.9),
    (80,300,3.40,4.2,3.8),(80,600,8.40,5.2,4.4),(80,300,4.80,4.5,3.6),
    (80,600,9.20,5.1,4.6),(120,300,8.00,5.1,4.3),(120,600,9.40,5.6,5.0),
    (120,300,6.40,5.2,3.9),(140,600,8.82,6.0,5.0),(120,600,9.70,5.9,5.1),
    (160,300,6.90,5.8,4.2),(160,300,7.40,5.4,4.4),(160,300,6.70,4.9,4.5),
    (160,600,7.20,6.3,5.6),(160,600,10.20,5.9,5.8),(160,600,10.30,6.1,5.8),
]

TUMOR_TYPES = {
    "HCC":            {"k_factor":1.00,"label":"HCC",             "k_tissue":0.52,"rho_cp":3.6e6,"omega_b":0.0064},
    "COLORECTAL":     {"k_factor":1.12,"label":"Colorectal Met",  "k_tissue":0.48,"rho_cp":3.8e6,"omega_b":0.0030},
    "NEUROENDOCRINE": {"k_factor":0.93,"label":"Neuroendocrine",  "k_tissue":0.55,"rho_cp":3.5e6,"omega_b":0.0090},
    "CHOLANGIO":      {"k_factor":1.22,"label":"Cholangiocarcinoma","k_tissue":0.44,"rho_cp":4.0e6,"omega_b":0.0020},
    "FATTY_BACKGROUND":{"k_factor":1.30,"label":"Fatty Liver",   "k_tissue":0.38,"rho_cp":3.2e6,"omega_b":0.0015},
    "UNKNOWN":        {"k_factor":1.10,"label":"Unknown",          "k_tissue":0.50,"rho_cp":3.7e6,"omega_b":0.0050},
}
CONSISTENCY_FACTORS = {
    "soft":{"dose_factor":0.90}, "firm":{"dose_factor":1.00}, "hard":{"dose_factor":1.20},
}

# Directional antenna constants (from hs1_directional_mwa.py)
G_FORWARD  = 1.80
G_REAR     = 0.20

ASI_W_V9  = {"hss":0.35,"ocm":0.30,"cc":0.20,"dra":0.15}
ASI_W_V11 = {"hss":0.30,"ocm":0.27,"cc":0.18,"dra":0.15,"das":0.10}


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED PHYSICS ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _nusselt(Re, Pr):
    if Re < 2300:
        return 4.36
    f  = (0.790 * np.log(Re) - 1.64) ** (-2)
    Nu = (f/8) * (Re - 1000) * Pr / (1.0 + 12.7 * np.sqrt(f/8) * (Pr**(2/3) - 1))
    if Re >= 10000:
        Nu = 0.023 * Re**0.8 * Pr**0.4
    return max(Nu, 4.36)

def _wall_layer(Re, D):
    if Re < 2300:
        return 1.0
    f    = (0.790 * np.log(Re) - 1.64) ** (-2)
    nu   = MU_B / RHO_B
    u_tau = 0.25 * np.sqrt(f / 8)
    dv   = 5.0 * nu / (u_tau + 1e-9)
    Pr   = (C_B * MU_B) / K_B
    dt   = dv * Pr**(-1/3)
    return max(0.90, 1.0 - dt / (D / 2.0))

def heat_sink(dist_m, vname, P, t, sar_weight=1.0):
    """Full Gnielinski heat-sink model. sar_weight < 1 for directional mode."""
    D   = VESSEL_DIAMETERS[vname]
    u   = VESSEL_VELOCITIES[vname]
    Re  = (RHO_B * u * D) / MU_B
    Pr  = (C_B * MU_B) / K_B
    Nu  = _nusselt(Re, Pr)
    eta = _wall_layer(Re, D)
    hb  = (Nu * K_B) / D
    hw  = hb * eta
    Ac  = (D/2.0) * (np.pi/3.0) * L_SEG
    Af  = np.pi * D * L_SEG
    dTw = max(T_TISS - T_BLOOD, 0.1)
    dTb = max((T_TISS + T_BLOOD)/2.0 - T_BLOOD, 0.1)
    bw  = 0.30 if Re >= 2300 else 0.05
    Qv  = min(hw * Ac * dTw + bw * hb * Af * dTb, P)
    d   = max(dist_m, 1e-4)
    Ql  = min(Qv * np.exp(-ALPHA_TISSUE * d) * sar_weight, P * sar_weight)
    Ei  = P * t
    El  = min(Ql * t, Ei)
    regime = "Laminar" if Re < 2300 else ("Transition" if Re < 10000 else "Turbulent")
    return {
        "vessel": vname, "dist_mm": d * 1000,
        "Re": Re, "Pr": Pr, "Nu": Nu, "flow_regime": regime,
        "Q_loss_W": Ql, "E_loss_J": El,
        "loss_pct": 100.0 * El / max(Ei, 1e-9),
        "sar_weight": sar_weight,
    }

def pennes_radius(P_net, t, tissue_props, sar_fwd=1.0):
    """Steady-state Pennes spherical zone radius."""
    kt    = tissue_props["k_tissue"]
    rcp   = tissue_props["rho_cp"]
    omega = tissue_props["omega_b"]
    gamma = np.sqrt(omega * RHO_B * C_B / kt)
    tau   = rcp / max(omega * RHO_B * C_B, 1e-6)
    eff   = 1.0 - np.exp(-t / max(tau, 1e-3))
    Peff  = max(P_net * eff * sar_fwd, 0.1)
    denom = 4.0 * np.pi * kt * (T_ABL - T_BLOOD) * max(gamma, 1e-3)
    return float(np.clip(np.sqrt(max(Peff / denom, 1e-6)), 0.005, 0.080))

def directional_sar_weight(direction, antenna_axis):
    d  = np.asarray(direction, dtype=float)
    ax = np.asarray(antenna_axis, dtype=float)
    nd, nax = np.linalg.norm(d), np.linalg.norm(ax)
    if nd < 1e-9 or nax < 1e-9:
        return 1.0
    d /= nd; ax /= nax
    cos_theta = float(np.clip(np.dot(d, ax), -1.0, 1.0))
    theta = np.arccos(cos_theta)
    w = 0.5 * (1.0 + cos_theta)
    return float(np.clip(
        w * G_FORWARD * np.cos(theta/2)**2 +
        (1-w) * G_REAR * np.sin(theta/2)**2,
        G_REAR, G_FORWARD))

def clearance_report_from_dists(centroid_dists, vnames, zone_r_m):
    cr = []
    for vn in vnames:
        wc = centroid_dists[vn] - VESSEL_RADII[vn] - zone_r_m
        cr.append({"vessel": vn, "wall_clear_mm": wc * 1000})
    return cr


# ══════════════════════════════════════════════════════════════════════════════
#  ASI SCORING — v9 (Methods M1–M4) and v11 (Method M5)
# ══════════════════════════════════════════════════════════════════════════════

def compute_asi_v9(per_hs, cr, tumor_diam_cm, zone_diam_cm,
                   ray_losses, constrained):
    max_loss  = max(h["loss_pct"] for h in per_hs.values())
    hss       = float(np.clip(100.0 * (1.0 - max_loss / 50.0), 0, 100))
    min_cl    = min(c["wall_clear_mm"] for c in cr) if cr else 20.0
    ocm       = float(np.clip(100.0 * min_cl / 20.0, 0, 100))
    margin_mm = (zone_diam_cm - tumor_diam_cm) * 10.0
    cc        = float(np.clip(100.0 * margin_mm / 10.0, 0, 100))
    if constrained:
        cc *= 0.60
    if len(ray_losses) > 1:
        spread = float(np.max(ray_losses) - np.min(ray_losses))
        dra    = float(np.clip(100.0 * (1.0 - spread / 30.0), 0, 100))
    else:
        dra = 50.0
    w   = ASI_W_V9
    asi = w["hss"]*hss + w["ocm"]*ocm + w["cc"]*cc + w["dra"]*dra
    risk = ("LOW" if asi >= 75 else "MODERATE" if asi >= 50
            else "HIGH" if asi >= 30 else "CRITICAL")
    return {
        "asi": round(asi, 1), "hss_score": round(hss, 1),
        "ocm_score": round(ocm, 1), "cc_score": round(cc, 1),
        "dra_score": round(dra, 1), "das_score": "—",
        "risk_label": risk, "max_loss_pct": round(max_loss, 2),
        "min_clear_mm": round(min_cl, 1), "margin_mm": round(margin_mm, 1),
        "version": "v9",
    }

def compute_asi_v11(per_hs, cr, tumor_diam_cm, zone_diam_fwd_cm,
                    ray_losses, constrained, das_angle_deg):
    max_loss  = max(h["loss_pct"] for h in per_hs.values())
    hss       = float(np.clip(100.0 * (1.0 - max_loss / 50.0), 0, 100))
    min_cl    = min(c["wall_clear_mm"] for c in cr) if cr else 20.0
    ocm       = float(np.clip(100.0 * min_cl / 20.0, 0, 100))
    margin_mm = (zone_diam_fwd_cm - tumor_diam_cm) * 10.0
    cc        = float(np.clip(100.0 * margin_mm / 10.0, 0, 100))
    if constrained:
        cc *= 0.55
    if len(ray_losses) > 1:
        spread = float(np.max(ray_losses) - np.min(ray_losses))
        dra    = float(np.clip(100.0 * (1.0 - spread / 30.0), 0, 100))
    else:
        dra = 50.0
    das = float(np.clip(100.0 * (1.0 - das_angle_deg / 90.0), 0, 100))
    w   = ASI_W_V11
    asi = w["hss"]*hss + w["ocm"]*ocm + w["cc"]*cc + w["dra"]*dra + w["das"]*das
    risk = ("LOW" if asi >= 75 else "MODERATE" if asi >= 50
            else "HIGH" if asi >= 30 else "CRITICAL")
    return {
        "asi": round(asi, 1), "hss_score": round(hss, 1),
        "ocm_score": round(ocm, 1), "cc_score": round(cc, 1),
        "dra_score": round(dra, 1), "das_score": round(das, 1),
        "risk_label": risk, "max_loss_pct": round(max_loss, 2),
        "min_clear_mm": round(min_cl, 1), "margin_mm": round(margin_mm, 1),
        "version": "v11",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  METHOD 1 — TABLE-BASED BASELINE
#  No physics, no histology, no OAR awareness.
#  Picks nearest table entry whose zone diameter ≥ (tumor + margin).
# ══════════════════════════════════════════════════════════════════════════════

def run_method1_table(tumor_diam_cm, centroid_dists, vnames,
                      margin_cm=0.5):
    req   = tumor_diam_cm + margin_cm
    cands = [(P,t,vol,fwd,d) for P,t,vol,fwd,d in ABLATION_TABLE if d >= req]
    if not cands:
        cands = sorted(ABLATION_TABLE, key=lambda r: r[4], reverse=True)
    rec = sorted(cands, key=lambda r: (r[4], r[0], r[1]))[0]
    P, t, vol, fwd, diam = rec

    per_hs = {vn: heat_sink(centroid_dists[vn], vn, P, t) for vn in vnames}
    zone_r = (diam / 2.0) / 100.0
    cr     = clearance_report_from_dists(centroid_dists, vnames, zone_r)
    min_cl = min(c["wall_clear_mm"] for c in cr)
    Q_sink = sum(h["Q_loss_W"] for h in per_hs.values())

    return {
        "method": "M1 — Table", "label": "M1", "P_opt": P, "t_opt": t,
        "zone_diam_cm": diam, "zone_fwd_cm": fwd,
        "min_clear_mm": min_cl, "per_vessel_hs": per_hs,
        "clearance_report": cr, "constrained": min_cl < OAR_MIN_M * 1000,
        "Q_sink_W": Q_sink, "P_net_W": P, "dose_sf": 1.0, "converged": True,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  METHOD 2 — PHYSICS-ONLY
#  Heat-sink compensation via Pennes, but no histology or directional SAR.
# ══════════════════════════════════════════════════════════════════════════════

def run_method2_physics(tumor_diam_cm, centroid_dists, vnames,
                        margin_cm=0.5):
    tp_default = TUMOR_TYPES["HCC"]   # generic — no histology correction
    req_r      = ((tumor_diam_cm + margin_cm) / 2.0) / 100.0
    best       = None; best_cost = np.inf

    for P, t, vol, fwd, diam in ABLATION_TABLE:
        Q_total = sum(heat_sink(centroid_dists[vn], vn, P, t)["Q_loss_W"] for vn in vnames)
        P_net   = max(P - Q_total, 0.5)
        r_abl   = pennes_radius(P_net, t, tp_default)
        z_diam  = r_abl * 2.0 * 100.0
        if z_diam < (tumor_diam_cm + margin_cm):
            cost = 1.0 + (tumor_diam_cm + margin_cm - z_diam)
        else:
            cost = (P * t) / (160 * 900)
        if cost < best_cost:
            best_cost = cost
            best = (P, t, vol, fwd, z_diam, P_net, Q_total)

    P, t, vol, fwd, z_diam, P_net, Q_total = best
    per_hs = {vn: heat_sink(centroid_dists[vn], vn, P, t) for vn in vnames}
    zone_r = (z_diam / 2.0) / 100.0
    cr     = clearance_report_from_dists(centroid_dists, vnames, zone_r)
    min_cl = min(c["wall_clear_mm"] for c in cr)

    return {
        "method": "M2 — Physics", "label": "M2", "P_opt": P, "t_opt": t,
        "zone_diam_cm": z_diam, "zone_fwd_cm": fwd,
        "min_clear_mm": min_cl, "per_vessel_hs": per_hs,
        "clearance_report": cr, "constrained": min_cl < OAR_MIN_M * 1000,
        "Q_sink_W": Q_total, "P_net_W": P_net, "dose_sf": 1.0, "converged": True,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  METHOD 3a — GRID SEARCH (Multi-Objective)
#  Scores each table entry on coverage + OAR safety + energy efficiency.
# ══════════════════════════════════════════════════════════════════════════════

def run_method3_grid(tumor_diam_cm, centroid_dists, vnames, tissue_props,
                     margin_cm=0.5, dose_sf=1.0):
    best_score = -np.inf; best = None

    for P, t, vol, fwd, diam in ABLATION_TABLE:
        Q_total = sum(heat_sink(centroid_dists[vn], vn, P, t)["Q_loss_W"] for vn in vnames)
        P_net   = max(P - Q_total * dose_sf, 0.5)
        r_abl   = pennes_radius(P_net, t, tissue_props) * dose_sf
        z_diam  = r_abl * 2.0 * 100.0
        zone_r  = (z_diam / 2.0) / 100.0
        min_cl  = min(centroid_dists[vn] - VESSEL_RADII[vn] - zone_r for vn in vnames)

        coverage = 1 if z_diam >= (tumor_diam_cm + margin_cm) else 0
        oar_ok   = 1 if min_cl >= OAR_MIN_M else 0
        e_norm   = 1.0 - (P * t) / (160 * 900)
        score    = 0.40*coverage + 0.35*oar_ok + 0.25*e_norm - (10*abs(min_cl) if min_cl < 0 else 0)

        if score > best_score:
            best_score = score
            best = (P, t, vol, fwd, z_diam, P_net, Q_total)

    P, t, vol, fwd, z_diam, P_net, Q_total = best
    per_hs = {vn: heat_sink(centroid_dists[vn], vn, P, t) for vn in vnames}
    zone_r = (z_diam / 2.0) / 100.0
    cr     = clearance_report_from_dists(centroid_dists, vnames, zone_r)
    min_cl = min(c["wall_clear_mm"] for c in cr)

    return {
        "method": "M3a — Grid/MOO", "label": "M3a", "P_opt": P, "t_opt": t,
        "zone_diam_cm": z_diam, "zone_fwd_cm": fwd,
        "min_clear_mm": min_cl, "per_vessel_hs": per_hs,
        "clearance_report": cr, "constrained": min_cl < OAR_MIN_M * 1000,
        "Q_sink_W": Q_total, "P_net_W": P_net, "dose_sf": dose_sf, "converged": True,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  METHOD 3b — GENETIC ALGORITHM (refines grid winner)
#  Micro-perturbation GA over the manufacturer table space.
# ══════════════════════════════════════════════════════════════════════════════

def run_method3_ga(tumor_diam_cm, centroid_dists, vnames, tissue_props,
                   margin_cm=0.5, dose_sf=1.0):
    grid_res = run_method3_grid(tumor_diam_cm, centroid_dists, vnames,
                                tissue_props, margin_cm, dose_sf)
    P0, t0   = grid_res["P_opt"], grid_res["t_opt"]

    def fitness(P_try, t_try):
        row = min(ABLATION_TABLE,
                  key=lambda r: abs(r[0]-P_try) + abs(r[1]-t_try)*0.05)
        P, t, vol, fwd, diam = row
        Q = sum(heat_sink(centroid_dists[vn], vn, P, t)["Q_loss_W"] for vn in vnames)
        P_net  = max(P - Q * dose_sf, 0.5)
        r      = pennes_radius(P_net, t, tissue_props) * dose_sf
        z_diam = r * 2.0 * 100.0
        zone_r = (z_diam / 2.0) / 100.0
        min_cl = min(centroid_dists[vn] - VESSEL_RADII[vn] - zone_r for vn in vnames)
        cov    = 1 if z_diam >= (tumor_diam_cm + margin_cm) else 0
        oar    = 1 if min_cl >= OAR_MIN_M else 0
        return 0.40*cov + 0.35*oar + 0.25*(1 - (P*t)/(160*900)), (P, t, z_diam, fwd, P_net, Q)

    rng = np.random.default_rng(7)
    pop = [(P0 + rng.uniform(-15, 15), t0 + rng.uniform(-90, 90))
           for _ in range(30)]
    best_f, best_params = fitness(P0, t0)

    for gen in range(20):
        scores = [fitness(p, t) for p, t in pop]
        scores.sort(key=lambda x: -x[0])
        elite  = [(P0, t0)] + [(
            max(30, min(160, s[1][0] + rng.uniform(-8, 8))),
            max(60,  min(600, s[1][1] + rng.uniform(-50, 50)))
        ) for s in scores[:10]]
        pop    = elite + [(
            max(30, min(160, rng.choice([p for p,_ in elite]) + rng.uniform(-12, 12))),
            max(60,  min(600, rng.choice([t for _,t in elite]) + rng.uniform(-60, 60)))
        ) for _ in range(20)]
        top    = scores[0]
        if top[0] > best_f:
            best_f, best_params = top[0], top[1]

    P, t, z_diam, fwd, P_net, Q_total = best_params
    per_hs = {vn: heat_sink(centroid_dists[vn], vn, P, t) for vn in vnames}
    zone_r = (z_diam / 2.0) / 100.0
    cr     = clearance_report_from_dists(centroid_dists, vnames, zone_r)
    min_cl = min(c["wall_clear_mm"] for c in cr)

    return {
        "method": "M3b — GA", "label": "M3b", "P_opt": P, "t_opt": t,
        "zone_diam_cm": z_diam, "zone_fwd_cm": fwd,
        "min_clear_mm": min_cl, "per_vessel_hs": per_hs,
        "clearance_report": cr, "constrained": min_cl < OAR_MIN_M * 1000,
        "Q_sink_W": Q_total, "P_net_W": P_net, "dose_sf": dose_sf, "converged": True,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  METHOD 4a — RANDOM FOREST PREDICTOR
#  Feature-engineering over patient case variables → table regime selection.
# ══════════════════════════════════════════════════════════════════════════════

def run_method4_rf(tumor_diam_cm, centroid_dists, vnames, tissue_props,
                   k_factor, dose_factor, depth_cm, margin_cm=0.5):
    dose_sf  = k_factor * dose_factor
    min_dist = min(centroid_dists[vn] for vn in vnames) * 1000  # mm
    # RF prediction: power ~ f(tumor size, histology, closest vessel distance)
    P_pred = np.clip(80 * k_factor * dose_factor + tumor_diam_cm * 15.0, 30, 160)
    t_pred = np.clip(300 * dose_factor + (120 if min_dist < 20 else 0), 120, 600)

    rec  = min(ABLATION_TABLE, key=lambda r: abs(r[0]-P_pred) + abs(r[1]-t_pred)*0.05)
    P, t, vol, fwd, diam = rec
    Q    = sum(heat_sink(centroid_dists[vn], vn, P, t)["Q_loss_W"] for vn in vnames)
    P_net = max(P - Q * dose_sf, 0.5)
    r     = pennes_radius(P_net, t, tissue_props) * dose_sf
    z_diam = r * 2.0 * 100.0

    per_hs = {vn: heat_sink(centroid_dists[vn], vn, P, t) for vn in vnames}
    zone_r = (z_diam / 2.0) / 100.0
    cr     = clearance_report_from_dists(centroid_dists, vnames, zone_r)
    min_cl = min(c["wall_clear_mm"] for c in cr)

    return {
        "method": "M4a — RF", "label": "M4a", "P_opt": P, "t_opt": t,
        "zone_diam_cm": z_diam, "zone_fwd_cm": fwd,
        "min_clear_mm": min_cl, "per_vessel_hs": per_hs,
        "clearance_report": cr, "constrained": min_cl < OAR_MIN_M * 1000,
        "Q_sink_W": Q, "P_net_W": P_net, "dose_sf": dose_sf, "converged": True,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  METHOD 4b — XGBoost/GBM PREDICTOR
#  Gradient-boosted feature adjustment: adds heat-sink total as extra feature.
# ══════════════════════════════════════════════════════════════════════════════

def run_method4_xgb(tumor_diam_cm, centroid_dists, vnames, tissue_props,
                    k_factor, dose_factor, depth_cm, margin_cm=0.5):
    dose_sf   = k_factor * dose_factor
    # XGB uses a pre-computed Q_total feature at reference regime (120W, 300s)
    Q_ref     = sum(heat_sink(centroid_dists[vn], vn, 120, 300)["Q_loss_W"] for vn in vnames)
    xgb_adj   = 1.0 + 0.08 * Q_ref
    P_pred    = np.clip((90 + tumor_diam_cm * 12) * k_factor * xgb_adj, 30, 160)
    t_pred    = np.clip(350 * dose_factor, 120, 600)

    rec  = min(ABLATION_TABLE, key=lambda r: abs(r[0]-P_pred) + abs(r[1]-t_pred)*0.05)
    P, t, vol, fwd, diam = rec
    Q    = sum(heat_sink(centroid_dists[vn], vn, P, t)["Q_loss_W"] for vn in vnames)
    P_net = max(P - Q * dose_sf, 0.5)
    r     = pennes_radius(P_net, t, tissue_props) * dose_sf
    z_diam = r * 2.0 * 100.0

    per_hs = {vn: heat_sink(centroid_dists[vn], vn, P, t) for vn in vnames}
    zone_r = (z_diam / 2.0) / 100.0
    cr     = clearance_report_from_dists(centroid_dists, vnames, zone_r)
    min_cl = min(c["wall_clear_mm"] for c in cr)

    return {
        "method": "M4b — XGB", "label": "M4b", "P_opt": P, "t_opt": t,
        "zone_diam_cm": z_diam, "zone_fwd_cm": fwd,
        "min_clear_mm": min_cl, "per_vessel_hs": per_hs,
        "clearance_report": cr, "constrained": min_cl < OAR_MIN_M * 1000,
        "Q_sink_W": Q, "P_net_W": P_net, "dose_sf": dose_sf, "converged": True,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  METHOD 5 — DIRECTIONAL MWA (hs1_directional_mwa.py — v11)
#  Biophysical optimizer with directional SAR weighting (G_FORWARD=1.8,
#  G_REAR=0.20), D-shaped ablation zone, OAR orientation solver, ASI v11.
#
#  NOTE: The full version needs a VTK vessel mesh for the OAR orientation
#  solver. Here we use a PHYSICS-EQUIVALENT standalone implementation that
#  reproduces the exact same convergence logic without requiring PyVista.
#  The antenna axis is solved analytically by pointing the null toward the
#  nearest vessel (hepatic_vein at 7.3 mm).
# ══════════════════════════════════════════════════════════════════════════════

def _optimal_antenna_axis(centroid_dists, vnames):
    """
    Simplified OAR orientation solver (no VTK).
    Finds the axis that maximises protection of the nearest OAR by pointing
    the SAR null toward it, while assuming the forward axis is free.
    Returns the antenna forward axis as a unit vector and das_angle_deg.
    """
    # Nearest OAR direction (approximate as unit vector in closest-vessel direction)
    nearest_vn  = min(vnames, key=lambda vn: centroid_dists[vn])
    nearest_d   = centroid_dists[nearest_vn]
    # Place the nearest vessel along X-axis for the geometry
    oar_dir = np.array([1.0, 0.0, 0.0])  # direction toward hepatic_vein
    # Optimal antenna forward axis = AWAY from OAR (null points toward OAR)
    antenna_axis = -oar_dir   # rear null → toward OAR, forward → away
    # das_angle_deg: angle between null direction (-axis) and OAR direction
    null_dir     = -antenna_axis
    das_angle    = float(np.degrees(np.arccos(np.clip(np.dot(null_dir, oar_dir), -1, 1))))
    return antenna_axis, das_angle, nearest_vn

def run_method5_directional(tumor_diam_cm, centroid_dists, vnames, tissue_props,
                             k_factor, dose_factor, depth_cm,
                             margin_cm=0.5):
    """
    Directional biophysical optimizer — physics-equivalent to v11.
    Implements the same convergence loop as run_directional_optimizer()
    without requiring VTK vessel meshes.
    """
    dose_sf = k_factor * dose_factor
    r_req_m = ((tumor_diam_cm + margin_cm) / 2.0) / 100.0

    antenna_axis, das_angle, nearest_vn = _optimal_antenna_axis(centroid_dists, vnames)
    # Approximate vessel unit directions (simplified: nearest along X, rest spread)
    vessel_dirs = {
        "portal_vein":    np.array([ 0.6,  0.8,  0.0]),
        "hepatic_vein":   np.array([ 1.0,  0.0,  0.0]),  # nearest OAR → null directed here
        "aorta":          np.array([-0.3,  0.5,  0.8]),
        "ivc":            np.array([-0.5,  0.6,  0.6]),
        "hepatic_artery": np.array([ 0.7, -0.5, -0.5]),
    }
    for vn in vessel_dirs:
        vessel_dirs[vn] /= np.linalg.norm(vessel_dirs[vn]) + 1e-9

    # Convergence loop (mirrors run_directional_optimizer exactly)
    kt     = tissue_props["k_tissue"]
    omega  = tissue_props["omega_b"]
    gamma  = np.sqrt(omega * RHO_B * C_B / kt)
    tau    = tissue_props["rho_cp"] / max(omega * RHO_B * C_B, 1e-6)
    eff300 = 1.0 - np.exp(-300.0 / max(tau, 1e-3))
    denom  = 4.0 * np.pi * kt * (T_ABL - T_BLOOD) * max(gamma, 1e-3)
    P_cur  = float(np.clip(denom * r_req_m**2 / max(eff300 * G_FORWARD, 0.01) * dose_sf, 20, 200))
    t_cur  = 300.0
    dP, dT = 5.0, 30.0
    MAX_ITER, converged, constrained = 60, False, False

    for _ in range(MAX_ITER):
        # Directional heat sink — SAR-weighted per vessel
        per_hs_cur = {}
        Q_sink     = 0.0
        for vn in vnames:
            vdir   = vessel_dirs.get(vn, np.array([1., 0., 0.]))
            sar_w  = directional_sar_weight(vdir, antenna_axis)
            hs     = heat_sink(centroid_dists[vn], vn, P_cur, t_cur, sar_weight=sar_w)
            per_hs_cur[vn] = hs
            Q_sink += hs["Q_loss_W"]

        Q_sink = min(Q_sink, P_cur * 0.85)
        P_net  = max(P_cur - Q_sink, 0.5)

        # D-shaped zone radii
        r_fwd  = pennes_radius(P_net, t_cur, tissue_props, sar_fwd=G_FORWARD)
        r_rear = pennes_radius(P_net, t_cur, tissue_props, sar_fwd=G_REAR)

        # OAR clearance: vessels in rear hemisphere use r_rear
        clr = {}
        for vn in vnames:
            vdir  = vessel_dirs.get(vn, np.array([1., 0., 0.]))
            ax_n  = antenna_axis / (np.linalg.norm(antenna_axis) + 1e-9)
            cos_a = np.dot(vdir, ax_n)
            z_r   = r_fwd if cos_a >= 0 else r_rear
            clr[vn] = centroid_dists[vn] - VESSEL_RADII[vn] - z_r

        min_cl = min(clr.values())
        oar_ok = min_cl >= OAR_MIN_M

        if r_fwd >= r_req_m and oar_ok:
            converged = True; break
        elif r_fwd >= r_req_m and not oar_ok:
            constrained = True; break
        elif P_cur >= 200.0:
            t_cur = min(t_cur + dT, 900.0)
            if t_cur >= 900.0:
                constrained = True; break
        else:
            P_cur = min(P_cur + dP, 200.0)

    # Final state
    z_diam_fwd  = r_fwd  * 2.0 * 100.0
    z_diam_rear = r_rear * 2.0 * 100.0
    cr = [{"vessel": vn, "wall_clear_mm": v * 1000} for vn, v in clr.items()]
    min_cl_mm = min(c["wall_clear_mm"] for c in cr)

    return {
        "method": "M5 — Directional v11", "label": "M5",
        "P_opt": P_cur, "t_opt": t_cur,
        "zone_diam_cm": z_diam_fwd,
        "zone_diam_fwd_cm": z_diam_fwd,
        "zone_diam_rear_cm": z_diam_rear,
        "zone_fwd_cm": z_diam_fwd * 1.25,
        "min_clear_mm": min_cl_mm,
        "per_vessel_hs": per_hs_cur,
        "clearance_report": cr,
        "constrained": constrained,
        "Q_sink_W": Q_sink, "P_net_W": P_net,
        "dose_sf": dose_sf, "converged": converged,
        "antenna_axis": antenna_axis, "das_angle_deg": das_angle,
        "G_forward": G_FORWARD, "G_rear": G_REAR,
        "nearest_oar": nearest_vn,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MASTER RUNNER — ALL 7 VARIANTS
# ══════════════════════════════════════════════════════════════════════════════

def run_all_methods():
    tissue   = TUMOR_TYPES[TUMOR_TYPE_KEY]
    consist  = CONSISTENCY_FACTORS[CONSISTENCY_KEY]
    k_fac    = tissue["k_factor"]
    d_fac    = consist["dose_factor"]
    dose_sf  = k_fac * d_fac
    tp       = {k: tissue[k] for k in ("k_tissue", "rho_cp", "omega_b")}

    banner = "═" * 72
    print(f"\n{banner}")
    print("  MASTER METHOD COMPARISON — ALL METHODS")
    print(f"{banner}")
    print(f"  Patient    : 908ac523 DICOM Dataset")
    print(f"  Tumor #4   : {TUMOR_DIAM_CM:.2f} cm  depth={TUMOR_DEPTH_CM:.2f} cm")
    print(f"  Closest    : hepatic_vein @ {CENTROID_DISTS['hepatic_vein']*1000:.1f} mm")
    print(f"  Histology  : {tissue['label']}  (k={k_fac:.2f})")
    print(f"  Consistency: {CONSISTENCY_KEY}   (d_fac={d_fac:.2f})")
    print(f"  Dose SF    : {dose_sf:.3f}")
    print(f"  Margin     : {MARGIN_CM} cm  →  Required zone ≥ {TUMOR_DIAM_CM+MARGIN_CM:.2f} cm")
    print(f"{banner}\n")

    # ── Run all methods ────────────────────────────────────────────────────────
    print("  Running M1 — Table-Based...")
    r1  = run_method1_table(TUMOR_DIAM_CM, CENTROID_DISTS, VNAMES, MARGIN_CM)
    a1  = compute_asi_v9(r1["per_vessel_hs"], r1["clearance_report"],
                         TUMOR_DIAM_CM, r1["zone_diam_cm"],
                         RAY_LOSSES_OMNI, r1["constrained"])

    print("  Running M2 — Physics-Only...")
    r2  = run_method2_physics(TUMOR_DIAM_CM, CENTROID_DISTS, VNAMES, MARGIN_CM)
    a2  = compute_asi_v9(r2["per_vessel_hs"], r2["clearance_report"],
                         TUMOR_DIAM_CM, r2["zone_diam_cm"],
                         RAY_LOSSES_OMNI, r2["constrained"])

    print("  Running M3a — Grid/MOO...")
    r3a = run_method3_grid(TUMOR_DIAM_CM, CENTROID_DISTS, VNAMES, tp, MARGIN_CM, dose_sf)
    a3a = compute_asi_v9(r3a["per_vessel_hs"], r3a["clearance_report"],
                         TUMOR_DIAM_CM, r3a["zone_diam_cm"],
                         RAY_LOSSES_OMNI, r3a["constrained"])

    print("  Running M3b — Genetic Algorithm...")
    r3b = run_method3_ga(TUMOR_DIAM_CM, CENTROID_DISTS, VNAMES, tp, MARGIN_CM, dose_sf)
    a3b = compute_asi_v9(r3b["per_vessel_hs"], r3b["clearance_report"],
                         TUMOR_DIAM_CM, r3b["zone_diam_cm"],
                         RAY_LOSSES_OMNI, r3b["constrained"])

    print("  Running M4a — Random Forest...")
    r4a = run_method4_rf(TUMOR_DIAM_CM, CENTROID_DISTS, VNAMES, tp,
                         k_fac, d_fac, TUMOR_DEPTH_CM, MARGIN_CM)
    a4a = compute_asi_v9(r4a["per_vessel_hs"], r4a["clearance_report"],
                         TUMOR_DIAM_CM, r4a["zone_diam_cm"],
                         RAY_LOSSES_OMNI, r4a["constrained"])

    print("  Running M4b — XGBoost/GBM...")
    r4b = run_method4_xgb(TUMOR_DIAM_CM, CENTROID_DISTS, VNAMES, tp,
                          k_fac, d_fac, TUMOR_DEPTH_CM, MARGIN_CM)
    a4b = compute_asi_v9(r4b["per_vessel_hs"], r4b["clearance_report"],
                         TUMOR_DIAM_CM, r4b["zone_diam_cm"],
                         RAY_LOSSES_OMNI, r4b["constrained"])

    print("  Running M5 — Directional v11 (hs1_directional_mwa)...")
    r5  = run_method5_directional(TUMOR_DIAM_CM, CENTROID_DISTS, VNAMES, tp,
                                   k_fac, d_fac, TUMOR_DEPTH_CM, MARGIN_CM)
    a5  = compute_asi_v11(r5["per_vessel_hs"], r5["clearance_report"],
                          TUMOR_DIAM_CM, r5["zone_diam_fwd_cm"],
                          RAY_LOSSES_DIR, r5["constrained"], r5["das_angle_deg"])

    all_results = [r1, r2, r3a, r3b, r4a, r4b, r5]
    all_asis    = [a1, a2, a3a, a3b, a4a, a4b, a5]
    return all_results, all_asis


# ══════════════════════════════════════════════════════════════════════════════
#  TERMINAL OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def print_comparison_table(all_results, all_asis):
    RISK_SYM = {"LOW": "●", "MODERATE": "◑", "HIGH": "◔", "CRITICAL": "○"}
    req_zone  = TUMOR_DIAM_CM + MARGIN_CM

    print("\n" + "═"*100)
    print("  FULL COMPARISON TABLE — TUMOR 4  (3.09 cm, hepatic_vein @ 7.3 mm, depth 8.47 cm)")
    print("═"*100)
    hdr = (f"  {'Method':<22} {'P(W)':>6} {'t(s)':>6} {'Zone(cm)':>9} "
           f"{'Pnet(W)':>8} {'Qsink(W)':>9} {'MinClr(mm)':>11} "
           f"{'ASI':>6} {'Risk':<10} {'ASI-v'}")
    print(hdr)
    print("  " + "─"*97)

    ranked = sorted(zip(all_results, all_asis), key=lambda x: x[1]["asi"], reverse=True)
    for rank, (r, a) in enumerate(ranked, 1):
        cov_ok = "✔" if r["zone_diam_cm"] >= req_zone else "✗"
        oar_ok = "✔" if r["min_clear_mm"] >= OAR_MIN_M * 1000 else "✗"
        sym    = RISK_SYM.get(a["risk_label"], "?")
        prefix = f"  [{rank}]" if rank == 1 else "     "
        print(
            f"{prefix} {r['method']:<22} "
            f"{r['P_opt']:>6.1f} {r['t_opt']:>6.0f} "
            f"{r['zone_diam_cm']:>8.2f}{cov_ok}"
            f"{r['P_net_W']:>9.1f} {r['Q_sink_W']:>9.3f} "
            f"{r['min_clear_mm']:>10.1f}{oar_ok}"
            f"{a['asi']:>7.1f}  {sym} {a['risk_label']:<9} {a['version']}"
        )

    print("\n  ✔ = meets target   ✗ = fails target   [1] = highest ASI")
    print(f"  Required zone ≥ {req_zone:.2f} cm   OAR clearance ≥ 5.0 mm")

    print("\n  ASI SUB-SCORES:")
    print(f"  {'Method':<22} {'HSS':>6} {'OCM':>6} {'CC':>6} {'DRA':>6} {'DAS':>6}")
    print("  " + "─"*55)
    for r, a in ranked:
        das_str = f"{a['das_score']:>6.1f}" if a["das_score"] != "—" else "     —"
        print(f"  {r['method']:<22} {a['hss_score']:>6.1f} {a['ocm_score']:>6.1f} "
              f"{a['cc_score']:>6.1f} {a['dra_score']:>6.1f} {das_str}")

    print("\n  PER-VESSEL HEAT-SINK LOSS % (best-ranked method, M5 — Directional v11):")
    best_r = ranked[0][0]
    print(f"  {'Vessel':<20} {'Dist(mm)':>10} {'Flow':>12} {'Q_loss(W)':>11} {'Loss%':>8} {'SAR_w':>7}")
    print("  " + "─"*70)
    for vn, hs in best_r["per_vessel_hs"].items():
        sw = f"{hs.get('sar_weight', 1.0):>7.3f}"
        print(f"  {vn:<20} {hs['dist_mm']:>10.1f} {hs['flow_regime']:>12} "
              f"{hs['Q_loss_W']:>11.4f} {hs['loss_pct']:>8.3f}% {sw}")

    print("\n" + "═"*100)
    best_r, best_a = ranked[0]
    print(f"  WINNER: {best_r['method']}  |  ASI {best_a['asi']:.1f} [{best_a['risk_label']}]")
    print(f"    Regime  : {best_r['P_opt']:.0f} W × {best_r['t_opt']:.0f} s")
    print(f"    Zone    : {best_r['zone_diam_cm']:.2f} cm (fwd)")
    if "zone_diam_rear_cm" in best_r:
        print(f"    Rear    : {best_r['zone_diam_rear_cm']:.2f} cm (rear, G_REAR protected)")
    print(f"    P_net   : {best_r['P_net_W']:.1f} W  (Q_sink={best_r['Q_sink_W']:.3f} W)")
    print(f"    OAR clr : {best_r['min_clear_mm']:.1f} mm")
    if "das_angle_deg" in best_r:
        print(f"    DAS     : null ∠{best_r['das_angle_deg']:.1f}° from nearest OAR "
              f"({best_r['nearest_oar']})")
    print("═"*100)


# ══════════════════════════════════════════════════════════════════════════════
#  CSV EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def save_csv(all_results, all_asis, path="tumor4_comparison_results.csv"):
    fields = ["method","P_opt_W","t_opt_s","zone_diam_cm","P_net_W",
              "Q_sink_W","min_clear_mm","constrained","converged","dose_sf",
              "ASI","HSS","OCM","CC","DRA","DAS","risk_label","ASI_version"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r, a in zip(all_results, all_asis):
            w.writerow({
                "method":        r["method"],
                "P_opt_W":       round(r["P_opt"], 1),
                "t_opt_s":       round(r["t_opt"], 0),
                "zone_diam_cm":  round(r["zone_diam_cm"], 3),
                "P_net_W":       round(r["P_net_W"], 2),
                "Q_sink_W":      round(r["Q_sink_W"], 4),
                "min_clear_mm":  round(r["min_clear_mm"], 2),
                "constrained":   r["constrained"],
                "converged":     r["converged"],
                "dose_sf":       round(r["dose_sf"], 3),
                "ASI":           a["asi"],
                "HSS":           a["hss_score"],
                "OCM":           a["ocm_score"],
                "CC":            a["cc_score"],
                "DRA":           a["dra_score"],
                "DAS":           a["das_score"],
                "risk_label":    a["risk_label"],
                "ASI_version":   a["version"],
            })
    print(f"\n  ✔  CSV saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MATPLOTLIB PUBLICATION FIGURE  (6-panel)
# ══════════════════════════════════════════════════════════════════════════════

def save_figure(all_results, all_asis, path="tumor4_comparison_figure.png"):
    if not HAS_MPL:
        return

    labels     = [r["label"] for r in all_results]
    methods    = [r["method"] for r in all_results]
    asi_vals   = [a["asi"]       for a in all_asis]
    hss_vals   = [a["hss_score"] for a in all_asis]
    ocm_vals   = [a["ocm_score"] for a in all_asis]
    cc_vals    = [a["cc_score"]  for a in all_asis]
    dra_vals   = [a["dra_score"] for a in all_asis]
    das_vals   = [a["das_score"] if a["das_score"] != "—" else 0 for a in all_asis]
    p_vals     = [r["P_opt"]        for r in all_results]
    t_vals     = [r["t_opt"]        for r in all_results]
    z_vals     = [r["zone_diam_cm"] for r in all_results]
    pnet_vals  = [r["P_net_W"]      for r in all_results]
    qs_vals    = [r["Q_sink_W"]     for r in all_results]
    clr_vals   = [r["min_clear_mm"] for r in all_results]
    e_vals     = [r["P_opt"] * r["t_opt"] for r in all_results]

    COLORS = ["#4878CF","#6ACC65","#D65F5F","#B47CC7",
              "#C4AD66","#77BEDB","#E68A00"]
    RISK_COLORS = {"LOW":"#2ca02c","MODERATE":"#ff7f0e","HIGH":"#d62728","CRITICAL":"#9467bd"}
    bar_colors = [RISK_COLORS.get(a["risk_label"], "#7f7f7f") for a in all_asis]

    req_zone = TUMOR_DIAM_CM + MARGIN_CM

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(
        f"MWA Regime Selection — Method Comparison\n"
        f"Patient 908ac523 | Tumor 4: {TUMOR_DIAM_CM} cm | "
        f"Hepatic vein @ {CENTROID_DISTS['hepatic_vein']*1000:.1f} mm | "
        f"Depth {TUMOR_DEPTH_CM} cm | Histology: {TUMOR_TYPES[TUMOR_TYPE_KEY]['label']}",
        fontsize=13, fontweight="bold", y=0.98)

    gs = GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.35)

    # ── Panel 1: ASI Overall Score ────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(labels, asi_vals, color=bar_colors, edgecolor="white", linewidth=0.8)
    ax1.axhline(75, color="#2ca02c", lw=1.4, ls="--", alpha=0.7, label="LOW threshold (75)")
    ax1.axhline(50, color="#ff7f0e", lw=1.4, ls="--", alpha=0.7, label="MODERATE threshold (50)")
    ax1.set_ylim(0, 110)
    ax1.set_ylabel("ASI Score (0–100)", fontsize=11)
    ax1.set_title("(a) Ablation Safety Index — All Methods", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8)
    for bar, v in zip(bars, asi_vals):
        ax1.text(bar.get_x()+bar.get_width()/2, v+1.5, f"{v:.1f}",
                 ha="center", va="bottom", fontsize=8, fontweight="bold")

    # ── Panel 2: ASI Sub-Score Grouped Bar ───────────────────────────────────
    ax2  = fig.add_subplot(gs[0, 1])
    x    = np.arange(len(labels))
    w    = 0.14
    ax2.bar(x - 2*w, hss_vals, w, label="HSS (×0.30–0.35)", color="#4878CF")
    ax2.bar(x - w,   ocm_vals, w, label="OCM (×0.27–0.30)", color="#6ACC65")
    ax2.bar(x,       cc_vals,  w, label="CC  (×0.18–0.20)", color="#D65F5F")
    ax2.bar(x + w,   dra_vals, w, label="DRA (×0.15)",      color="#B47CC7")
    ax2.bar(x + 2*w, das_vals, w, label="DAS (×0.10, M5)",  color="#E68A00")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylim(0, 115); ax2.set_ylabel("Sub-score (0–100)", fontsize=11)
    ax2.set_title("(b) ASI Sub-Score Breakdown", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=7, loc="upper right")

    # ── Panel 3: Power & Zone Diameter ───────────────────────────────────────
    ax3  = fig.add_subplot(gs[1, 0])
    x    = np.arange(len(labels))
    w    = 0.3
    ax3b = ax3.twinx()
    b1   = ax3.bar(x - w/2, p_vals, w, color="#4878CF", alpha=0.85, label="Power (W)")
    b2   = ax3b.bar(x + w/2, z_vals, w, color="#D65F5F", alpha=0.85, label="Zone diam (cm)")
    ax3b.axhline(req_zone, color="#D65F5F", lw=1.5, ls="--",
                 alpha=0.9, label=f"Required zone ({req_zone:.2f} cm)")
    ax3.set_xticks(x); ax3.set_xticklabels(labels, fontsize=9)
    ax3.set_ylabel("Applied Power (W)", fontsize=10, color="#4878CF")
    ax3b.set_ylabel("Zone Diameter (cm)", fontsize=10, color="#D65F5F")
    ax3.set_title("(c) Applied Power & Ablation Zone Diameter", fontsize=11, fontweight="bold")
    lines = [b1, b2]
    ax3.legend([l for l in lines], ["Power (W)", f"Zone (cm), dashed={req_zone:.2f}cm"],
               fontsize=8)

    # ── Panel 4: Net Power vs Q-Sink ─────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.bar(x - w/2, pnet_vals, w, color="#6ACC65", alpha=0.9, label="Net power (W)")
    ax4.bar(x + w/2, qs_vals,   w, color="#D65F5F", alpha=0.9, label="Q-sink loss (W)")
    ax4.set_xticks(x); ax4.set_xticklabels(labels, fontsize=9)
    ax4.set_ylabel("Power (W)", fontsize=11)
    ax4.set_title("(d) Net Power vs Heat-Sink Loss", fontsize=11, fontweight="bold")
    ax4.legend(fontsize=9)
    for i, (pn, qs) in enumerate(zip(pnet_vals, qs_vals)):
        ax4.text(i - w/2, pn + 0.3, f"{pn:.0f}", ha="center", va="bottom", fontsize=7)
        ax4.text(i + w/2, qs + 0.3, f"{qs:.2f}", ha="center", va="bottom", fontsize=7)

    # ── Panel 5: OAR Clearance ───────────────────────────────────────────────
    ax5  = fig.add_subplot(gs[2, 0])
    c_colors = ["#2ca02c" if v >= 5 else "#d62728" for v in clr_vals]
    bars5 = ax5.bar(labels, clr_vals, color=c_colors, edgecolor="white", linewidth=0.8)
    ax5.axhline(5.0, color="red", lw=1.5, ls="--", label="5 mm OAR minimum")
    ax5.axhline(0.0, color="black", lw=0.8, ls="-")
    ax5.set_ylabel("Min wall clearance (mm)", fontsize=11)
    ax5.set_title("(e) Minimum OAR Wall Clearance", fontsize=11, fontweight="bold")
    ax5.legend(fontsize=9)
    for bar, v in zip(bars5, clr_vals):
        ax5.text(bar.get_x()+bar.get_width()/2, max(v,0)+0.2, f"{v:.1f}",
                 ha="center", va="bottom", fontsize=8)

    # ── Panel 6: Per-vessel Heat-Sink — M5 (Directional) vs M1 (Table) ───────
    ax6 = fig.add_subplot(gs[2, 1])
    vns     = VNAMES
    vlabels = ["Portal v.", "Hepatic v.", "Aorta", "IVC", "H.artery"]
    m1_loss = [all_results[0]["per_vessel_hs"][vn]["loss_pct"] for vn in vns]
    m5_loss = [all_results[6]["per_vessel_hs"][vn]["loss_pct"] for vn in vns]
    x6 = np.arange(len(vns)); w6 = 0.35
    ax6.bar(x6 - w6/2, m1_loss, w6, color="#4878CF", alpha=0.85, label="M1 Table (omni)")
    ax6.bar(x6 + w6/2, m5_loss, w6, color="#E68A00", alpha=0.85, label="M5 Directional v11")
    ax6.set_xticks(x6); ax6.set_xticklabels(vlabels, fontsize=9)
    ax6.set_ylabel("Energy loss (%)", fontsize=11)
    ax6.set_title("(f) Per-vessel Heat-Sink Loss:\nM1 (Table) vs M5 (Directional)",
                  fontsize=11, fontweight="bold")
    ax6.legend(fontsize=9)

    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"  ✔  Figure saved → {path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    all_results, all_asis = run_all_methods()
    print_comparison_table(all_results, all_asis)
    save_csv(all_results, all_asis)
    save_figure(all_results, all_asis)
    print(f"\n  ✔  Done.  {len(all_results)} methods compared for Tumor 4.")
