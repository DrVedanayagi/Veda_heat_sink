#!/usr/bin/env python3
"""
Master Comparison Runner v2 - MWA Regime Selection Study
Methods: 1 (Table), 2 (Physics-Only), 3 (MOO/GA), 4 (ML)

Usage:
  python compare_all_methods_v2.py          # interactive tumor selector
  python compare_all_methods_v2.py 2        # pass tumor number directly

Plots saved as PNG:
  1. ASI Radar Chart            6. OAR Wall Clearance per vessel
  2. ASI Bar Chart (ranked)     7. Cost Surface Heatmap (M3 grid)
  3. Power x Time Scatter       8. GA Convergence Curve (M3)
  4. Zone vs Required Diam      9. ML Feature Importances (M4)
  5. Heat-Sink Heatmap         10. Summary Dashboard (6-panel)
"""

import sys
import numpy as np
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.cm import get_cmap

warnings.filterwarnings("ignore")

from method3_moo_optimizer import run_method3, compute_asi_moo
from method4_ml_predictor import run_method4, print_master_comparison

# ---------------------------------------------------------------------------
# PHYSICAL CONSTANTS
# ---------------------------------------------------------------------------
RHO_B = 1060.0; MU_B = 3.5e-3; C_B = 3700.0; K_B = 0.52
T_BLOOD = 37.0; T_TISS = 90.0; T_ABL = 60.0
ALPHA_TISSUE = 70.0; L_SEG = 0.01; OAR_MIN_CLEAR_M = 5e-3

VESSEL_DIAMETERS = {
    "portal_vein": 12e-3, "hepatic_vein": 8e-3, "aorta": 25e-3,
    "ivc": 20e-3, "hepatic_artery": 4.5e-3,
}
VESSEL_VELOCITIES = {
    "portal_vein": 0.15, "hepatic_vein": 0.20, "aorta": 0.40,
    "ivc": 0.35, "hepatic_artery": 0.30,
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
    "HCC":           {"k_factor":1.00,"label":"Hepatocellular Carcinoma","k_tissue":0.52,"rho_cp":3.6e6,"omega_b":0.0064},
    "COLORECTAL":    {"k_factor":1.12,"label":"Colorectal Metastasis",   "k_tissue":0.48,"rho_cp":3.8e6,"omega_b":0.0030},
    "NEUROENDOCRINE":{"k_factor":0.93,"label":"Neuroendocrine Tumour",   "k_tissue":0.55,"rho_cp":3.5e6,"omega_b":0.0090},
    "CHOLANGIO":     {"k_factor":1.22,"label":"Cholangiocarcinoma",      "k_tissue":0.44,"rho_cp":4.0e6,"omega_b":0.0020},
    "FATTY_BACKGROUND":{"k_factor":1.30,"label":"Tumour in Fatty Liver","k_tissue":0.38,"rho_cp":3.2e6,"omega_b":0.0015},
    "UNKNOWN":       {"k_factor":1.10,"label":"Unknown",                 "k_tissue":0.50,"rho_cp":3.7e6,"omega_b":0.0050},
}
CONSISTENCY_FACTORS = {
    "soft": {"dose_factor":0.90,"label":"Soft"},
    "firm": {"dose_factor":1.00,"label":"Firm"},
    "hard": {"dose_factor":1.20,"label":"Hard"},
}

# ---------------------------------------------------------------------------
# FOUR PRE-DEFINED TUMOR CASES
# ---------------------------------------------------------------------------
TUMOR_CASES = {
    1: {
        "label": "Small HCC — Portal Vein Close",
        "tumor_diam_cm": 2.5, "type_key": "HCC", "consist_key": "firm", "depth_cm": 8.0,
        "centroid_dists": {
            "portal_vein": 0.012, "hepatic_vein": 0.035, "aorta": 0.080,
            "ivc": 0.065, "hepatic_artery": 0.040,
        },
    },
    2: {
        "label": "Medium Colorectal Met — Multiple Vessels",
        "tumor_diam_cm": 3.5, "type_key": "COLORECTAL", "consist_key": "firm", "depth_cm": 12.0,
        "centroid_dists": {
            "portal_vein": 0.018, "hepatic_vein": 0.025, "aorta": 0.060,
            "ivc": 0.045, "hepatic_artery": 0.030,
        },
    },
    3: {
        "label": "Large Cholangio — Deep, Hard Consistency",
        "tumor_diam_cm": 4.8, "type_key": "CHOLANGIO", "consist_key": "hard", "depth_cm": 16.0,
        "centroid_dists": {
            "portal_vein": 0.022, "hepatic_vein": 0.050, "aorta": 0.090,
            "ivc": 0.055, "hepatic_artery": 0.045,
        },
    },
    4: {
        "label": "Neuroendocrine — Fatty Background, Soft",
        "tumor_diam_cm": 3.0, "type_key": "NEUROENDOCRINE", "consist_key": "soft", "depth_cm": 10.0,
        "centroid_dists": {
            "portal_vein": 0.030, "hepatic_vein": 0.020, "aorta": 0.100,
            "ivc": 0.075, "hepatic_artery": 0.055,
        },
    },
}

# ---------------------------------------------------------------------------
# METHODS 1 & 2 — physics helpers
# ---------------------------------------------------------------------------

def _nusselt(Re, Pr):
    if Re < 2300:
        return 4.36
    f = (0.790 * np.log(Re) - 1.64) ** (-2)
    Nu = (f / 8) * (Re - 1000) * Pr / (1.0 + 12.7 * np.sqrt(f / 8) * (Pr ** (2 / 3) - 1))
    if Re >= 10000:
        Nu = 0.023 * Re ** 0.8 * Pr ** 0.4
    return max(Nu, 4.36)


def _hs(d_m, vn, P, t):
    D = VESSEL_DIAMETERS[vn]; u = VESSEL_VELOCITIES[vn]
    Re = (RHO_B * u * D) / MU_B; Pr = (C_B * MU_B) / K_B; Nu = _nusselt(Re, Pr)
    hb = (Nu * K_B) / D; hw = hb * max(0.9, 1.0)
    Ac = (D / 2) * (np.pi / 3) * L_SEG; Af = np.pi * D * L_SEG
    dTw = max(T_TISS - T_BLOOD, 0.1); dTb = max((T_TISS + T_BLOOD) / 2 - T_BLOOD, 0.1)
    Qv = min(hw * Ac * dTw + (0.30 if Re >= 2300 else 0.05) * hb * Af * dTb, P)
    d = max(d_m, 1e-4); Ql = min(Qv * np.exp(-ALPHA_TISSUE * d), P)
    Ei = P * t; El = min(Ql * t, Ei)
    return {
        "vessel": vn, "dist_mm": d * 1000, "Re": Re, "Pr": Pr, "Nu": Nu,
        "flow_regime": "Laminar" if Re < 2300 else "Turbulent",
        "Q_loss_W": Ql, "E_loss_J": El, "loss_pct": 100.0 * El / max(Ei, 1e-9),
    }


def run_method1_table(tumor_diam_cm, centroid_dists, vnames, margin_cm=0.5, verbose=True):
    req = tumor_diam_cm + margin_cm
    cands = [(P, t, vol, fwd, diam) for P, t, vol, fwd, diam in ABLATION_TABLE if diam >= req]
    if not cands:
        cands = sorted(ABLATION_TABLE, key=lambda r: r[4], reverse=True)
    rec = sorted(cands, key=lambda r: (r[4], r[0], r[1]))[0]
    P_rec, t_rec, vol_rec, fwd_rec, diam_rec = rec
    per_hs = {vn: _hs(centroid_dists[vn], vn, P_rec, t_rec) for vn in vnames}
    zone_r = (diam_rec / 2.0) / 100.0
    clr = {vn: centroid_dists[vn] - VESSEL_RADII[vn] - zone_r for vn in vnames}
    min_cl = min(clr.values())
    cr = [{"vessel": vn, "wall_clear_mm": v * 1000} for vn, v in clr.items()]
    Q_sink = sum(hs["Q_loss_W"] for hs in per_hs.values())
    if verbose:
        print(f"\n  [M1-Table] {P_rec:.0f}W x {t_rec:.0f}s  diam={diam_rec:.2f}cm  clear={min_cl * 1000:.1f}mm")
    return {
        "method": "TableBased", "P_opt": P_rec, "t_opt": t_rec,
        "zone_diam_cm": diam_rec, "zone_fwd_cm": fwd_rec, "min_clear_mm": min_cl * 1000,
        "per_vessel_hs": per_hs, "clearance_report": cr,
        "constrained": min_cl < OAR_MIN_CLEAR_M, "converged": True,
        "margin_cm": margin_cm, "dose_sf": 1.0, "Q_sink_W": Q_sink, "P_net_W": P_rec,
    }


def _pennes_r(Pnet, t):
    kt = 0.52; rcp = 3.6e6; omega = 0.0064
    gam = np.sqrt(omega * RHO_B * C_B / kt); tau = rcp / max(omega * RHO_B * C_B, 1e-6)
    eff = 1.0 - np.exp(-t / max(tau, 1e-3))
    Peff = max(Pnet * eff, 0.1); den = 4 * np.pi * kt * (T_ABL - T_BLOOD) * max(gam, 1e-3)
    return float(np.clip(np.sqrt(max(Peff / den, 1e-6)), 0.005, 0.080))


def run_method2_physics_only(tumor_diam_cm, centroid_dists, vnames, margin_cm=0.5, verbose=True):
    best = None; best_cost = np.inf
    for P, t, vol, fwd, diam in ABLATION_TABLE:
        Q_total = sum(_hs(centroid_dists[vn], vn, P, t)["Q_loss_W"] for vn in vnames)
        Pnet = max(P - Q_total, 0.5); r_abl = _pennes_r(Pnet, t); zone_diam = r_abl * 2.0 * 100.0
        if zone_diam < (tumor_diam_cm + margin_cm):
            cost = 1.0 + (tumor_diam_cm + margin_cm - zone_diam)
        else:
            cost = (P * t) / (160 * 900)
        if cost < best_cost:
            best_cost = cost; best = (P, t, vol, fwd, zone_diam, Pnet, Q_total)
    P_rec, t_rec, vol_rec, fwd_rec, diam_rec, Pnet_rec, Q_rec = best
    per_hs = {vn: _hs(centroid_dists[vn], vn, P_rec, t_rec) for vn in vnames}
    zone_r = (diam_rec / 2.0) / 100.0
    clr = {vn: centroid_dists[vn] - VESSEL_RADII[vn] - zone_r for vn in vnames}
    min_cl = min(clr.values())
    cr = [{"vessel": vn, "wall_clear_mm": v * 1000} for vn, v in clr.items()]
    if verbose:
        print(f"\n  [M2-Physics] {P_rec:.0f}W x {t_rec:.0f}s  zone={diam_rec:.2f}cm  clear={min_cl * 1000:.1f}mm")
    return {
        "method": "PhysicsOnly", "P_opt": P_rec, "t_opt": t_rec,
        "zone_diam_cm": diam_rec, "zone_fwd_cm": fwd_rec, "min_clear_mm": min_cl * 1000,
        "per_vessel_hs": per_hs, "clearance_report": cr,
        "constrained": min_cl < OAR_MIN_CLEAR_M, "converged": True,
        "margin_cm": margin_cm, "dose_sf": 1.0, "Q_sink_W": Q_rec, "P_net_W": Pnet_rec,
    }


def compute_asi(opt_result, tumor_diam_cm, ray_losses=None):
    per_hs = opt_result["per_vessel_hs"]; cr = opt_result["clearance_report"]
    zone = opt_result["zone_diam_cm"]; const = opt_result["constrained"]
    max_loss = max(hs["loss_pct"] for hs in per_hs.values())
    hss_score = float(np.clip(100.0 * (1.0 - max_loss / 50.0), 0, 100))
    min_cl_mm = min(c["wall_clear_mm"] for c in cr) if cr else 20.0
    ocm_score = float(np.clip(100.0 * min_cl_mm / 20.0, 0, 100))
    margin_mm = (zone - tumor_diam_cm) * 10.0
    cc_score = float(np.clip(100.0 * margin_mm / 10.0, 0, 100))
    if const:
        cc_score *= 0.60
    if ray_losses and len(ray_losses) > 1:
        spread = float(np.max(ray_losses) - np.min(ray_losses))
        dra_score = float(np.clip(100.0 * (1.0 - spread / 30.0), 0, 100))
    else:
        dra_score = 50.0
    w = {"hss": 0.35, "ocm": 0.30, "cc": 0.20, "dra": 0.15}
    asi = w["hss"] * hss_score + w["ocm"] * ocm_score + w["cc"] * cc_score + w["dra"] * dra_score
    risk = "LOW" if asi >= 75 else "MODERATE" if asi >= 50 else "HIGH" if asi >= 30 else "CRITICAL"
    return {
        "asi": round(asi, 1), "hss_score": round(hss_score, 1),
        "ocm_score": round(ocm_score, 1), "cc_score": round(cc_score, 1),
        "dra_score": round(dra_score, 1), "risk_label": risk,
        "max_loss_pct": round(max_loss, 2), "min_clear_mm": round(min_cl_mm, 1),
        "margin_mm": round(margin_mm, 1), "method": opt_result.get("method", ""),
    }

# ---------------------------------------------------------------------------
# TUMOR SELECTOR
# ---------------------------------------------------------------------------

def select_tumor_case():
    print("\n" + "=" * 66)
    print("  MWA COMPARISON STUDY  —  TUMOR CASE SELECTOR")
    print("=" * 66)
    for num, case in TUMOR_CASES.items():
        t = case["tumor_diam_cm"]
        h = TUMOR_TYPES[case["type_key"]]["label"]
        c = CONSISTENCY_FACTORS[case["consist_key"]]["label"]
        print(f"  [{num}]  {case['label']}")
        print(f"       diameter {t:.1f} cm  |  {h}  |  {c}")
        if num < 4:
            print("       " + "-" * 56)
    print("=" * 66)

    if len(sys.argv) > 1:
        try:
            ch = int(sys.argv[1])
            if ch in TUMOR_CASES:
                print(f"\n  Using command-line argument: Tumor {ch}")
                return ch
        except ValueError:
            pass

    while True:
        raw = input("\n  Enter tumor number [1 / 2 / 3 / 4]: ").strip()
        try:
            ch = int(raw)
            if ch in TUMOR_CASES:
                return ch
            print("  Please enter 1, 2, 3, or 4.")
        except ValueError:
            print("  Invalid — enter a digit.")

# ---------------------------------------------------------------------------
# MASTER RUNNER
# ---------------------------------------------------------------------------

def run_all_methods_comparison(
        tumor_diam_cm, centroid_dists, vnames,
        type_key="HCC", consist_key="firm",
        depth_cm=10.0, margin_cm=0.5, ray_losses=None, verbose=True):

    tissue = TUMOR_TYPES[type_key]; consist = CONSISTENCY_FACTORS[consist_key]
    k_fac = tissue["k_factor"]; d_fac = consist["dose_factor"]; dose_sf = k_fac * d_fac
    tissue_props = {
        "k_tissue": tissue["k_tissue"],
        "rho_cp":   tissue["rho_cp"],
        "omega_b":  tissue["omega_b"],
    }

    print("\n" + "=" * 72)
    print("  MASTER COMPARISON — ALL 4 REGIME SELECTION METHODS")
    print("=" * 72)

    all_results = {}; all_asis = {}

    # Method 1
    print(f"\n{'─' * 72}\n  METHOD 1 — TABLE-BASED\n{'─' * 72}")
    r1 = run_method1_table(tumor_diam_cm, centroid_dists, vnames, margin_cm, verbose)
    a1 = compute_asi(r1, tumor_diam_cm, ray_losses)
    all_results["M1_Table"] = r1; all_asis["M1_Table"] = a1

    # Method 2
    print(f"\n{'─' * 72}\n  METHOD 2 — PHYSICS-ONLY\n{'─' * 72}")
    r2 = run_method2_physics_only(tumor_diam_cm, centroid_dists, vnames, margin_cm, verbose)
    a2 = compute_asi(r2, tumor_diam_cm, ray_losses)
    all_results["M2_Physics"] = r2; all_asis["M2_Physics"] = a2

    # Method 3
    print(f"\n{'─' * 72}\n  METHOD 3 — MOO (Grid Search + Genetic Algorithm)\n{'─' * 72}")
    m3_out = run_method3(
        tumor_diam_cm=tumor_diam_cm, centroid_dists=centroid_dists,
        vnames=vnames, tissue_props=tissue_props, margin_cm=margin_cm,
        dose_sf=dose_sf, ray_losses=ray_losses, verbose=verbose)
    gs_result, gs_asi = m3_out[0], m3_out[1]
    ga_result, ga_asi = m3_out[4], m3_out[5]
    ga_history = m3_out[6]; gs_grid = m3_out[3]
    all_results["M3_GridSrch"] = gs_result; all_asis["M3_GridSrch"] = gs_asi
    all_results["M3_GA"] = ga_result;       all_asis["M3_GA"] = ga_asi

    # Method 4
    print(f"\n{'─' * 72}\n  METHOD 4 — ML PREDICTOR (RF + XGBoost/GBM)\n{'─' * 72}")
    m4_out = run_method4(
        tumor_diam_cm=tumor_diam_cm, centroid_dists=centroid_dists,
        vnames=vnames, tissue_props=tissue_props, k_factor=k_fac,
        dose_factor=d_fac, depth_cm=depth_cm, margin_cm=margin_cm,
        ray_losses=ray_losses, verbose=verbose)
    m4_results, m4_asis, m4_models, m4_metrics, m4_importances = m4_out
    for model_name, res in m4_results.items():
        label = f"M4_{model_name[:7]}"
        all_results[label] = res; all_asis[label] = m4_asis[model_name]

    print_master_comparison(all_results, all_asis)

    print("\n  RANKING BY ASI SCORE:")
    ranked = sorted(all_asis.items(), key=lambda x: x[1]["asi"], reverse=True)
    for rank, (label, asi) in enumerate(ranked, 1):
        r = all_results[label]
        print(f"  {rank}. {label:<18}  ASI={asi['asi']:.1f}  [{asi['risk_label']}]  "
              f"{r['P_opt']:.0f}W x {r['t_opt']:.0f}s  zone={r['zone_diam_cm']:.2f}cm  "
              f"clear={r['min_clear_mm']:.1f}mm")

    return all_results, all_asis, ga_history, gs_grid, m4_importances

# ---------------------------------------------------------------------------
# COLOUR HELPERS
# ---------------------------------------------------------------------------
_M_COLORS = {
    "M1_T": "#E63946", "M2_P": "#F4A261", "M3_G": "#2A9D8F",
    "M3_GA": "#457B9D", "M4_R": "#6A4C93", "M4_X": "#1D3461", "M4_G_b": "#1D3461",
}
_R_COLORS = {"LOW": "#2dc653", "MODERATE": "#f4c542", "HIGH": "#f4722b", "CRITICAL": "#c0392b"}

def _mcol(label):
    for pre, col in _M_COLORS.items():
        if label.startswith(pre.replace("_b", "")):
            return col
    return "#888888"

def _rcol(risk):
    return _R_COLORS.get(risk, "#888")

# ---------------------------------------------------------------------------
# PLOT HELPERS — dark theme base
# ---------------------------------------------------------------------------

def _dark_ax(ax):
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#444")
    ax.grid(color="#333", linewidth=0.5)
    for t in ax.get_xticklabels(): t.set_color("white")
    for t in ax.get_yticklabels(): t.set_color("white")

def _dark_fig():
    fig, ax = plt.subplots()
    fig.patch.set_facecolor("#0d1117")
    return fig, ax

# ---------------------------------------------------------------------------
# PLOT 1 — ASI Radar
# ---------------------------------------------------------------------------

def plot_asi_radar(all_asis, case_label, pfx):
    cats = ["HSS", "OCM", "CC", "DRA", "Overall"]; N = len(cats)
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles = base_angles.tolist() + [base_angles[0]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")

    for label, asi in all_asis.items():
        vals = [asi["hss_score"], asi["ocm_score"], asi["cc_score"],
                asi["dra_score"], asi["asi"]] + [asi["hss_score"]]
        col = _mcol(label)
        ax.plot(angles, vals, "o-", linewidth=2, color=col, label=label, markersize=4)
        ax.fill(angles, vals, alpha=0.10, color=col)

    ax.set_xticks(angles[:-1]); ax.set_xticklabels(cats, color="white", size=12, fontweight="bold")
    ax.set_ylim(0, 100); ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], color="#aaa", size=8)
    ax.tick_params(colors="white"); ax.spines["polar"].set_color("#444")
    ax.grid(color="#333", linewidth=0.6)
    ax.legend(loc="upper right", bbox_to_anchor=(1.38, 1.12),
              labelcolor="white", fontsize=9, facecolor="#1a1a2e", edgecolor="#444")
    ax.set_title(f"ASI Sub-Score Radar\n{case_label}", color="white",
                 fontsize=13, fontweight="bold", pad=25)

    plt.tight_layout()
    fname = f"{pfx}_1_radar.png"
    plt.savefig(fname, dpi=150, facecolor=fig.get_facecolor()); plt.close()
    print(f"  Saved: {fname}"); return fname

# ---------------------------------------------------------------------------
# PLOT 2 — ASI Bar Chart (ranked)
# ---------------------------------------------------------------------------

def plot_asi_bars(all_asis, case_label, pfx):
    ranked = sorted(all_asis.items(), key=lambda x: x[1]["asi"], reverse=True)
    labels = [r[0] for r in ranked]
    scores = [r[1]["asi"] for r in ranked]
    colors = [_rcol(r[1]["risk_label"]) for r in ranked]

    fig, ax = plt.subplots(figsize=(10, 5)); fig.patch.set_facecolor("#0d1117")
    _dark_ax(ax); ax.set_facecolor("#161b22")
    ax.grid(axis="x", color="#333", linewidth=0.5); ax.grid(axis="y", visible=False)

    bars = ax.barh(labels, scores, color=colors, edgecolor="#333", height=0.55, zorder=3)
    for bar, score, (_, asi) in zip(bars, scores, ranked):
        ax.text(score + 0.8, bar.get_y() + bar.get_height() / 2,
                f"{score:.1f}  [{asi['risk_label']}]",
                va="center", ha="left", color="white", fontsize=10, fontweight="bold")

    ax.axvline(75, color="#2dc653", linestyle="--", linewidth=1.2, label="LOW (75)", zorder=4)
    ax.axvline(50, color="#f4c542", linestyle="--", linewidth=1.2, label="MODERATE (50)", zorder=4)
    ax.axvline(30, color="#f4722b", linestyle="--", linewidth=1.2, label="HIGH (30)", zorder=4)

    ax.set_xlim(0, 115); ax.set_xlabel("ASI Score (/100)", color="white", fontsize=11)
    ax.set_title(f"ASI Score Ranking\n{case_label}", color="white", fontsize=13, fontweight="bold")
    ax.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white", fontsize=9)

    plt.tight_layout()
    fname = f"{pfx}_2_asi_bars.png"
    plt.savefig(fname, dpi=150, facecolor=fig.get_facecolor()); plt.close()
    print(f"  Saved: {fname}"); return fname

# ---------------------------------------------------------------------------
# PLOT 3 — Regime Scatter (P vs t)
# ---------------------------------------------------------------------------

def plot_regime_scatter(all_results, all_asis, case_label, pfx):
    fig, ax = plt.subplots(figsize=(9, 6)); fig.patch.set_facecolor("#0d1117")
    _dark_ax(ax)
    for label, res in all_results.items():
        asi = all_asis[label]["asi"]; col = _mcol(label); size = 120 + asi * 2.5
        ax.scatter(res["t_opt"], res["P_opt"], s=size, c=col, edgecolors="white",
                   linewidths=0.8, zorder=5, label=label, alpha=0.92)
        ax.annotate(f"{label}\nASI={asi:.0f}", (res["t_opt"], res["P_opt"]),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=7.5, color="white", zorder=6)
    ax.set_xlabel("Ablation Time (s)", color="white", fontsize=11)
    ax.set_ylabel("Power (W)", color="white", fontsize=11)
    ax.set_title(f"Regime: Power x Time\n{case_label}", color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fname = f"{pfx}_3_regime_scatter.png"
    plt.savefig(fname, dpi=150, facecolor=fig.get_facecolor()); plt.close()
    print(f"  Saved: {fname}"); return fname

# ---------------------------------------------------------------------------
# PLOT 4 — Zone Coverage
# ---------------------------------------------------------------------------

def plot_zone_coverage(all_results, tumor_diam_cm, margin_cm, case_label, pfx):
    required = tumor_diam_cm + margin_cm
    labels = list(all_results.keys())
    zones = [all_results[l]["zone_diam_cm"] for l in labels]

    fig, ax = plt.subplots(figsize=(10, 5)); fig.patch.set_facecolor("#0d1117")
    _dark_ax(ax); ax.grid(axis="y", color="#333", linewidth=0.5); ax.grid(axis="x", visible=False)

    x = np.arange(len(labels))
    bars = ax.bar(x, zones, color=[_mcol(l) for l in labels], edgecolor="#555", width=0.55, zorder=3)
    ax.axhline(required, color="#e63946", linewidth=2, linestyle="--",
               label=f"Required = {required:.2f} cm", zorder=4)
    ax.axhline(tumor_diam_cm, color="#f4a261", linewidth=1.2, linestyle=":",
               label=f"Tumor Ø = {tumor_diam_cm:.2f} cm", zorder=4)

    for bar, z in zip(bars, zones):
        ok = z >= required
        ax.text(bar.get_x() + bar.get_width() / 2, z + 0.05,
                f"{z:.2f}\n{'OK' if ok else 'MISS'}",
                ha="center", va="bottom",
                color="#2dc653" if ok else "#e63946", fontsize=9, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", color="white")
    ax.set_ylabel("Zone Diameter (cm)", color="white", fontsize=11)
    ax.set_title(f"Ablation Zone vs Required Coverage\n{case_label}", color="white",
                 fontsize=13, fontweight="bold")
    ax.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white", fontsize=9)
    plt.tight_layout()
    fname = f"{pfx}_4_zone_coverage.png"
    plt.savefig(fname, dpi=150, facecolor=fig.get_facecolor()); plt.close()
    print(f"  Saved: {fname}"); return fname

# ---------------------------------------------------------------------------
# PLOT 5 — Heat-Sink Loss Heatmap
# ---------------------------------------------------------------------------

def plot_heatsink_heatmap(all_results, case_label, pfx):
    methods = list(all_results.keys())
    vessels = list(VESSEL_DIAMETERS.keys())
    matrix = np.zeros((len(methods), len(vessels)))
    for i, m in enumerate(methods):
        per_hs = all_results[m]["per_vessel_hs"]
        for j, vn in enumerate(vessels):
            if vn in per_hs:
                matrix[i, j] = per_hs[vn]["loss_pct"]

    fig, ax = plt.subplots(figsize=(10, 6)); fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22"); ax.tick_params(colors="white")

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=50)
    for i in range(len(methods)):
        for j in range(len(vessels)):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    fontsize=9, color="black" if val > 25 else "white", fontweight="bold")

    ax.set_xticks(range(len(vessels)))
    ax.set_xticklabels([v.replace("_", " ").title() for v in vessels],
                       rotation=20, ha="right", color="white", fontsize=9)
    ax.set_yticks(range(len(methods))); ax.set_yticklabels(methods, color="white", fontsize=9)
    ax.set_title(f"Heat-Sink Energy Loss %  (Vessel x Method)\n{case_label}",
                 color="white", fontsize=13, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Energy Loss (%)", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    plt.tight_layout()
    fname = f"{pfx}_5_heatsink_heatmap.png"
    plt.savefig(fname, dpi=150, facecolor=fig.get_facecolor()); plt.close()
    print(f"  Saved: {fname}"); return fname

# ---------------------------------------------------------------------------
# PLOT 6 — OAR Wall Clearance
# ---------------------------------------------------------------------------

def plot_oar_clearance(all_results, case_label, pfx):
    methods = list(all_results.keys())
    vessels = list(VESSEL_DIAMETERS.keys())
    n_v = len(vessels); n_m = len(methods)
    x = np.arange(n_v); width = 0.8 / n_m

    fig, ax = plt.subplots(figsize=(12, 6)); fig.patch.set_facecolor("#0d1117")
    _dark_ax(ax); ax.grid(axis="y", color="#333", linewidth=0.5); ax.grid(axis="x", visible=False)

    for i, (label, res) in enumerate(all_results.items()):
        cr_dict = {c["vessel"]: c["wall_clear_mm"] for c in res["clearance_report"]}
        vals = [cr_dict.get(vn, 0.0) for vn in vessels]
        offset = (i - n_m / 2 + 0.5) * width
        ax.bar(x + offset, vals, width=width * 0.9, color=_mcol(label),
               label=label, edgecolor="#333", zorder=3)

    ax.axhline(5.0, color="#e63946", linewidth=2, linestyle="--",
               label="Min clearance = 5 mm", zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels([v.replace("_", "\n").title() for v in vessels], color="white", fontsize=9)
    ax.set_ylabel("Wall Clearance (mm)", color="white", fontsize=11)
    ax.set_title(f"OAR Wall Clearance by Vessel\n{case_label}", color="white",
                 fontsize=13, fontweight="bold")
    ax.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white", fontsize=8,
              ncol=2, loc="upper right")
    plt.tight_layout()
    fname = f"{pfx}_6_oar_clearance.png"
    plt.savefig(fname, dpi=150, facecolor=fig.get_facecolor()); plt.close()
    print(f"  Saved: {fname}"); return fname

# ---------------------------------------------------------------------------
# PLOT 7 — Cost Surface (M3 grid search)
# ---------------------------------------------------------------------------

def plot_cost_surface(gs_grid, best_P, best_t, case_label, pfx):
    P_vals = gs_grid["P_vals"]; t_vals = gs_grid["t_vals"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5)); fig.patch.set_facecolor("#0d1117")

    panels = [
        ("cost",  "Total Cost J",       "inferno_r"),
        ("under", "Under-coverage Term", "Blues"),
        ("oar",   "OAR Risk Term",       "Reds"),
    ]
    for ax, (key, title, cmap_name) in zip(axes, panels):
        ax.set_facecolor("#161b22"); ax.tick_params(colors="white")
        data = gs_grid[key]
        im = ax.contourf(t_vals, P_vals, data, levels=20, cmap=cmap_name)
        ax.contour(t_vals, P_vals, data, levels=8, colors="white", linewidths=0.4, alpha=0.4)
        ax.scatter([best_t], [best_P], marker="*", s=200, color="#00ff88", zorder=10,
                   label=f"Opt {best_P:.0f}W x {best_t:.0f}s")
        ax.set_xlabel("Time (s)", color="white", fontsize=10)
        ax.set_ylabel("Power (W)", color="white", fontsize=10)
        ax.set_title(title, color="white", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, facecolor="#222", edgecolor="#555", labelcolor="white", loc="upper left")
        cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
        for t in ax.get_xticklabels(): t.set_color("white")
        for t in ax.get_yticklabels(): t.set_color("white")

    fig.suptitle(f"Method 3 — Grid-Search Cost Surface\n{case_label}",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fname = f"{pfx}_7_cost_surface.png"
    plt.savefig(fname, dpi=150, facecolor=fig.get_facecolor()); plt.close()
    print(f"  Saved: {fname}"); return fname

# ---------------------------------------------------------------------------
# PLOT 8 — GA Convergence
# ---------------------------------------------------------------------------

def plot_ga_convergence(ga_history, case_label, pfx):
    fig, ax = plt.subplots(figsize=(9, 5)); fig.patch.set_facecolor("#0d1117")
    _dark_ax(ax)
    gens = np.arange(1, len(ga_history) + 1)
    ax.plot(gens, ga_history, color="#457B9D", linewidth=2.2, label="Best Cost per Generation")
    ax.fill_between(gens, ga_history, alpha=0.15, color="#457B9D")
    ax.scatter([len(ga_history)], [ga_history[-1]], s=100, color="#00ff88", zorder=5,
               label=f"Final = {ga_history[-1]:.5f}")
    improv = (ga_history[0] - ga_history[-1]) / ga_history[0] * 100
    ax.annotate(f"Improvement: {improv:.1f}%",
                xy=(len(ga_history) * 0.55, (ga_history[0] + ga_history[-1]) / 2),
                color="#f4c542", fontsize=11, fontweight="bold")
    ax.set_xlabel("Generation", color="white", fontsize=11)
    ax.set_ylabel("Cost J  (lower = better)", color="white", fontsize=11)
    ax.set_title(f"Method 3 — Genetic Algorithm Convergence\n{case_label}",
                 color="white", fontsize=13, fontweight="bold")
    ax.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white", fontsize=10)
    plt.tight_layout()
    fname = f"{pfx}_8_ga_convergence.png"
    plt.savefig(fname, dpi=150, facecolor=fig.get_facecolor()); plt.close()
    print(f"  Saved: {fname}"); return fname

# ---------------------------------------------------------------------------
# PLOT 9 — ML Feature Importances
# ---------------------------------------------------------------------------

def plot_feature_importance(m4_importances, case_label, pfx):
    if not m4_importances:
        print("  (No ML importances available — skipping plot 9)")
        return None
    n_models = len(m4_importances)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6))
    if n_models == 1:
        axes = [axes]
    fig.patch.set_facecolor("#0d1117"); cmap = get_cmap("viridis")

    for ax, (model_name, imp_dict) in zip(axes, m4_importances.items()):
        ax.set_facecolor("#161b22"); ax.tick_params(colors="white")
        sorted_items = sorted(imp_dict.items(), key=lambda x: x[1])
        feats = [i[0].replace("_", "\n") for i in sorted_items]
        values = [i[1] for i in sorted_items]
        colors = [cmap(v / max(values)) for v in values]
        ax.barh(feats, values, color=colors, edgecolor="#333", height=0.6)
        for feat_i, v in enumerate(values):
            ax.text(v + 0.002, feat_i, f"{v:.3f}", va="center", color="white", fontsize=8.5)
        ax.set_xlim(0, max(values) * 1.28)
        ax.set_xlabel("Importance", color="white", fontsize=10)
        ax.set_title(f"[{model_name}] Feature Importance", color="white",
                     fontsize=11, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#444")
        ax.grid(axis="x", color="#333", linewidth=0.5)
        for t in ax.get_xticklabels(): t.set_color("white")
        for t in ax.get_yticklabels(): t.set_color("white")

    fig.suptitle(f"Method 4 — ML Feature Importances\n{case_label}",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fname = f"{pfx}_9_feature_importance.png"
    plt.savefig(fname, dpi=150, facecolor=fig.get_facecolor()); plt.close()
    print(f"  Saved: {fname}"); return fname

# ---------------------------------------------------------------------------
# PLOT 10 — 6-Panel Dashboard
# ---------------------------------------------------------------------------

def plot_dashboard(all_results, all_asis, tumor_diam_cm, margin_cm,
                   ga_history, case_label, pfx):
    fig = plt.figure(figsize=(18, 12)); fig.patch.set_facecolor("#0d1117")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

    # A — Radar
    ax_r = fig.add_subplot(gs[0, 0], polar=True); ax_r.set_facecolor("#0d1117")
    cats = ["HSS", "OCM", "CC", "DRA", "Overall"]; N = len(cats)
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles = base_angles.tolist() + [base_angles[0]]
    for label, asi in all_asis.items():
        vals = [asi["hss_score"], asi["ocm_score"], asi["cc_score"],
                asi["dra_score"], asi["asi"]] + [asi["hss_score"]]
        col = _mcol(label)
        ax_r.plot(angles, vals, "o-", linewidth=1.8, color=col, label=label, markersize=3)
        ax_r.fill(angles, vals, alpha=0.07, color=col)
    ax_r.set_xticks(angles[:-1]); ax_r.set_xticklabels(cats, color="white", size=8)
    ax_r.set_ylim(0, 100); ax_r.set_yticks([25, 50, 75, 100])
    ax_r.set_yticklabels(["25", "50", "75", "100"], color="#aaa", size=6)
    ax_r.grid(color="#333", linewidth=0.5); ax_r.spines["polar"].set_color("#444")
    ax_r.legend(loc="upper right", bbox_to_anchor=(1.5, 1.12),
                labelcolor="white", fontsize=7, facecolor="#1a1a2e", edgecolor="#444")
    ax_r.set_title("ASI Radar", color="white", fontsize=10, fontweight="bold", pad=14)

    # B — ASI bars
    ax_b = fig.add_subplot(gs[0, 1]); ax_b.set_facecolor("#161b22")
    ranked = sorted(all_asis.items(), key=lambda x: x[1]["asi"], reverse=True)
    m_labels = [r[0] for r in ranked]; m_scores = [r[1]["asi"] for r in ranked]
    ax_b.barh(m_labels, m_scores,
              color=[_rcol(r[1]["risk_label"]) for r in ranked], edgecolor="#333", height=0.5)
    ax_b.axvline(75, color="#2dc653", linestyle="--", linewidth=1)
    ax_b.axvline(50, color="#f4c542", linestyle="--", linewidth=1)
    for lbl, sc in zip(m_labels, m_scores):
        ax_b.text(sc + 0.5, m_labels.index(lbl), f"{sc:.1f}", va="center", color="white", fontsize=8)
    ax_b.set_xlim(0, 115); ax_b.set_xlabel("ASI (/100)", color="white", fontsize=9)
    ax_b.set_title("ASI Ranking", color="white", fontsize=10, fontweight="bold")
    ax_b.tick_params(colors="white"); ax_b.spines[["top", "right"]].set_visible(False)
    ax_b.spines[["left", "bottom"]].set_color("#444"); ax_b.grid(axis="x", color="#333", linewidth=0.4)
    for t in ax_b.get_xticklabels(): t.set_color("white")
    for t in ax_b.get_yticklabels(): t.set_color("white")

    # C — Regime scatter
    ax_sc = fig.add_subplot(gs[0, 2]); ax_sc.set_facecolor("#161b22")
    for label, res in all_results.items():
        col = _mcol(label)
        ax_sc.scatter(res["t_opt"], res["P_opt"], s=90, c=col,
                      edgecolors="white", linewidths=0.5, zorder=5)
        ax_sc.annotate(label, (res["t_opt"], res["P_opt"]),
                       textcoords="offset points", xytext=(5, 3),
                       fontsize=7, color="white", zorder=6)
    ax_sc.set_xlabel("Time (s)", color="white", fontsize=9)
    ax_sc.set_ylabel("Power (W)", color="white", fontsize=9)
    ax_sc.set_title("Regime: P x t", color="white", fontsize=10, fontweight="bold")
    ax_sc.tick_params(colors="white"); ax_sc.spines[["top", "right"]].set_visible(False)
    ax_sc.spines[["left", "bottom"]].set_color("#444"); ax_sc.grid(color="#333", linewidth=0.4)
    for t in ax_sc.get_xticklabels(): t.set_color("white")
    for t in ax_sc.get_yticklabels(): t.set_color("white")

    # D — Zone coverage
    ax_zn = fig.add_subplot(gs[1, 0]); ax_zn.set_facecolor("#161b22")
    required = tumor_diam_cm + margin_cm
    b_labels = list(all_results.keys())
    zones = [all_results[l]["zone_diam_cm"] for l in b_labels]
    ax_zn.bar(range(len(b_labels)), zones, color=[_mcol(l) for l in b_labels], edgecolor="#555", width=0.55)
    ax_zn.axhline(required, color="#e63946", linewidth=1.5, linestyle="--",
                  label=f"Need {required:.2f} cm")
    ax_zn.set_xticks(range(len(b_labels)))
    ax_zn.set_xticklabels(b_labels, rotation=25, ha="right", color="white", fontsize=7)
    ax_zn.set_ylabel("Zone Ø (cm)", color="white", fontsize=9)
    ax_zn.set_title(f"Zone vs Required ({required:.2f} cm)", color="white", fontsize=10, fontweight="bold")
    ax_zn.tick_params(colors="white"); ax_zn.spines[["top", "right"]].set_visible(False)
    ax_zn.spines[["left", "bottom"]].set_color("#444"); ax_zn.grid(axis="y", color="#333", linewidth=0.4)
    for t in ax_zn.get_yticklabels(): t.set_color("white")
    ax_zn.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white", fontsize=8)

    # E — Min OAR clearance
    ax_cl = fig.add_subplot(gs[1, 1]); ax_cl.set_facecolor("#161b22")
    clears = [all_results[l]["min_clear_mm"] for l in b_labels]
    c_colors = ["#2dc653" if c >= 5.0 else "#e63946" for c in clears]
    ax_cl.bar(range(len(b_labels)), clears, color=c_colors, edgecolor="#555", width=0.55)
    ax_cl.axhline(5.0, color="#f4c542", linewidth=1.5, linestyle="--", label="5 mm min")
    ax_cl.set_xticks(range(len(b_labels)))
    ax_cl.set_xticklabels(b_labels, rotation=25, ha="right", color="white", fontsize=7)
    ax_cl.set_ylabel("Min Wall Clear (mm)", color="white", fontsize=9)
    ax_cl.set_title("OAR Min Clearance", color="white", fontsize=10, fontweight="bold")
    ax_cl.tick_params(colors="white"); ax_cl.spines[["top", "right"]].set_visible(False)
    ax_cl.spines[["left", "bottom"]].set_color("#444"); ax_cl.grid(axis="y", color="#333", linewidth=0.4)
    for t in ax_cl.get_yticklabels(): t.set_color("white")
    ax_cl.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white", fontsize=8)

    # F — GA convergence
    ax_ga = fig.add_subplot(gs[1, 2]); ax_ga.set_facecolor("#161b22")
    if ga_history:
        gens = np.arange(1, len(ga_history) + 1)
        ax_ga.plot(gens, ga_history, color="#457B9D", linewidth=2)
        ax_ga.fill_between(gens, ga_history, alpha=0.15, color="#457B9D")
    ax_ga.set_xlabel("Generation", color="white", fontsize=9)
    ax_ga.set_ylabel("Cost J", color="white", fontsize=9)
    ax_ga.set_title("GA Convergence (M3)", color="white", fontsize=10, fontweight="bold")
    ax_ga.tick_params(colors="white"); ax_ga.spines[["top", "right"]].set_visible(False)
    ax_ga.spines[["left", "bottom"]].set_color("#444"); ax_ga.grid(color="#333", linewidth=0.4)
    for t in ax_ga.get_xticklabels(): t.set_color("white")
    for t in ax_ga.get_yticklabels(): t.set_color("white")

    fig.suptitle(f"MWA Comparison Dashboard — {case_label}",
                 color="white", fontsize=15, fontweight="bold", y=0.98)
    fname = f"{pfx}_10_dashboard.png"
    plt.savefig(fname, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight"); plt.close()
    print(f"  Saved: {fname}"); return fname

# ---------------------------------------------------------------------------
# GENERATE ALL 10 PLOTS
# ---------------------------------------------------------------------------

def generate_all_plots(all_results, all_asis, tumor_diam_cm, margin_cm,
                        ga_history, gs_grid, m4_importances, gs_result,
                        case_label, tumor_num):
    pfx = f"tumor{tumor_num}_mwa"
    print("\n" + "=" * 64)
    print("  GENERATING ALL 10 PLOTS ...")
    print("=" * 64)

    files = []
    files.append(plot_asi_radar(all_asis, case_label, pfx))
    files.append(plot_asi_bars(all_asis, case_label, pfx))
    files.append(plot_regime_scatter(all_results, all_asis, case_label, pfx))
    files.append(plot_zone_coverage(all_results, tumor_diam_cm, margin_cm, case_label, pfx))
    files.append(plot_heatsink_heatmap(all_results, case_label, pfx))
    files.append(plot_oar_clearance(all_results, case_label, pfx))
    if gs_grid is not None:
        files.append(plot_cost_surface(gs_grid, gs_result["P_opt"],
                                       gs_result["t_opt"], case_label, pfx))
    files.append(plot_ga_convergence(ga_history, case_label, pfx))
    fi = plot_feature_importance(m4_importances, case_label, pfx)
    if fi:
        files.append(fi)
    files.append(plot_dashboard(all_results, all_asis, tumor_diam_cm, margin_cm,
                                 ga_history, case_label, pfx))

    print("\n" + "=" * 64)
    print(f"  Done.  {len(files)} PNG files saved with prefix  '{pfx}_*.png'")
    print("=" * 64)
    return files

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tumor_num = select_tumor_case()
    case = TUMOR_CASES[tumor_num]
    case_label = f"Tumor {tumor_num}: {case['label']}"

    print(f"\n  Selected: {case_label}")
    print(f"  diameter {case['tumor_diam_cm']:.1f} cm  |  "
          f"{TUMOR_TYPES[case['type_key']]['label']}  |  "
          f"{CONSISTENCY_FACTORS[case['consist_key']]['label']}  |  "
          f"depth {case['depth_cm']:.1f} cm")

    np.random.seed(7)
    demo_rays = np.random.uniform(0.5, 8.0, 200).tolist()
    vnames = list(case["centroid_dists"].keys())

    all_results, all_asis, ga_history, gs_grid, m4_importances = run_all_methods_comparison(
        tumor_diam_cm=case["tumor_diam_cm"],
        centroid_dists=case["centroid_dists"],
        vnames=vnames,
        type_key=case["type_key"],
        consist_key=case["consist_key"],
        depth_cm=case["depth_cm"],
        margin_cm=0.5,
        ray_losses=demo_rays,
        verbose=True,
    )

    gs_result = all_results.get("M3_GridSrch", list(all_results.values())[0])

    generate_all_plots(
        all_results=all_results,
        all_asis=all_asis,
        tumor_diam_cm=case["tumor_diam_cm"],
        margin_cm=0.5,
        ga_history=ga_history,
        gs_grid=gs_grid,
        m4_importances=m4_importances,
        gs_result=gs_result,
        case_label=case_label,
        tumor_num=tumor_num,
    )

    print(f"\n  Complete.  {len(all_results)} methods compared for {case_label}.")
