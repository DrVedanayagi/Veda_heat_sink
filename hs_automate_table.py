#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        ADAPTIVE MICROWAVE ABLATION PLANNING SYSTEM  —  v10                 ║
║        Heat Sink + Biophysical Dose Optimizer + ASI Risk Index              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Author  : Veda Nunna                                                        ║
║  Version : 10.0                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  WHAT IS NEW IN v10                                                          ║
║  ─────────────────                                                           ║
║  1. BIOPHYSICAL DOSE OPTIMIZER  (replaces static table lookup)               ║
║     Derives required power and time from first principles using the          ║
║     Pennes Bioheat equation. Iterates to convergence accounting for          ║
║     heat-sink energy loss, tissue thermal conductivity, and perfusion.       ║
║     Outputs a continuous (P_opt, t_opt) prescription — not a table row.     ║
║                                                                              ║
║  2. TUMOR TYPE + CONSISTENCY SELECTION                                       ║
║     User selects tumor histological type (HCC, colorectal met, etc.)        ║
║     and tissue consistency (soft / firm / hard).  Each type carries         ║
║     its own thermal conductivity, perfusion correction, and dielectric       ║
║     property factors that adjust the dose accordingly.                       ║
║                                                                              ║
║  3. ITERATIVE HEAT-SINK COMPENSATION LOOP                                    ║
║     Power is escalated iteratively until the ablation zone covers the        ║
║     tumor + margin, accounting for the energy stolen by each nearby          ║
║     vessel at each power level. If power alone cannot solve it, the          ║
║     algorithm switches to time-extension mode.                               ║
║                                                                              ║
║  4. FULL 3-PHASE WORKFLOW PRESERVED                                          ║
║     Phase 1 — Overview visualisation (all tumors)                            ║
║     Phase 2 — User selects tumor + inputs type and consistency               ║
║     Phase 3 — Optimized treatment plan + animated visualisation              ║
╚══════════════════════════════════════════════════════════════════════════════╝

EXECUTION FLOW
──────────────
  Run script
     │
     ▼
  Phase 1: 3D Overview window — all tumors labelled, all vessels shown
     │  (close window to proceed)
     ▼
  Phase 2: Terminal — tumor metrics table
           User types tumor number
           User selects tumor type  (1–6 menu)
           User selects consistency (1–3 menu)
     │
     ▼
  Biophysical optimizer runs:
    - Computes tissue-adjusted thermal parameters
    - Computes heat-sink Q_loss for all vessels at trial power
    - Iterates (P, t) until zone diameter >= required with OAR safety
    - Outputs optimal (P_opt, t_opt, zone_diam, zone_fwd)
     │
     ▼
  ASI Risk Index computed on optimizer output
     │
     ▼
  Phase 3: Animated treatment-planning visualisation
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
#  FILE PATHS
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
#  PHYSICAL CONSTANTS  (blood)
# ══════════════════════════════════════════════════════════════════════════════

RHO_B   = 1060.0    # kg/m³
MU_B    = 3.5e-3    # Pa·s
C_B     = 3700.0    # J/(kg·K)
K_B     = 0.52      # W/(m·K)
T_BLOOD = 37.0      # °C
T_ABL   = 60.0      # °C  — cell-death isotherm (conservative)
T_TISS  = 90.0      # °C  — ablation visualisation temperature

ALPHA_TISSUE    = 70.0   # tissue thermal attenuation  1/m
L_SEG           = 0.01   # vessel contact segment length  m
OAR_MIN_CLEAR_M = 5e-3   # 5 mm OAR clearance

MIN_DIAMETER_CM = 3.0
MAX_DIAMETER_CM = 5.0
MAX_DEPTH_CM    = 26.0

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
#  TUMOR BIOLOGY LIBRARY
#  ─────────────────────
#  Each entry encodes:
#    k_tissue   — thermal conductivity  W/(m·K)
#    rho_cp     — volumetric heat capacity  J/(m³·K)
#    omega_b    — blood perfusion rate  1/s
#    epsilon_r  — relative permittivity at 2.45 GHz
#    sigma      — electrical conductivity  S/m
#    k_factor   — dose scaling factor (>1 = harder to ablate)
#    description— clinical note
#
#  Sources: Haemmerich 2003, Brace 2011, Rossmann & Haemmerich 2014
# ══════════════════════════════════════════════════════════════════════════════

TUMOR_TYPES = {
    "HCC": {
        "label":       "Hepatocellular Carcinoma (HCC)",
        "k_tissue":    0.52,     # W/(m·K)
        "rho_cp":      3.6e6,    # J/(m³·K)
        "omega_b":     0.0064,   # 1/s  — hypervascular
        "epsilon_r":   43.0,
        "sigma":       1.69,     # S/m
        "k_factor":    1.00,     # baseline
        "description": "Hypervascular; standard MWA response",
    },
    "COLORECTAL": {
        "label":       "Colorectal Liver Metastasis",
        "k_tissue":    0.48,
        "rho_cp":      3.8e6,
        "omega_b":     0.0030,   # hypovascular
        "epsilon_r":   39.5,
        "sigma":       1.55,
        "k_factor":    1.12,     # denser, needs more energy
        "description": "Hypovascular, denser; requires ~12% more energy",
    },
    "NEUROENDOCRINE": {
        "label":       "Neuroendocrine Tumour Metastasis",
        "k_tissue":    0.55,
        "rho_cp":      3.5e6,
        "omega_b":     0.0090,   # highly vascular
        "epsilon_r":   45.0,
        "sigma":       1.75,
        "k_factor":    0.93,     # more vascular, slightly easier
        "description": "Highly vascular; slight dose reduction possible",
    },
    "CHOLANGIO": {
        "label":       "Cholangiocarcinoma / Biliary Origin",
        "k_tissue":    0.44,
        "rho_cp":      4.0e6,
        "omega_b":     0.0020,   # fibrotic, hypovascular
        "epsilon_r":   37.0,
        "sigma":       1.40,
        "k_factor":    1.22,     # fibrotic — hardest to ablate
        "description": "Fibrotic, low conductivity; needs ~22% more energy",
    },
    "FATTY_BACKGROUND": {
        "label":       "Tumour in Fatty/Cirrhotic Liver",
        "k_tissue":    0.38,
        "rho_cp":      3.2e6,
        "omega_b":     0.0015,   # cirrhotic — low perfusion
        "epsilon_r":   34.0,
        "sigma":       1.20,
        "k_factor":    1.30,     # low conductivity background
        "description": "Fatty/cirrhotic liver background; zone spreads differently",
    },
    "UNKNOWN": {
        "label":       "Unknown / Not Biopsied",
        "k_tissue":    0.50,
        "rho_cp":      3.7e6,
        "omega_b":     0.0050,
        "epsilon_r":   41.0,
        "sigma":       1.60,
        "k_factor":    1.10,     # conservative 10% uplift
        "description": "Conservative estimate — treat as moderately difficult",
    },
}

CONSISTENCY_FACTORS = {
    "soft":  {"label": "Soft  (necrotic core, cystic, well-vascularised)",
              "dose_factor": 0.90, "note": "Easier ablation — 10% dose reduction"},
    "firm":  {"label": "Firm  (solid, typical)",
              "dose_factor": 1.00, "note": "Standard dose"},
    "hard":  {"label": "Hard  (fibrotic, desmoplastic, calcified)",
              "dose_factor": 1.20, "note": "Resistant — 20% dose increase required"},
}

# ASI weights
ASI_WEIGHTS = {"hss": 0.35, "ocm": 0.30, "cc": 0.20, "dra": 0.15}

# Optimizer bounds
P_MIN_W  = 20.0
P_MAX_W  = 200.0
T_MIN_S  = 60.0
T_MAX_S  = 900.0
MAX_ITER = 60
CONV_TOL = 0.005   # 0.5% convergence


# ══════════════════════════════════════════════════════════════════════════════
#  MESH UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def load_vtk(path):
    if not os.path.exists(path):
        print(f"  ✘ Missing: {path}")
        return None
    m = pv.read(path)
    print(f"  ✔ {os.path.basename(path)}  ({m.n_points} pts, {m.n_cells} cells)")
    return m

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
    print("\n🔍 Extracting individual tumors...")
    tumors = tumor_mesh.connectivity().split_bodies()
    print(f"   Detected {len(tumors)} tumor(s)")
    return tumors

def tumor_metrics(tumors, surface, vessels, vnames):
    s_tree  = cKDTree(np.array(surface.points))
    v_trees = [cKDTree(np.array(v.points)) for v in vessels]
    metrics = []
    for i, t in enumerate(tumors):
        c   = np.array(t.center)
        b   = t.bounds
        dm  = max(b[1]-b[0], b[3]-b[2], b[5]-b[4])
        dep = float(s_tree.query(c, k=1)[0])
        vd  = [float(vt.query(c, k=1)[0]) for vt in v_trees]
        elig = (MIN_DIAMETER_CM <= dm*100 <= MAX_DIAMETER_CM
                and dep*100 <= MAX_DEPTH_CM)
        metrics.append({
            "idx": i, "centroid": c,
            "diameter_cm": dm * 100.0, "depth_cm": dep * 100.0,
            "vessel_dists_m": vd, "min_vessel_m": min(vd),
            "closest_vessel": vnames[int(np.argmin(vd))],
            "eligible": elig,
        })
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  HEAT-SINK PHYSICS ENGINE  (unchanged from v9)
# ══════════════════════════════════════════════════════════════════════════════

def nusselt_full(Re, Pr):
    if Re < 2300:
        return 4.36
    f  = (0.790 * np.log(Re) - 1.64) ** (-2)
    Nu = (f/8) * (Re-1000) * Pr / (1.0 + 12.7 * np.sqrt(f/8) * (Pr**(2/3)-1))
    if Re >= 10000:
        Nu = 0.023 * Re**0.8 * Pr**0.4
    return max(Nu, 4.36)

def wall_layer_correction(Re, D):
    if Re < 2300:
        return 1.0
    f     = (0.790 * np.log(Re) - 1.64) ** (-2)
    nu    = MU_B / RHO_B
    u_tau = 0.25 * np.sqrt(f/8)
    dv    = 5.0 * nu / (u_tau + 1e-9)
    Pr    = (C_B * MU_B) / K_B
    dt    = dv * Pr**(-1/3)
    return max(0.90, 1.0 - dt / (D/2.0))

def heat_sink_physics(distance_m, vessel_name, power_w, time_s):
    D      = VESSEL_DIAMETERS[vessel_name]
    u_mean = VESSEL_VELOCITIES[vessel_name]
    Re     = (RHO_B * u_mean * D) / MU_B
    Pr     = (C_B * MU_B) / K_B
    Nu     = nusselt_full(Re, Pr)
    eta    = wall_layer_correction(Re, D)
    h_bulk = (Nu * K_B) / D
    h_wall = h_bulk * eta
    A_c    = (D/2.0) * (np.pi/3.0) * L_SEG
    A_f    = np.pi * D * L_SEG
    dTw    = max(T_TISS - T_BLOOD, 0.1)
    dTb    = max((T_TISS + T_BLOOD)/2.0 - T_BLOOD, 0.1)
    Qw     = h_wall * A_c * dTw
    bw     = 0.30 if Re >= 2300 else 0.05
    Qbulk  = bw * h_bulk * A_f * dTb
    Qv     = min(Qw + Qbulk, power_w)
    d      = max(distance_m, 1e-4)
    Q_loss = min(Qv * np.exp(-ALPHA_TISSUE * d), power_w)
    E_in   = power_w * time_s
    E_loss = min(Q_loss * time_s, E_in)
    regime = ("Laminar" if Re < 2300 else
              "Transition" if Re < 10000 else "Turbulent")
    return {
        "vessel": vessel_name, "dist_mm": d*1000,
        "Re": Re, "Pr": Pr, "Nu": Nu, "flow_regime": regime,
        "eta_wall": eta, "h_bulk": h_bulk, "h_wall": h_wall,
        "Q_loss_W": Q_loss, "E_loss_J": E_loss,
        "loss_pct": 100.0 * E_loss / E_in,
        "Q_wall_W": Qw, "Q_bulk_W": Qbulk,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ┌─────────────────────────────────────────────────────────────────────────┐
#  │   BIOPHYSICAL DOSE OPTIMIZER  —  CORE NOVEL CONTRIBUTION  v10          │
#  └─────────────────────────────────────────────────────────────────────────┘
#
#  Derives the ablation zone radius achievable by depositing power P
#  for time t in tissue characterised by (k_t, omega_b, rho_cp) using
#  the Pennes Bioheat steady-state spherical source approximation:
#
#       r_abl = sqrt( P_eff / (4π · k_t · (T_abl - T_blood) · Γ) )
#
#  where Γ = sqrt(ω_b · ρ_b · c_b / k_t)  is the perfusion decay constant
#  and   P_eff = P_net · (1 - exp(-t / τ))   is the effective deposited power
#  accounting for thermal build-up time τ = rho_cp / (omega_b * rho_b * c_b).
#
#  The optimizer iterates:
#    Step 1  Compute total Q_sink from all vessels at current (P, t)
#    Step 2  P_net = P - Q_sink  (net power available to heat tissue)
#    Step 3  r_abl = biophysical_radius(P_net, t, tissue_params)
#    Step 4  If r_abl >= r_required → converged
#            Else → increase P by delta_P and repeat
#    Step 5  If P hits ceiling → extend time, reset P to midpoint
#    Step 6  Final OAR clearance check on achieved zone
# ══════════════════════════════════════════════════════════════════════════════

def biophysical_zone_radius(P_net_w, time_s, tissue):
    """
    Compute ablation radius (m) using Pennes steady-state spherical model.
    P_net_w  : net power after heat-sink subtraction  (W)
    time_s   : ablation duration  (s)
    tissue   : dict from TUMOR_TYPES with k_tissue, rho_cp, omega_b
    """
    k_t    = tissue["k_tissue"]          # W/(m·K)
    rho_cp = tissue["rho_cp"]            # J/(m³·K)
    omega  = tissue["omega_b"]           # 1/s
    rho_b  = RHO_B                       # kg/m³
    c_b    = C_B                         # J/(kg·K)

    # Perfusion decay length (m)
    gamma  = np.sqrt(omega * rho_b * c_b / k_t)

    # Thermal time constant (s)
    tau    = rho_cp / max(omega * rho_b * c_b, 1e-6)

    # Effective power fraction deposited (approaches 1 as t >> tau)
    eff    = 1.0 - np.exp(-time_s / max(tau, 1e-3))

    # Net effective power
    P_eff  = max(P_net_w * eff, 0.1)

    # Spherical bioheat radius
    denom  = 4.0 * np.pi * k_t * (T_ABL - T_BLOOD) * max(gamma, 1e-3)
    r_abl  = np.sqrt(max(P_eff / denom, 1e-6))

    # Clamp to physically plausible range (5 mm – 60 mm)
    return float(np.clip(r_abl, 0.005, 0.060))


def total_heat_sink(centroid_dists, vnames, power_w, time_s):
    """Sum Q_loss across all vessels at given (power, time)."""
    total_q = 0.0
    per_hs  = {}
    for vn in vnames:
        hs = heat_sink_physics(centroid_dists[vn], vn, power_w, time_s)
        per_hs[vn] = hs
        total_q   += hs["Q_loss_W"]
    return min(total_q, power_w * 0.85), per_hs


def oar_zone_clearance(zone_r_m, centroid_dists, vnames):
    """
    Returns dict: vessel_name → wall_clearance_m
    Negative means the zone encroaches the vessel wall.
    """
    clr = {}
    for vn in vnames:
        clr[vn] = centroid_dists[vn] - VESSEL_RADII[vn] - zone_r_m
    return clr


def run_biophysical_optimizer(tumor_diam_cm, tumor_type_key, consistency_key,
                               centroid_dists, vnames,
                               margin_cm=0.5):
    """
    Iterative heat-sink compensated dose optimizer.

    Returns
    -------
    result : dict
        P_opt       — optimal power  (W)
        t_opt       — optimal time   (s)
        zone_diam_cm— predicted ablation diameter  (cm)
        zone_fwd_cm — predicted forward extent  (cm)  (1.25× diameter for MWA)
        Q_sink_W    — total heat sink power at optimum
        per_vessel_hs — per-vessel heat sink dict
        clearances  — vessel wall clearances at optimum zone
        constrained — True if OAR constraint could not be fully satisfied
        converged   — True if optimizer converged before MAX_ITER
        iterations  — number of iterations taken
        log         — list of iteration strings for display
        tissue      — tissue params used
        consistency — consistency params used
    """
    tissue  = TUMOR_TYPES[tumor_type_key]
    consist = CONSISTENCY_FACTORS[consistency_key]

    # Required ablation radius (with margin) in metres
    r_req_m = ((tumor_diam_cm + margin_cm) / 2.0) / 100.0

    # Combined dose scaling factor from tumor biology + consistency
    dose_sf = tissue["k_factor"] * consist["dose_factor"]

    # Starting power: heuristic based on required radius and tissue properties
    # From inverted biophysical model at t=300s
    k_t    = tissue["k_tissue"]
    omega  = tissue["omega_b"]
    gamma  = np.sqrt(omega * RHO_B * C_B / k_t)
    tau    = tissue["rho_cp"] / max(omega * RHO_B * C_B, 1e-6)
    eff300 = 1.0 - np.exp(-300.0 / max(tau, 1e-3))
    denom  = 4.0 * np.pi * k_t * (T_ABL - T_BLOOD) * max(gamma, 1e-3)
    P_start = np.clip(denom * r_req_m**2 / max(eff300, 0.01) * dose_sf,
                      P_MIN_W, P_MAX_W)

    P_cur   = P_start
    t_cur   = 300.0    # start at 5 minutes
    delta_P = 5.0      # power step per iteration  (W)
    delta_T = 30.0     # time step when power is capped  (s)

    converged   = False
    constrained = False
    log         = []
    per_hs_final = {}

    print(f"\n{'─'*65}")
    print(f"  BIOPHYSICAL DOSE OPTIMIZER")
    print(f"{'─'*65}")
    print(f"  Tumor type   : {tissue['label']}")
    print(f"  Consistency  : {consist['label']}")
    print(f"  Dose scale   : ×{dose_sf:.3f}  "
          f"(type ×{tissue['k_factor']:.2f} × consistency ×{consist['dose_factor']:.2f})")
    print(f"  Required r   : {r_req_m*100:.2f} cm  "
          f"(tumor {tumor_diam_cm:.2f} cm + margin {margin_cm:.1f} cm)")
    print(f"  Starting P   : {P_cur:.1f} W   t : {t_cur:.0f} s")
    print(f"{'─'*65}")
    print(f"  {'Iter':>4}  {'P(W)':>7}  {'t(s)':>6}  {'Q_sink(W)':>10}  "
          f"{'P_net(W)':>9}  {'r_abl(cm)':>10}  {'Status'}")
    print(f"  {'─'*60}")

    for it in range(1, MAX_ITER + 1):
        # Step 1: compute heat sink at current (P, t)
        Q_sink, per_hs = total_heat_sink(centroid_dists, vnames, P_cur, t_cur)

        # Step 2: net power
        P_net = max(P_cur - Q_sink, 0.5)

        # Step 3: predicted ablation radius
        r_abl = biophysical_zone_radius(P_net, t_cur, tissue)

        # OAR clearance at this zone
        clr   = oar_zone_clearance(r_abl, centroid_dists, vnames)
        min_cl = min(clr.values())
        oar_ok = min_cl >= OAR_MIN_CLEAR_M

        status = ""
        if r_abl >= r_req_m and oar_ok:
            status = "✔ CONVERGED"
            converged     = True
            per_hs_final  = per_hs
            break
        elif r_abl >= r_req_m and not oar_ok:
            status = "⚠ OAR ENCROACH"
            constrained   = True
            per_hs_final  = per_hs
            # Can't increase zone — accept and break
            break
        elif P_cur >= P_MAX_W:
            # Hit power ceiling — extend time
            status = f"↑ time (P={P_cur:.0f}W ceiling)"
            t_cur  = min(t_cur + delta_T, T_MAX_S)
            if t_cur >= T_MAX_S:
                status = "✘ TIME LIMIT"
                per_hs_final = per_hs
                constrained  = True
                break
        else:
            status = "↑ power"
            P_cur  = min(P_cur + delta_P, P_MAX_W)

        row = (f"  {it:>4}  {P_cur:>7.1f}  {t_cur:>6.0f}  "
               f"{Q_sink:>10.3f}  {P_net:>9.3f}  "
               f"{r_abl*100:>10.3f}  {status}")
        print(row)
        log.append(row)
        per_hs_final = per_hs

    # Final state
    Q_sink_f, per_hs_final = total_heat_sink(centroid_dists, vnames, P_cur, t_cur)
    P_net_f  = max(P_cur - Q_sink_f, 0.5)
    r_abl_f  = biophysical_zone_radius(P_net_f, t_cur, tissue)
    clr_f    = oar_zone_clearance(r_abl_f, centroid_dists, vnames)

    # MWA zones are typically prolate: forward ~1.25× diameter
    zone_diam_cm = r_abl_f * 2.0 * 100.0
    zone_fwd_cm  = zone_diam_cm * 1.25

    print(f"\n  {'─'*60}")
    print(f"  OPTIMIZER RESULT:")
    print(f"    Power        : {P_cur:.1f} W")
    print(f"    Time         : {t_cur:.0f} s  ({t_cur/60:.1f} min)")
    print(f"    Q_sink total : {Q_sink_f:.3f} W")
    print(f"    P_net        : {P_net_f:.3f} W")
    print(f"    Zone diameter: {zone_diam_cm:.2f} cm")
    print(f"    Zone forward : {zone_fwd_cm:.2f} cm")
    print(f"    Converged    : {'YES' if converged else 'NO'}")
    print(f"    Constrained  : {'YES — OAR encroachment' if constrained else 'NO'}")
    print(f"  {'─'*60}")

    clearance_report = [
        {"vessel": vn, "wall_clear_mm": v * 1000}
        for vn, v in clr_f.items()
    ]

    return {
        "P_opt":          P_cur,
        "t_opt":          t_cur,
        "zone_diam_cm":   zone_diam_cm,
        "zone_fwd_cm":    zone_fwd_cm,
        "Q_sink_W":       Q_sink_f,
        "P_net_W":        P_net_f,
        "per_vessel_hs":  per_hs_final,
        "clearances":     clr_f,
        "clearance_report": clearance_report,
        "constrained":    constrained,
        "converged":      converged,
        "iterations":     it,
        "log":            log,
        "tissue":         tissue,
        "consistency":    consist,
        "dose_sf":        dose_sf,
        "r_required_cm":  r_req_m * 100.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  RAY UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def generate_rays(n_theta=20, n_phi=40):
    rays = []
    for t in np.linspace(0, np.pi, n_theta):
        for p in np.linspace(0, 2*np.pi, n_phi):
            rays.append([np.sin(t)*np.cos(p),
                         np.sin(t)*np.sin(p), np.cos(t)])
    return np.array(rays)

def ray_segment_dist(origin, direction, path_d, vessel_pts,
                     fallback_dist, n_sample=30):
    ts      = np.linspace(0.0, path_d, n_sample)
    samples = origin + np.outer(ts, direction)
    dists, _ = cKDTree(vessel_pts).query(samples, k=1)
    return max(float(np.min(dists)), fallback_dist * 0.5)


# ══════════════════════════════════════════════════════════════════════════════
#  OAR IDENTIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def identify_oars(centroid, vessels, vnames, fwd_cm, diam_cm, needle_dir=None):
    a     = (fwd_cm  / 2.0) / 100.0
    b     = (diam_cm / 2.0) / 100.0
    n_hat = (np.array(needle_dir) / (np.linalg.norm(needle_dir) + 1e-9)
             if needle_dir is not None else np.array([0., 0., 1.]))
    oars  = []
    for vessel, vname in zip(vessels, vnames):
        pts    = np.array(vessel.points)
        rel    = pts - centroid
        ax     = rel.dot(n_hat)
        perp   = np.linalg.norm(rel - np.outer(ax, n_hat), axis=1)
        inside = (ax/a)**2 + (perp/b)**2 <= 1.0
        n_in   = int(inside.sum())
        if n_in > 0:
            cl_c    = float(np.min(np.linalg.norm(rel[inside], axis=1)))
            v_r     = VESSEL_RADII.get(vname, 0.0)
            cl_wall = max(cl_c - v_r, 0.0)
            risk    = "CRITICAL" if cl_wall < OAR_MIN_CLEAR_M else "HIGH"
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


# ══════════════════════════════════════════════════════════════════════════════
#  STAGED PLAN
# ══════════════════════════════════════════════════════════════════════════════

def compute_staged_plan(centroid, needle_dir, centroid_dists,
                        vnames, opt_result):
    """Two overlapping sub-zones at reduced power that each respect OAR."""
    if not vnames:
        return None, None
    vn_oar       = min(centroid_dists, key=centroid_dists.get)
    wall_dist_m  = centroid_dists[vn_oar] - VESSEL_RADII.get(vn_oar, 0.0)
    max_r_m      = max(wall_dist_m - OAR_MIN_CLEAR_M - 2e-3, 0.005)
    sub_diam_m   = max_r_m * 2.0
    sub_fwd_m    = sub_diam_m * 1.25
    sub_P        = opt_result["P_opt"] * 0.70
    sub_t        = opt_result["t_opt"] * 1.10
    nd           = np.array(needle_dir, dtype=float)
    nd          /= np.linalg.norm(nd) + 1e-9
    overlap      = 0.30 * sub_fwd_m
    offset       = sub_fwd_m - overlap
    return [
        {"centre": centroid - nd * offset/2.0,
         "fwd_m": sub_fwd_m, "diam_m": sub_diam_m,
         "label": f"Stage 1: {sub_P:.0f}W × {sub_t:.0f}s"},
        {"centre": centroid + nd * offset/2.0,
         "fwd_m": sub_fwd_m, "diam_m": sub_diam_m,
         "label": f"Stage 2: {sub_P:.0f}W × {sub_t:.0f}s"},
    ], (sub_P, sub_t, sub_diam_m*100)


# ══════════════════════════════════════════════════════════════════════════════
#  ABLATION SAFETY INDEX  (ASI)
# ══════════════════════════════════════════════════════════════════════════════

def compute_asi(per_vessel_hs, clearance_report, tumor_diam_cm,
                zone_diam_cm, ray_losses, constrained):
    max_loss  = max(hs["loss_pct"] for hs in per_vessel_hs.values())
    hss_score = float(np.clip(100.0 * (1.0 - max_loss / 50.0), 0, 100))

    if clearance_report:
        min_cl_mm = min(cr["wall_clear_mm"] for cr in clearance_report)
    else:
        min_cl_mm = 20.0
    ocm_score = float(np.clip(100.0 * min_cl_mm / 20.0, 0, 100))

    margin_mm = (zone_diam_cm - tumor_diam_cm) * 10.0
    cc_score  = float(np.clip(100.0 * margin_mm / 10.0, 0, 100))
    if constrained:
        cc_score *= 0.55

    if len(ray_losses) > 1:
        spread    = float(np.max(ray_losses) - np.min(ray_losses))
        dra_score = float(np.clip(100.0 * (1.0 - spread / 30.0), 0, 100))
    else:
        dra_score = 50.0

    w   = ASI_WEIGHTS
    asi = (w["hss"]*hss_score + w["ocm"]*ocm_score +
           w["cc"]*cc_score   + w["dra"]*dra_score)

    risk = ("LOW"      if asi >= 75 else
            "MODERATE" if asi >= 50 else
            "HIGH"     if asi >= 30 else "CRITICAL")

    interp = {
        "LOW":      "Ablation expected to achieve complete coverage with low risk.",
        "MODERATE": "Vessel proximity may reduce zone — monitor margins carefully.",
        "HIGH":     "Significant heat sink; optimizer has escalated dose to compensate.",
        "CRITICAL": "Zone compromised — staged treatment or repositioning strongly advised.",
    }[risk]

    return {
        "asi": round(asi, 1), "hss_score": round(hss_score, 1),
        "ocm_score": round(ocm_score, 1), "cc_score": round(cc_score, 1),
        "dra_score": round(dra_score, 1), "risk_label": risk,
        "max_loss_pct": round(max_loss, 2),
        "min_clear_mm": round(min_cl_mm, 1),
        "margin_mm": round(margin_mm, 1),
        "spread_pct": round(float(np.max(ray_losses) - np.min(ray_losses))
                            if len(ray_losses) > 1 else 0.0, 2),
        "interpretation": interp,
    }

def print_asi(asi):
    bar_len = 40
    filled  = int(round(asi["asi"] / 100.0 * bar_len))
    sym     = {"LOW":"🟢","MODERATE":"🟡","HIGH":"🟠","CRITICAL":"🔴"}[asi["risk_label"]]
    bar     = sym * filled + "⬜" * (bar_len - filled)
    print("\n" + "═"*70)
    print("  ABLATION SAFETY INDEX  (ASI)")
    print("═"*70)
    print(f"  Overall ASI : {asi['asi']:>5.1f} / 100   [{asi['risk_label']}]")
    print(f"  {bar}")
    print(f"\n  Sub-scores:")
    print(f"  {'Heat Sink Severity':<32} HSS = {asi['hss_score']:>5.1f}  (w=0.35)"
          f"   max loss {asi['max_loss_pct']:.2f}%")
    print(f"  {'OAR Clearance Margin':<32} OCM = {asi['ocm_score']:>5.1f}  (w=0.30)"
          f"   min wall {asi['min_clear_mm']:.1f} mm")
    print(f"  {'Coverage Confidence':<32}  CC = {asi['cc_score']:>5.1f}  (w=0.20)"
          f"   margin {asi['margin_mm']:.1f} mm")
    print(f"  {'Directional Risk Asymmetry':<32} DRA = {asi['dra_score']:>5.1f}  (w=0.15)"
          f"   spread {asi['spread_pct']:.2f}%")
    print(f"\n  ▶  {asi['interpretation']}")
    print("═"*70)


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
        _, _, vt = np.linalg.svd(cen[:min(5000, len(cen))],
                                  full_matrices=False)
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
#  ELLIPSOID
# ══════════════════════════════════════════════════════════════════════════════

def make_ellipsoid(centroid, fwd_m, diam_m, needle_dir=None):
    if fwd_m < 1e-4 or diam_m < 1e-4:
        return None
    a, c = diam_m/2.0, fwd_m/2.0
    ell  = pv.ParametricEllipsoid(xradius=a, yradius=a, zradius=c,
                                   u_res=30, v_res=30, w_res=10)
    if needle_dir is not None:
        n = np.array(needle_dir, dtype=float)
        n /= np.linalg.norm(n) + 1e-9
        z    = np.array([0., 0., 1.])
        axis = np.cross(z, n)
        an   = np.linalg.norm(axis)
        if an > 1e-6:
            axis /= an
            ang   = np.degrees(np.arccos(np.clip(np.dot(z, n), -1, 1)))
            ell   = ell.rotate_vector(axis, ang, inplace=False)
    ell.points += centroid
    rn = np.linalg.norm(ell.points - centroid, axis=1) / (max(a, c) + 1e-9)
    ell["Temperature_C"] = T_BLOOD + (T_TISS - T_BLOOD) * np.exp(-2.0 * rn**2)
    return ell

def create_heat_flow_arrows(centroid, vessels, vnames, per_vessel_hs, plotter):
    losses = [hs["loss_pct"] for hs in per_vessel_hs.values()]
    mx, mn = max(losses), min(losses)
    BASE   = 0.04
    def col(p):
        t = (p-mn)/max(mx-mn, 0.01)
        return [2*t, 1.0, 0.0] if t < 0.5 else [1.0, 2*(1-t), 0.0]
    for vn, hs in per_vessel_hs.items():
        if vn not in vnames: continue
        vessel = vessels[vnames.index(vn)]
        pts    = np.array(vessel.points)
        _, idx = cKDTree(pts).query(centroid, k=1)
        raw    = pts[idx] - centroid
        dist   = np.linalg.norm(raw)
        if dist < 1e-6: continue
        unit    = raw / dist
        arr_len = max(BASE * hs["loss_pct"] / max(mx, 1.), 0.005)
        plotter.add_mesh(
            pv.Arrow(start=centroid, direction=unit, scale=arr_len,
                     tip_length=0.3, tip_radius=0.05, shaft_radius=0.02),
            color=col(hs["loss_pct"]), opacity=0.95)
        plotter.add_point_labels(
            pv.PolyData([centroid + unit * arr_len * 1.2]),
            [f"{vn.replace('_',' ')}\n{hs['loss_pct']:.2f}%\nQ={hs['Q_loss_W']:.3f}W"],
            font_size=9, text_color=col(hs["loss_pct"]),
            point_size=1, always_visible=True, shape_opacity=0.0)
    plotter.add_text("Heat Flow:  Green=Low  Yellow=Mid  Red=High",
                     position="lower_right", font_size=9, color="white")


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — OVERVIEW VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def phase1_overview(surface, vessels, vnames, tumors, metrics):
    print("\n" + "═"*70)
    print("  PHASE 1 — OVERVIEW  (close window to proceed)")
    print("═"*70)

    plotter = pv.Plotter(window_size=[1400, 900],
                         title="OVERVIEW — All Tumors  |  Close to continue")
    plotter.background_color = "black"
    plotter.add_mesh(surface, color="lightgray", opacity=0.07, label="Body Surface")

    for v, vn in zip(vessels, vnames):
        plotter.add_mesh(v, color=VESSEL_COLOR_MAP.get(vn, "gray"),
                         opacity=0.60, label=vn.replace("_"," ").title())

    for i, (t, m) in enumerate(zip(tumors, metrics)):
        tc   = TUMOR_COLORS[i % len(TUMOR_COLORS)]
        elig = "✔ ELIGIBLE" if m["eligible"] else "✗ ineligible"
        plotter.add_mesh(t, color=tc, opacity=0.80,
                         label=f"T{i+1} {m['diameter_cm']:.1f}cm {elig}")
        plotter.add_mesh(pv.Sphere(radius=0.007, center=m["centroid"]),
                         color="white", opacity=0.95)
        plotter.add_point_labels(
            pv.PolyData([m["centroid"] + np.array([0, 0, 0.013])]),
            [f"T{i+1}"], font_size=14, text_color=tc,
            point_size=1, always_visible=True, shape_opacity=0.0)

    plotter.add_axes()
    plotter.add_legend(loc="upper right", size=(0.28, 0.52))
    plotter.add_text(
        "PHASE 1 — All Tumors Overview\n"
        "White sphere = centroid  |  Coloured label = tumor number\n"
        "Close this window to select a tumor for analysis",
        position="upper_left", font_size=11, color="white")
    plotter.add_text(
        "Vessels:\n  Red = Aorta  |  Med Blue = Portal vein\n"
        "  Dodger Blue = Hepatic v + IVC  |  Orange = Hepatic artery",
        position="lower_left", font_size=10, color="lightgray")

    try:
        plotter.show(auto_close=False)
    except Exception as e:
        print(f"  Visualisation error (non-fatal): {e}")
    finally:
        plotter.close()


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — TUMOR + BIOLOGY SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def _pick_menu(title, options_dict):
    """Generic numbered menu picker. Returns chosen key."""
    keys   = list(options_dict.keys())
    labels = [options_dict[k]["label"] for k in keys]
    print(f"\n  {title}")
    for i, lbl in enumerate(labels, 1):
        print(f"    {i}. {lbl}")
    while True:
        try:
            raw = input(f"  ▶  Enter choice [1–{len(keys)}]: ").strip()
            n   = int(raw)
            if 1 <= n <= len(keys):
                return keys[n - 1]
            print(f"  ✘ Enter a number between 1 and {len(keys)}.")
        except ValueError:
            print("  ✘ Invalid — enter an integer.")
        except (EOFError, KeyboardInterrupt):
            print("\n  Exiting.")
            sys.exit(0)

def phase2_pick_tumor(metrics, vnames):
    print("\n" + "═"*70)
    print("  PHASE 2 — TUMOR & BIOLOGY SELECTION")
    print("═"*70)

    # ── Tumor table ────────────────────────────────────────────────────
    print(f"\n  {'#':<5} {'Diam(cm)':<11} {'Depth(cm)':<11} "
          f"{'Closest vessel':<20} {'Dist(mm)':<11} {'Eligible?'}")
    print("  " + "─"*70)
    for m in metrics:
        elig = "✔ YES" if m["eligible"] else "✗ NO "
        print(f"  {m['idx']+1:<5} {m['diameter_cm']:<11.2f} "
              f"{m['depth_cm']:<11.2f} {m['closest_vessel']:<20} "
              f"{m['min_vessel_m']*1000:<11.1f} {elig}")

    eligible_ids = [m["idx"]+1 for m in metrics if m["eligible"]]
    if eligible_ids:
        print(f"\n  ✔ Eligible tumors (MWA criteria 3–5 cm, depth ≤26 cm): {eligible_ids}")
    else:
        print("\n  ⚠  No tumors meet standard MWA criteria — any tumor can still be selected.")

    # ── Tumor number ───────────────────────────────────────────────────
    while True:
        try:
            raw = input(f"\n  ▶  Enter tumor number to analyse [1–{len(metrics)}]: ").strip()
            n   = int(raw)
            if 1 <= n <= len(metrics):
                sel = metrics[n - 1]
                print(f"\n  ✔ Tumor {n} selected  "
                      f"({sel['diameter_cm']:.2f} cm, depth {sel['depth_cm']:.2f} cm, "
                      f"closest: {sel['closest_vessel']} @ {sel['min_vessel_m']*1000:.1f} mm)")
                break
            print(f"  ✘ Enter 1–{len(metrics)}.")
        except ValueError:
            print("  ✘ Invalid input.")
        except (EOFError, KeyboardInterrupt):
            print("\n  Exiting.")
            sys.exit(0)

    # ── Tumor type ─────────────────────────────────────────────────────
    print("\n  ─────────────────────────────────────────────")
    print("  SELECT TUMOR HISTOLOGICAL TYPE")
    print("  (affects thermal conductivity, perfusion, dose)")
    type_key = _pick_menu("Tumor type:", TUMOR_TYPES)
    print(f"  ✔ Type: {TUMOR_TYPES[type_key]['label']}")
    print(f"     → {TUMOR_TYPES[type_key]['description']}")

    # ── Consistency ────────────────────────────────────────────────────
    print("\n  ─────────────────────────────────────────────")
    print("  SELECT TUMOR CONSISTENCY")
    print("  (from pre-procedure imaging / radiologist report)")
    consist_key = _pick_menu("Consistency:", CONSISTENCY_FACTORS)
    print(f"  ✔ Consistency: {CONSISTENCY_FACTORS[consist_key]['label']}")
    print(f"     → {CONSISTENCY_FACTORS[consist_key]['note']}")

    return sel, type_key, consist_key


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 3 — TREATMENT PLANNING VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def phase3_visualise(surface, vessels, vnames, tumors, centroids,
                     sel_idx, results, opt_result, asi,
                     oar_list, safest_dir, particle_systems,
                     centroid_dists):

    print("\n🎬  Building treatment-planning visualisation...")

    power_w    = opt_result["P_opt"]
    time_s     = opt_result["t_opt"]
    fwd_m      = opt_result["zone_fwd_cm"]  / 100.0
    diam_m     = opt_result["zone_diam_cm"] / 100.0
    centroid   = centroids[sel_idx]
    needle_dir = safest_dir
    constrained= opt_result["constrained"]
    per_hs     = opt_result["per_vessel_hs"]
    tissue     = opt_result["tissue"]
    consist    = opt_result["consistency"]

    plotter = pv.Plotter(
        window_size=[1500, 1000],
        title=(f"Treatment Plan — Tumor {sel_idx+1}  |  "
               f"{power_w:.0f}W × {time_s:.0f}s  |  "
               f"ASI {asi['asi']:.1f} [{asi['risk_label']}]"))
    plotter.background_color = "black"

    # Body surface
    plotter.add_mesh(surface, color="lightgray", opacity=0.07, label="Body Surface")

    # Vessels — OARs highlighted red
    for v, vn in zip(vessels, vnames):
        is_oar = any(o["vessel"] == vn for o in oar_list)
        col    = VESSEL_COLOR_MAP.get(vn, "gray")
        if is_oar:
            plotter.add_mesh(v, color="red", opacity=0.90,
                             label=f"⚠ OAR: {vn.replace('_',' ').title()}")
        else:
            plotter.add_mesh(v, color=col, opacity=0.60,
                             label=vn.replace("_"," ").title())

    # Tumors
    for i, t in enumerate(tumors):
        td    = smooth_tumor(t) if i == sel_idx else t
        op    = 0.85 if i == sel_idx else 0.22
        label = f"Tumor {i+1} [TARGET]" if i == sel_idx else f"Tumor {i+1}"
        plotter.add_mesh(td, color=TUMOR_COLORS[i % len(TUMOR_COLORS)],
                         opacity=op, label=label)
    plotter.add_mesh(pv.Sphere(radius=0.006, center=centroid),
                     color="white", label="Tumour centroid")

    # Heat flow arrows
    create_heat_flow_arrows(centroid, vessels, vnames, per_hs, plotter)

    # Ray lines
    ray_actor_names, ray_meshes = [], []
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
            vn  = oar["vessel"]
            er  = VESSEL_RADII.get(vn, 0.005) + OAR_MIN_CLEAR_M
            ctr = oar.get("nearest_pt", centroid)
            sph = pv.Sphere(radius=er, center=ctr,
                            theta_resolution=20, phi_resolution=20)
            plotter.add_mesh(sph, color="red", opacity=0.18,
                             label=f"OAR exclusion: {vn.replace('_',' ')}")
            plotter.add_mesh(sph, color="red", opacity=0.55,
                             style="wireframe", line_width=1.2)

    # Needle reposition arrow
    if constrained and oar_list:
        closest_oar = min(oar_list, key=lambda o: o["wall_clear_mm"])
        oar_pt      = closest_oar.get("nearest_pt", centroid)
        away        = centroid - oar_pt
        an          = np.linalg.norm(away)
        away        = away / an if an > 1e-6 else np.array([0., 1., 0.])
        need_mm     = OAR_MIN_CLEAR_M*1000 - closest_oar["wall_clear_mm"]
        plotter.add_mesh(
            pv.Arrow(start=centroid, direction=away, scale=0.04,
                     tip_length=0.25, tip_radius=0.08, shaft_radius=0.03),
            color="cyan", opacity=0.95, label="Suggested needle shift")
        plotter.add_point_labels(
            pv.PolyData([centroid + away * 0.048]),
            [f"↑ Shift needle\nAway from: {closest_oar['vessel'].replace('_',' ')}\n"
             f"Need: {need_mm:.1f}mm more"],
            font_size=10, text_color="cyan",
            point_size=1, always_visible=True, shape_opacity=0.0)

    # Staged plan
    staged_plan = None
    if constrained:
        staged_plan, sub_info = compute_staged_plan(
            centroid, needle_dir, centroid_dists, vnames, opt_result)
        if sub_info:
            print(f"  Staged plan: 2× {sub_info[0]:.0f}W × {sub_info[1]:.0f}s "
                  f"(each diam {sub_info[2]:.1f}cm)")

    # ── animation state ───────────────────────────────────────────────
    mode_state = {"staged": False, "rays_on": True}
    asi_col    = {"LOW":"lime","MODERATE":"yellow",
                  "HIGH":"orange","CRITICAL":"tomato"}[asi["risk_label"]]

    def clear_dynamic():
        for nm in ["ablation","ablation_s1","ablation_s2","particles","hud"]:
            try: plotter.remove_actor(nm)
            except: pass

    def draw_single(frac):
        ell = make_ellipsoid(centroid, fwd_m*frac, diam_m*frac, needle_dir)
        if ell:
            plotter.add_mesh(ell, scalars="Temperature_C", cmap="plasma",
                             clim=[T_BLOOD, T_TISS], opacity=0.62,
                             name="ablation",
                             scalar_bar_args={"title":"Temperature (°C)",
                                              "n_labels":5,"label_font_size":11,
                                              "title_font_size":12,"position_x":0.02,
                                              "position_y":0.25,"width":0.08,
                                              "height":0.40,"color":"white"})
        return fwd_m*frac, diam_m*frac

    def draw_staged(frac):
        if not staged_plan: return 0, 0
        for idx_s, stage in enumerate(staged_plan):
            ell = make_ellipsoid(stage["centre"],
                                 stage["fwd_m"]*frac, stage["diam_m"]*frac,
                                 needle_dir)
            if ell:
                plotter.add_mesh(ell, scalars="Temperature_C", cmap="plasma",
                                 clim=[T_BLOOD,T_TISS], opacity=0.55,
                                 name=f"ablation_s{idx_s+1}",
                                 scalar_bar_args={"title":"Temperature (°C)",
                                                  "n_labels":4,"label_font_size":10,
                                                  "title_font_size":11,"position_x":0.02,
                                                  "position_y":0.25,"width":0.07,
                                                  "height":0.35,"color":"white"})
        s = staged_plan[0]
        return s["fwd_m"], s["diam_m"]

    def update(t_val):
        t    = float(t_val)
        frac = min(t / max(time_s, 1.0), 1.0)
        clear_dynamic()

        if mode_state["staged"] and staged_plan:
            cf, cd   = draw_staged(frac)
            mlabel   = "⚡ STAGED (2× partial)"
            snote    = (f"  {staged_plan[0]['label']}\n"
                        f"  {staged_plan[1]['label']}\n"
                        f"  each D={cd*100:.1f}cm F={cf*100:.1f}cm")
        else:
            cf, cd   = draw_single(frac)
            mlabel   = "● Single zone"
            snote    = ""

        # Particles
        all_pts, all_vel = [], []
        for ps in particle_systems:
            pts, vel = ps.update(t)
            all_pts.append(pts); all_vel.append(vel)
        if all_pts:
            cloud = pv.PolyData(np.vstack(all_pts))
            cloud["blood_velocity_m_s"] = np.concatenate(all_vel)
            plotter.add_mesh(cloud, scalars="blood_velocity_m_s", cmap="coolwarm",
                             clim=[0.0, max(VESSEL_VELOCITIES.values())*2.0],
                             point_size=5, render_points_as_spheres=True,
                             name="particles",
                             scalar_bar_args={"title":"Blood velocity (m/s)",
                                              "n_labels":3,"label_font_size":10,
                                              "title_font_size":11,"position_x":0.12,
                                              "position_y":0.25,"width":0.08,
                                              "height":0.40,"color":"white"})

        zone_str = (f"Zone: {cf*100:.1f}cm × {cd*100:.1f}cm"
                    if frac > 0.01 else "Zone: growing...")
        warn_line = ("⚠  CONSTRAINED — OAR encroachment\n"
                     if constrained else "✔  OAR-SAFE (optimizer)\n")

        hud = (
            f"{'─'*32}\n"
            f"  ABLATION SAFETY INDEX\n"
            f"  ASI={asi['asi']:.1f}/100  [{asi['risk_label']}]\n"
            f"  HSS={asi['hss_score']:.0f} OCM={asi['ocm_score']:.0f} "
            f"CC={asi['cc_score']:.0f} DRA={asi['dra_score']:.0f}\n"
            f"{'─'*32}\n"
            f"  BIOPHYSICAL OPTIMIZER\n"
            f"  P={power_w:.0f}W  t={time_s:.0f}s ({time_s/60:.1f}min)\n"
            f"  Q_sink={opt_result['Q_sink_W']:.2f}W\n"
            f"  P_net={opt_result['P_net_W']:.2f}W\n"
            f"  Dose factor ×{opt_result['dose_sf']:.2f}\n"
            f"{'─'*32}\n"
            f"  {tissue['label'][:26]}\n"
            f"  k={tissue['k_tissue']:.2f} W/mK\n"
            f"  ω={tissue['omega_b']:.4f} 1/s\n"
            f"{'─'*32}\n"
            f"{warn_line}"
            f"  Mode: {mlabel}\n"
            f"  t={t:.0f}s/{time_s:.0f}s ({frac*100:.0f}%)\n"
            f"  {zone_str}\n"
            + (f"{'─'*32}\nStaged:\n{snote}\n" if snote else "")
            + f"{'─'*32}\n"
            f"  OARs: {len(oar_list)}\n"
            f"  Ablation: Purple→White\n"
            f"  Blood: Blue=slow Red=fast\n"
        )
        plotter.add_text(hud, position="lower_left", font_size=9,
                         color=asi_col, name="hud")
        plotter.render()

    # ── controls ──────────────────────────────────────────────────────
    play_state = {"playing": False, "t": 0.0}

    def toggle_play(flag):
        play_state["playing"] = bool(flag)
        if play_state["playing"]:
            plotter.add_timer_event(max_steps=100000, duration=100,
                                    callback=_tick)

    def _tick(step):
        if not play_state["playing"]: return
        play_state["t"] = (play_state["t"] + 5.0) % (time_s + 1.0)
        slider_w.GetRepresentation().SetValue(play_state["t"])
        update(play_state["t"])

    slider_w = plotter.add_slider_widget(
        update, rng=[0.0, time_s], value=0.0,
        title="Ablation Time (s)",
        pointa=(0.22, 0.05), pointb=(0.90, 0.05), style="modern")

    plotter.add_checkbox_button_widget(
        toggle_play, value=False, position=(30, 30),
        size=45, border_size=3, color_on="lime", color_off="gray")
    plotter.add_text("▶ Play", position=(80, 38), font_size=11, color="white")

    def toggle_rays(flag):
        mode_state["rays_on"] = bool(flag)
        if flag:
            for nm, (lm, col) in zip(ray_actor_names, ray_meshes):
                plotter.add_mesh(lm, color=col, line_width=2.5,
                                 opacity=0.55, name=nm)
        else:
            for nm in ray_actor_names:
                try: plotter.remove_actor(nm)
                except: pass
        plotter.render()

    plotter.add_checkbox_button_widget(
        toggle_rays, value=True, position=(30, 100),
        size=45, border_size=3, color_on="yellow", color_off="dimgray")
    plotter.add_text("◉ Ray lines", position=(80, 108),
                     font_size=11, color="yellow")

    if constrained and staged_plan:
        def toggle_staged(flag):
            mode_state["staged"] = bool(flag)
            update(play_state["t"])
        plotter.add_checkbox_button_widget(
            toggle_staged, value=False, position=(30, 170),
            size=45, border_size=3, color_on="cyan", color_off="dimgray")
        plotter.add_text("⚡ Staged 2×", position=(80, 178),
                         font_size=11, color="cyan")

    plotter.add_legend(loc="upper right", size=(0.27, 0.48))
    plotter.add_text(
        f"Biophysical MWA Plan — Tumor {sel_idx+1}  |  "
        f"{power_w:.0f}W × {time_s:.0f}s  |  "
        + ("⚠ CONSTRAINED" if constrained else "✔ OAR-SAFE")
        + f"  |  ASI {asi['asi']:.1f} [{asi['risk_label']}]",
        position="upper_left", font_size=12, color=asi_col)
    plotter.add_axes()

    update(0.0)
    try:
        plotter.show(auto_close=False)
    except Exception as e:
        print(f"  Vis error: {e}")
    finally:
        plotter.close()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔" + "═"*68 + "╗")
    print("║  ADAPTIVE MWA PLANNING SYSTEM  v10                               ║")
    print("║  Biophysical Dose Optimizer  +  ASI Risk Index                   ║")
    print("╚" + "═"*68 + "╝")

    if not os.path.exists(DATASET_BASE):
        print(f"\n  ✘ Dataset not found: {DATASET_BASE}")
        return

    # ── Load ───────────────────────────────────────────────────────────
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

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1
    # ══════════════════════════════════════════════════════════════════
    phase1_overview(surface, vessels, vnames, tumors, metrics)

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2
    # ══════════════════════════════════════════════════════════════════
    sel, type_key, consist_key = phase2_pick_tumor(metrics, vnames)
    sel_idx  = sel["idx"]
    centroid = sel["centroid"]
    sel_diam = sel["diameter_cm"]

    # Per-vessel distances from chosen centroid
    centroid_dists = {
        vnames[i]: float(cKDTree(np.array(v.points)).query(centroid, k=1)[0])
        for i, v in enumerate(vessels)
    }

    # ── Ray tracing ────────────────────────────────────────────────────
    print("\n  Ray tracing (directional heat-sink map)...")
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
            hs     = heat_sink_physics(seg_d[dom_vn], dom_vn, 60.0, 300.0)
            hs["ray_direction"] = direction
            hs["path_distance"] = path_d
            results.append(hs)
        except Exception:
            continue

    all_losses = [r["loss_pct"] for r in results]
    sorted_res = sorted(results, key=lambda x: x["loss_pct"], reverse=True)
    safest_dir = (sorted_res[-1]["ray_direction"]
                  if results else np.array([0., 0., 1.]))
    print(f"  {len(results)} rays | "
          f"loss {np.min(all_losses):.2f}% – {np.max(all_losses):.2f}%")

    # ══════════════════════════════════════════════════════════════════
    # BIOPHYSICAL OPTIMIZER
    # ══════════════════════════════════════════════════════════════════
    opt_result = run_biophysical_optimizer(
        tumor_diam_cm  = sel_diam,
        tumor_type_key = type_key,
        consistency_key= consist_key,
        centroid_dists = centroid_dists,
        vnames         = vnames,
        margin_cm      = 0.5,
    )

    # OAR identification on optimized zone
    oar_list = identify_oars(
        centroid, vessels, vnames,
        opt_result["zone_fwd_cm"],
        opt_result["zone_diam_cm"],
        safest_dir)

    print(f"\n  OARs encroached: {len(oar_list)}")
    for o in oar_list:
        print(f"    {o['vessel']}  pts_inside={o['points_inside']}  "
              f"wall={o['wall_clear_mm']:.1f}mm  [{o['risk']}]")

    # Per-vessel heat sink at final optimized power
    per_vessel_hs_final = opt_result["per_vessel_hs"]

    # ── ASI ────────────────────────────────────────────────────────────
    asi = compute_asi(
        per_vessel_hs    = per_vessel_hs_final,
        clearance_report = opt_result["clearance_report"],
        tumor_diam_cm    = sel_diam,
        zone_diam_cm     = opt_result["zone_diam_cm"],
        ray_losses       = all_losses,
        constrained      = opt_result["constrained"],
    )
    print_asi(asi)

    # ── Final prescription summary ─────────────────────────────────────
    print("\n" + "═"*70)
    print("  FINAL TREATMENT PRESCRIPTION  (Biophysical Optimizer Output)")
    print("═"*70)
    print(f"  Tumor         : {sel_idx+1}  ({sel_diam:.2f} cm,  depth {sel['depth_cm']:.2f} cm)")
    print(f"  Histology     : {TUMOR_TYPES[type_key]['label']}")
    print(f"  Consistency   : {CONSISTENCY_FACTORS[consist_key]['label']}")
    print(f"  Dose factor   : ×{opt_result['dose_sf']:.3f}")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Power         : {opt_result['P_opt']:.1f} W")
    print(f"  Time          : {opt_result['t_opt']:.0f} s  "
          f"({opt_result['t_opt']/60:.1f} min)")
    print(f"  Zone diameter : {opt_result['zone_diam_cm']:.2f} cm")
    print(f"  Zone forward  : {opt_result['zone_fwd_cm']:.2f} cm")
    print(f"  Q_sink total  : {opt_result['Q_sink_W']:.3f} W")
    print(f"  P_net         : {opt_result['P_net_W']:.3f} W")
    print(f"  Converged     : {'YES' if opt_result['converged'] else 'NO'}")
    print(f"  Constrained   : "
          f"{'YES — OAR encroachment; staged plan shown' if opt_result['constrained'] else 'NO'}")
    print(f"  ASI           : {asi['asi']:.1f} / 100  [{asi['risk_label']}]")
    print("═"*70)

    # ── Particles ──────────────────────────────────────────────────────
    print("\n  Building blood particle systems...")
    particle_systems = []
    for v, vn in zip(vessels, vnames):
        ps = VesselParticleSystem(v, vn, n_particles=80)
        particle_systems.append(ps)
        print(f"   {vn}: {ps.n} particles, Re={ps.Re:.0f} "
              f"({'Laminar' if ps.Re<2300 else 'Turbulent/Transition'})")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 3
    # ══════════════════════════════════════════════════════════════════
    phase3_visualise(
        surface, vessels, vnames, tumors, centroids,
        sel_idx, results, opt_result, asi,
        oar_list, safest_dir, particle_systems,
        centroid_dists)

    print("\n  ✔  Analysis complete.")
    return opt_result, asi


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  Stopped.")
    except Exception as e:
        print(f"\n  Error: {e}")
        raise
