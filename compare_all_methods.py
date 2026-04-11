#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   MASTER COMPARISON RUNNER — MWA REGIME SELECTION STUDY                    ║
║   Integrates Method 1 (Table), Method 2 (Physics-Only),                    ║
║              Method 3 (MOO), Method 4 (ML)                                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  HOW TO USE                                                                  ║
║  ──────────                                                                  ║
║  Option A — Full pipeline (run after phase2 in your existing scripts):      ║
║    from compare_all_methods import run_all_methods_comparison               ║
║    compare_results = run_all_methods_comparison(                            ║
║        tumor_diam_cm   = sel_diam,                                          ║
║        centroid_dists  = centroid_dists,                                   ║
║        vnames          = vnames,                                            ║
║        type_key        = type_key,      # from hs1_directional_mwa.py      ║
║        consist_key     = consist_key,   # from hs1_directional_mwa.py      ║
║        depth_cm        = sel["depth_cm"],                                   ║
║        ray_losses      = all_losses,    # from ray tracing                 ║
║    )                                                                         ║
║                                                                              ║
║  Option B — Standalone demo (no VTK files needed):                          ║
║    python compare_all_methods.py                                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Import Methods 3 and 4
from method3_moo_optimizer import run_method3, compute_asi_moo
from method4_ml_predictor  import run_method4, print_master_comparison

# ─────────────────────────────────────────────────────────────────────────────
# PHYSICAL CONSTANTS  (shared, identical to all other files)
# ─────────────────────────────────────────────────────────────────────────────

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
VESSEL_RADII = {vn: d/2.0 for vn, d in VESSEL_DIAMETERS.items()}

# Manufacturer ablation table (same as in both existing files)
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

# Tumor types (from hs1_directional_mwa.py)
TUMOR_TYPES = {
    "HCC":           {"k_factor":1.00,"label":"Hepatocellular Carcinoma","k_tissue":0.52,"rho_cp":3.6e6,"omega_b":0.0064},
    "COLORECTAL":    {"k_factor":1.12,"label":"Colorectal Metastasis",  "k_tissue":0.48,"rho_cp":3.8e6,"omega_b":0.0030},
    "NEUROENDOCRINE":{"k_factor":0.93,"label":"Neuroendocrine Tumour",  "k_tissue":0.55,"rho_cp":3.5e6,"omega_b":0.0090},
    "CHOLANGIO":     {"k_factor":1.22,"label":"Cholangiocarcinoma",      "k_tissue":0.44,"rho_cp":4.0e6,"omega_b":0.0020},
    "FATTY_BACKGROUND":{"k_factor":1.30,"label":"Tumour in Fatty Liver","k_tissue":0.38,"rho_cp":3.2e6,"omega_b":0.0015},
    "UNKNOWN":       {"k_factor":1.10,"label":"Unknown",                 "k_tissue":0.50,"rho_cp":3.7e6,"omega_b":0.0050},
}
CONSISTENCY_FACTORS = {
    "soft": {"dose_factor":0.90,"label":"Soft"},
    "firm": {"dose_factor":1.00,"label":"Firm"},
    "hard": {"dose_factor":1.20,"label":"Hard"},
}


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 1 — TABLE-BASED (BASELINE)
# ─────────────────────────────────────────────────────────────────────────────

def _nusselt(Re, Pr):
    if Re < 2300: return 4.36
    f  = (0.790*np.log(Re)-1.64)**(-2)
    Nu = (f/8)*(Re-1000)*Pr/(1.0+12.7*np.sqrt(f/8)*(Pr**(2/3)-1))
    if Re >= 10000: Nu = 0.023*Re**0.8*Pr**0.4
    return max(Nu, 4.36)

def _hs(d_m, vn, P, t):
    D=VESSEL_DIAMETERS[vn]; u=VESSEL_VELOCITIES[vn]
    Re=(RHO_B*u*D)/MU_B; Pr=(C_B*MU_B)/K_B; Nu=_nusselt(Re,Pr)
    eta=max(0.9,1.0) if Re<2300 else max(0.90,1.0)
    hb=(Nu*K_B)/D; hw=hb*eta
    Ac=(D/2)*(np.pi/3)*L_SEG; Af=np.pi*D*L_SEG
    dTw=max(T_TISS-T_BLOOD,0.1); dTb=max((T_TISS+T_BLOOD)/2-T_BLOOD,0.1)
    Qv=min(hw*Ac*dTw+(0.30 if Re>=2300 else 0.05)*hb*Af*dTb, P)
    d=max(d_m,1e-4); Ql=min(Qv*np.exp(-ALPHA_TISSUE*d),P)
    Ei=P*t; El=min(Ql*t,Ei)
    return {"vessel":vn,"dist_mm":d*1000,"Re":Re,"Pr":Pr,"Nu":Nu,
            "flow_regime":"Laminar" if Re<2300 else "Turbulent",
            "Q_loss_W":Ql,"E_loss_J":El,"loss_pct":100.0*El/max(Ei,1e-9),
            "Q_wall_W":hw*Ac*dTw,"Q_bulk_W":(0.30 if Re>=2300 else 0.05)*hb*Af*dTb}

def run_method1_table(tumor_diam_cm, centroid_dists, vnames,
                       margin_cm=0.5, verbose=True):
    """
    Method 1: Pure table lookup — picks nearest diameter ≥ required.
    No heat-sink, no histology, no OAR awareness.
    This is the control group / baseline.
    """
    req = tumor_diam_cm + margin_cm
    cands = [(P,t,vol,fwd,diam) for P,t,vol,fwd,diam in ABLATION_TABLE if diam >= req]
    if not cands:
        cands = sorted(ABLATION_TABLE, key=lambda r: r[4], reverse=True)
    rec = sorted(cands, key=lambda r: (r[4], r[0], r[1]))[0]
    P_rec, t_rec, vol_rec, fwd_rec, diam_rec = rec

    # Compute heat-sink post-hoc (for fair ASI comparison)
    per_hs = {vn: _hs(centroid_dists[vn], vn, P_rec, t_rec) for vn in vnames}
    zone_r = (diam_rec/2.0)/100.0
    clr    = {vn: centroid_dists[vn]-VESSEL_RADII[vn]-zone_r for vn in vnames}
    min_cl = min(clr.values())
    cr     = [{"vessel":vn,"wall_clear_mm":v*1000} for vn,v in clr.items()]
    Q_sink = sum(hs["Q_loss_W"] for hs in per_hs.values())

    if verbose:
        print(f"\n  [Method 1 — Table]  {P_rec:.0f}W × {t_rec:.0f}s  "
              f"diam={diam_rec:.2f}cm  min_clear={min_cl*1000:.1f}mm")

    return {
        "method":"TableBased","P_opt":P_rec,"t_opt":t_rec,
        "zone_diam_cm":diam_rec,"zone_fwd_cm":fwd_rec,"min_clear_mm":min_cl*1000,
        "per_vessel_hs":per_hs,"clearance_report":cr,
        "constrained":min_cl<OAR_MIN_CLEAR_M,"converged":True,
        "margin_cm":margin_cm,"dose_sf":1.0,"Q_sink_W":Q_sink,"P_net_W":P_rec,
    }


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 2 — PHYSICS-ONLY (NO CLINICAL HEURISTICS)
# ─────────────────────────────────────────────────────────────────────────────

def pennes_r(Pnet, t):
    """Default HCC tissue properties, no histology correction."""
    kt=0.52; rcp=3.6e6; omega=0.0064
    gam=np.sqrt(omega*RHO_B*C_B/kt); tau=rcp/max(omega*RHO_B*C_B,1e-6)
    eff=1.0-np.exp(-t/max(tau,1e-3))
    Peff=max(Pnet*eff,0.1); den=4*np.pi*kt*(T_ABL-T_BLOOD)*max(gam,1e-3)
    return float(np.clip(np.sqrt(max(Peff/den,1e-6)),0.005,0.080))

def run_method2_physics_only(tumor_diam_cm, centroid_dists, vnames,
                              margin_cm=0.5, verbose=True):
    """
    Method 2: Physics-only optimization.
    Uses heat-sink model to compensate power, but:
      • No histology correction (k_factor = 1.0)
      • No consistency correction (dose_factor = 1.0)
      • No directional SAR
      • No OAR-aware selection
    Optimizes over table entries only (matches regime format).
    """
    req_r = ((tumor_diam_cm + margin_cm)/2.0)/100.0
    best  = None; best_cost = np.inf

    for P, t, vol, fwd, diam in ABLATION_TABLE:
        Q_total = sum(_hs(centroid_dists[vn],vn,P,t)["Q_loss_W"] for vn in vnames)
        Pnet    = max(P - Q_total, 0.5)
        r_abl   = pennes_r(Pnet, t)
        zone_diam = r_abl * 2.0 * 100.0

        if zone_diam < (tumor_diam_cm + margin_cm):
            cost = 1.0 + (tumor_diam_cm + margin_cm - zone_diam)
        else:
            cost = (P * t) / (160 * 900)   # minimise energy if coverage met

        if cost < best_cost:
            best_cost = cost
            best = (P, t, vol, fwd, zone_diam, Pnet, Q_total)

    P_rec, t_rec, vol_rec, fwd_rec, diam_rec, Pnet_rec, Q_rec = best
    per_hs = {vn: _hs(centroid_dists[vn],vn,P_rec,t_rec) for vn in vnames}
    zone_r = (diam_rec/2.0)/100.0
    clr    = {vn: centroid_dists[vn]-VESSEL_RADII[vn]-zone_r for vn in vnames}
    min_cl = min(clr.values())
    cr     = [{"vessel":vn,"wall_clear_mm":v*1000} for vn,v in clr.items()]

    if verbose:
        print(f"\n  [Method 2 — PhysicsOnly]  {P_rec:.0f}W × {t_rec:.0f}s  "
              f"zone={diam_rec:.2f}cm  P_net={Pnet_rec:.1f}W  "
              f"Q_sink={Q_rec:.3f}W  min_clear={min_cl*1000:.1f}mm")

    return {
        "method":"PhysicsOnly","P_opt":P_rec,"t_opt":t_rec,
        "zone_diam_cm":diam_rec,"zone_fwd_cm":fwd_rec,"min_clear_mm":min_cl*1000,
        "per_vessel_hs":per_hs,"clearance_report":cr,
        "constrained":min_cl<OAR_MIN_CLEAR_M,"converged":True,
        "margin_cm":margin_cm,"dose_sf":1.0,"Q_sink_W":Q_rec,"P_net_W":Pnet_rec,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ASI COMPUTATION  (v9 formula, omnidirectional — Methods 1, 2, 3)
# ─────────────────────────────────────────────────────────────────────────────

def compute_asi(opt_result, tumor_diam_cm, ray_losses=None):
    per_hs = opt_result["per_vessel_hs"]
    cr     = opt_result["clearance_report"]
    zone   = opt_result["zone_diam_cm"]
    const  = opt_result["constrained"]
    max_loss  = max(hs["loss_pct"] for hs in per_hs.values())
    hss_score = float(np.clip(100.0*(1.0-max_loss/50.0),0,100))
    min_cl_mm = min(c["wall_clear_mm"] for c in cr) if cr else 20.0
    ocm_score = float(np.clip(100.0*min_cl_mm/20.0,0,100))
    margin_mm = (zone-tumor_diam_cm)*10.0
    cc_score  = float(np.clip(100.0*margin_mm/10.0,0,100))
    if const: cc_score *= 0.60
    if ray_losses and len(ray_losses)>1:
        spread=float(np.max(ray_losses)-np.min(ray_losses))
        dra_score=float(np.clip(100.0*(1.0-spread/30.0),0,100))
    else:
        dra_score=50.0
    w={"hss":0.35,"ocm":0.30,"cc":0.20,"dra":0.15}
    asi=w["hss"]*hss_score+w["ocm"]*ocm_score+w["cc"]*cc_score+w["dra"]*dra_score
    risk=("LOW" if asi>=75 else "MODERATE" if asi>=50 else "HIGH" if asi>=30 else "CRITICAL")
    return {
        "asi":round(asi,1),"hss_score":round(hss_score,1),
        "ocm_score":round(ocm_score,1),"cc_score":round(cc_score,1),
        "dra_score":round(dra_score,1),"risk_label":risk,
        "max_loss_pct":round(max_loss,2),"min_clear_mm":round(min_cl_mm,1),
        "margin_mm":round(margin_mm,1),
        "spread_pct":round(float(np.max(ray_losses)-np.min(ray_losses))
                           if ray_losses and len(ray_losses)>1 else 0.0,2),
        "method":opt_result.get("method",""),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MASTER RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_all_methods_comparison(
        tumor_diam_cm, centroid_dists, vnames,
        type_key="HCC", consist_key="firm",
        depth_cm=10.0, margin_cm=0.5, ray_losses=None,
        verbose=True):
    """
    Run all four regime selection methods on the same patient case and
    produce a unified comparison table.

    Parameters
    ----------
    tumor_diam_cm  : float — measured tumor diameter in cm
    centroid_dists : dict  — {vessel_name: distance_m} from tumor_metrics()
    vnames         : list  — vessel names present
    type_key       : str   — histology key from TUMOR_TYPES
    consist_key    : str   — consistency key from CONSISTENCY_FACTORS
    depth_cm       : float — insertion depth from tumor_metrics
    margin_cm      : float — ablation margin (default 0.5 cm)
    ray_losses     : list  — from ray tracing (for DRA sub-score)
    verbose        : bool  — print intermediate results

    Returns
    -------
    all_results : dict {method_label: result_dict}
    all_asis    : dict {method_label: asi_dict}
    """
    tissue   = TUMOR_TYPES[type_key]
    consist  = CONSISTENCY_FACTORS[consist_key]
    k_fac    = tissue["k_factor"]
    d_fac    = consist["dose_factor"]
    dose_sf  = k_fac * d_fac

    tissue_props = {
        "k_tissue": tissue["k_tissue"],
        "rho_cp":   tissue["rho_cp"],
        "omega_b":  tissue["omega_b"],
    }

    print("\n" + "╔" + "═"*70 + "╗")
    print("║  MASTER COMPARISON — ALL 4 REGIME SELECTION METHODS                     ║")
    print("╚" + "═"*70 + "╝")
    print(f"\n  Case:  Tumor {tumor_diam_cm:.2f} cm  |  "
          f"Histology: {tissue['label']}  |  "
          f"Consistency: {consist['label']}")
    print(f"         dose_sf = {dose_sf:.3f}  |  depth = {depth_cm:.1f} cm  |  "
          f"margin = {margin_cm} cm")

    all_results = {}
    all_asis    = {}

    # ── Method 1: Table
    print(f"\n{'━'*72}")
    print("  METHOD 1 — TABLE-BASED (Pure Manufacturer Table, No Physics)")
    print(f"{'━'*72}")
    r1 = run_method1_table(tumor_diam_cm, centroid_dists, vnames, margin_cm, verbose)
    a1 = compute_asi(r1, tumor_diam_cm, ray_losses)
    all_results["M1_Table"]   = r1
    all_asis["M1_Table"]      = a1

    # ── Method 2: Physics-Only
    print(f"\n{'━'*72}")
    print("  METHOD 2 — PHYSICS-ONLY (Heat-Sink, No Histology/Consistency/Direction)")
    print(f"{'━'*72}")
    r2 = run_method2_physics_only(tumor_diam_cm, centroid_dists, vnames, margin_cm, verbose)
    a2 = compute_asi(r2, tumor_diam_cm, ray_losses)
    all_results["M2_Physics"] = r2
    all_asis["M2_Physics"]    = a2

    # ── Method 3: MOO (Grid Search + GA)
    print(f"\n{'━'*72}")
    print("  METHOD 3 — MULTI-OBJECTIVE OPTIMIZATION (Grid Search + Genetic Algorithm)")
    print(f"{'━'*72}")
    m3_out = run_method3(
        tumor_diam_cm=tumor_diam_cm,
        centroid_dists=centroid_dists,
        vnames=vnames,
        tissue_props=tissue_props,
        margin_cm=margin_cm,
        dose_sf=dose_sf,
        ray_losses=ray_losses,
        verbose=verbose,
    )
    gs_result, gs_asi = m3_out[0], m3_out[1]
    ga_result, ga_asi = m3_out[4], m3_out[5]
    all_results["M3_GridSrch"] = gs_result; all_asis["M3_GridSrch"] = gs_asi
    all_results["M3_GA"]       = ga_result; all_asis["M3_GA"]       = ga_asi

    # ── Method 4: ML
    print(f"\n{'━'*72}")
    print("  METHOD 4 — ML-BASED PREDICTOR (Random Forest + XGBoost/GBM)")
    print(f"{'━'*72}")
    m4_out = run_method4(
        tumor_diam_cm=tumor_diam_cm,
        centroid_dists=centroid_dists,
        vnames=vnames,
        tissue_props=tissue_props,
        k_factor=k_fac,
        dose_factor=d_fac,
        depth_cm=depth_cm,
        margin_cm=margin_cm,
        ray_losses=ray_losses,
        verbose=verbose,
    )
    m4_results, m4_asis = m4_out[0], m4_out[1]
    for model_name, res in m4_results.items():
        label = f"M4_{model_name[:7]}"
        all_results[label] = res
        all_asis[label]    = m4_asis[model_name]

    # ── Master comparison table
    print_master_comparison(all_results, all_asis)

    # ── Ranking by ASI
    print("\n  RANKING BY ASI SCORE (highest = safest / best):")
    ranked = sorted(all_asis.items(), key=lambda x: x[1]["asi"], reverse=True)
    for rank, (label, asi) in enumerate(ranked, 1):
        r = all_results[label]
        print(f"  {rank}. {label:<18}  ASI={asi['asi']:.1f}  [{asi['risk_label']}]  "
              f"{r['P_opt']:.0f}W × {r['t_opt']:.0f}s  "
              f"zone={r['zone_diam_cm']:.2f}cm  "
              f"clear={r['min_clear_mm']:.1f}mm")

    return all_results, all_asis


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  MASTER COMPARISON DEMO  (synthetic patient case)")

    demo_tumor  = 3.5    # cm
    demo_dists  = {
        "portal_vein":   0.018,
        "hepatic_vein":  0.025,
        "aorta":         0.060,
        "ivc":           0.045,
        "hepatic_artery":0.030,
    }
    demo_vnames = list(demo_dists.keys())

    np.random.seed(7)
    demo_rays = np.random.uniform(0.5, 8.0, 200).tolist()

    results, asis = run_all_methods_comparison(
        tumor_diam_cm  = demo_tumor,
        centroid_dists = demo_dists,
        vnames         = demo_vnames,
        type_key       = "COLORECTAL",
        consist_key    = "firm",
        depth_cm       = 12.0,
        margin_cm      = 0.5,
        ray_losses     = demo_rays,
        verbose        = True,
    )

    print(f"\n  ✔  Complete. {len(results)} methods compared.")
