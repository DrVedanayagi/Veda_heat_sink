#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   METHOD 3 — MULTI-OBJECTIVE OPTIMIZATION REGIME SELECTOR                  ║
║   For Microwave Ablation Planning Comparison Study                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Author  : Veda Nunna (algorithm design)                                    ║
║  Version : 1.0                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  SCIENTIFIC BASIS                                                            ║
║  ──────────────                                                              ║
║  Formulates regime selection as a constrained multi-objective problem:      ║
║                                                                              ║
║    Minimize  J = w1·Undercoverage + w2·OAR_Risk + w3·Energy                ║
║                                                                              ║
║  Subject to:                                                                 ║
║    • Thermal constraints:  Pennes bioheat steady-state (Pennes 1948)        ║
║    • OAR clearance:        wall_clear ≥ OAR_MIN_CLEARANCE_M                ║
║    • Power bounds:         [20, 200] W                                       ║
║    • Time bounds:          [60, 900] s                                       ║
║    • Vessel heat loss:     Gnielinski / Dittus-Boelter correlations          ║
║                                                                              ║
║  Two optimization strategies are implemented and compared:                  ║
║    A) Grid Search — exhaustive over (P, t) with 20×20 resolution            ║
║       Brute-force; guaranteed to find global optimum on grid;               ║
║       transparent and reproducible.                                          ║
║                                                                              ║
║    B) Genetic Algorithm (GA) — population-based stochastic optimizer        ║
║       Population: 40 individuals, 50 generations                            ║
║       Selection: Tournament (k=3)                                            ║
║       Crossover: BLX-α (blend crossover, α=0.3)                             ║
║       Mutation:  Gaussian perturbation, σ=10% of range                      ║
║       Elitism:   top 2 preserved each generation                            ║
║       Ref: Deb et al. (2002) NSGA-II framework (single-objective reduction) ║
║                                                                              ║
║  OUTPUTS (same structure as heatsink_tumorselect.py + hs1_directional_mwa) ║
║    • regime tuple  (P_W, t_s, vol_cc, fwd_cm, diam_cm)                     ║
║    • per-vessel heat-sink dict                                               ║
║    • clearance report                                                        ║
║    • ASI-compatible score dict                                               ║
║    • full Pareto front (grid search) for multi-objective visualisation      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# SHARED PHYSICAL CONSTANTS  (keep in sync with the other two files)
# ─────────────────────────────────────────────────────────────────────────────

RHO_B   = 1060.0    # blood density  kg/m³
MU_B    = 3.5e-3    # dynamic viscosity  Pa·s
C_B     = 3700.0    # blood specific heat  J/(kg·K)
K_B     = 0.52      # blood thermal conductivity  W/(m·K)
T_BLOOD = 37.0      # °C
T_TISS  = 90.0      # °C  (ablation target)
T_ABL   = 60.0      # °C  (cell-death isotherm)

ALPHA_TISSUE    = 70.0    # tissue attenuation  1/m
L_SEG           = 0.01    # vessel contact segment  m
OAR_MIN_CLEAR_M = 5e-3    # 5 mm minimum wall clearance

# Tissue defaults (HCC, standard — override via tissue_props dict)
K_TISSUE_DEFAULT = 0.52       # W/(m·K)
RHO_CP_DEFAULT   = 3.6e6     # J/(m³·K)
OMEGA_B_DEFAULT  = 0.0064    # blood perfusion rate  1/s

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

# ─────────────────────────────────────────────────────────────────────────────
# OPTIMIZER BOUNDS AND WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────

P_MIN_W = 20.0
P_MAX_W = 200.0
T_MIN_S = 60.0
T_MAX_S = 900.0

# Objective weights — tune for clinical priority
# w1: under-coverage penalty (highest priority)
# w2: OAR risk penalty
# w3: energy (minimise unnecessary dose)
OBJ_WEIGHTS = {"undercoverage": 0.50,
               "oar_risk":      0.35,
               "energy":        0.15}

# Grid resolution for grid search (NxN)
GRID_N = 25

# GA parameters
GA_POP_SIZE   = 50
GA_GENERATIONS= 60
GA_TOURNEY_K  = 3
GA_BLX_ALPHA  = 0.30
GA_SIGMA_FRAC = 0.10    # mutation σ as fraction of range
GA_ELITE_N    = 2
GA_SEED       = 42


# ─────────────────────────────────────────────────────────────────────────────
# HEAT-SINK PHYSICS  (identical to v9/v11 for fair comparison)
# ─────────────────────────────────────────────────────────────────────────────

def _nusselt(Re, Pr):
    if Re < 2300:
        return 4.36
    f  = (0.790 * np.log(Re) - 1.64) ** (-2)
    Nu = (f/8) * (Re - 1000) * Pr / (1.0 + 12.7 * np.sqrt(f/8) * (Pr**(2/3) - 1))
    if Re >= 10000:
        Nu = 0.023 * Re**0.8 * Pr**0.4
    return max(Nu, 4.36)

def _wall_correction(Re, D):
    if Re < 2300:
        return 1.0
    f     = (0.790 * np.log(Re) - 1.64) ** (-2)
    nu    = MU_B / RHO_B
    u_tau = 0.25 * np.sqrt(f/8)
    dv    = 5.0 * nu / (u_tau + 1e-9)
    Pr    = (C_B * MU_B) / K_B
    dt    = dv * Pr**(-1/3)
    return max(0.90, 1.0 - dt / (D/2.0))

def vessel_heat_sink(distance_m, vessel_name, power_w, time_s):
    """
    Compute heat-sink energy loss for a single vessel.
    Returns a dict matching the format in heatsink_tumorselect.py.
    """
    D      = VESSEL_DIAMETERS[vessel_name]
    u      = VESSEL_VELOCITIES[vessel_name]
    Re     = (RHO_B * u * D) / MU_B
    Pr     = (C_B * MU_B) / K_B
    Nu     = _nusselt(Re, Pr)
    eta    = _wall_correction(Re, D)
    h_bulk = (Nu * K_B) / D
    h_wall = h_bulk * eta

    Ac     = (D/2.0) * (np.pi/3.0) * L_SEG
    Af     = np.pi * D * L_SEG
    dTw    = max(T_TISS - T_BLOOD, 0.1)
    dTb    = max((T_TISS + T_BLOOD)/2.0 - T_BLOOD, 0.1)
    Qw     = h_wall * Ac * dTw
    bw     = 0.30 if Re >= 2300 else 0.05
    Qbulk  = bw * h_bulk * Af * dTb
    Qv     = min(Qw + Qbulk, power_w)

    d      = max(distance_m, 1e-4)
    Q_loss = min(Qv * np.exp(-ALPHA_TISSUE * d), power_w)
    E_in   = power_w * time_s
    E_loss = min(Q_loss * time_s, E_in)
    regime = ("Laminar"    if Re < 2300 else
              "Transition" if Re < 10000 else "Turbulent")

    return {
        "vessel":      vessel_name,
        "dist_mm":     d * 1000,
        "Re": Re, "Pr": Pr, "Nu": Nu,
        "flow_regime": regime,
        "h_bulk":      h_bulk,
        "h_wall":      h_wall,
        "Q_loss_W":    Q_loss,
        "E_loss_J":    E_loss,
        "loss_pct":    100.0 * E_loss / max(E_in, 1e-9),
    }

def total_heat_sink(centroid_dists, vnames, power_w, time_s):
    """Sum heat-sink across all vessels; return total Q_loss and per-vessel dict."""
    total = 0.0
    per   = {}
    for vn in vnames:
        hs = vessel_heat_sink(centroid_dists[vn], vn, power_w, time_s)
        per[vn]  = hs
        total   += hs["Q_loss_W"]
    return min(total, power_w * 0.85), per


# ─────────────────────────────────────────────────────────────────────────────
# PENNES BIOHEAT — ablation zone radius
# (same physics as directional optimizer, omnidirectional mode)
# ─────────────────────────────────────────────────────────────────────────────

def pennes_zone_radius(P_net_w, time_s, tissue_props):
    """
    Steady-state spherical Pennes bioheat model.
    r_abl ∝ sqrt(P_eff / (4π k_t ΔT γ))
    """
    k_t    = tissue_props.get("k_tissue",  K_TISSUE_DEFAULT)
    rho_cp = tissue_props.get("rho_cp",    RHO_CP_DEFAULT)
    omega  = tissue_props.get("omega_b",   OMEGA_B_DEFAULT)
    gamma  = np.sqrt(omega * RHO_B * C_B / k_t)
    tau    = rho_cp / max(omega * RHO_B * C_B, 1e-6)
    eff    = 1.0 - np.exp(-time_s / max(tau, 1e-3))
    P_eff  = max(P_net_w * eff, 0.1)
    denom  = 4.0 * np.pi * k_t * (T_ABL - T_BLOOD) * max(gamma, 1e-3)
    r_abl  = np.sqrt(max(P_eff / denom, 1e-6))
    return float(np.clip(r_abl, 0.005, 0.080))  # 0.5 – 8 cm


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-OBJECTIVE COST FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def compute_cost(P_w, t_s, tumor_diam_cm, centroid_dists, vnames,
                 tissue_props, margin_cm=0.5, dose_sf=1.0):
    """
    Evaluate the three-term cost function for a (P, t) candidate.

    Returns
    -------
    cost_total   : scalar float (lower is better)
    cost_terms   : dict with individual term values
    zone_diam_cm : achieved ablation diameter in cm
    per_hs       : per-vessel heat-sink dict
    min_clear_m  : minimum wall clearance across all vessels (metres)
    """
    # Effective power after histology/consistency scaling
    P_eff_input = P_w * dose_sf

    # Heat sink
    Q_sink, per_hs = total_heat_sink(centroid_dists, vnames,
                                      P_eff_input, t_s)
    P_net = max(P_eff_input - Q_sink, 0.5)

    # Ablation zone
    r_abl      = pennes_zone_radius(P_net, t_s, tissue_props)
    zone_diam  = r_abl * 2.0 * 100.0           # cm
    target_req = tumor_diam_cm + margin_cm      # required diameter cm

    # ── Term 1: Undercoverage (normalised, 0 = exact coverage, >0 = insufficient)
    under  = max(target_req - zone_diam, 0.0) / max(target_req, 1e-3)

    # ── Term 2: OAR risk (1 - min_wall_clearance / threshold, clipped to [0,1])
    zone_r = (zone_diam / 2.0) / 100.0    # metres
    clr_vals = []
    for vn in vnames:
        wall_cl = centroid_dists[vn] - VESSEL_RADII[vn] - zone_r
        clr_vals.append(wall_cl)
    min_clear = min(clr_vals) if clr_vals else 0.10
    oar_risk  = float(np.clip(1.0 - min_clear / OAR_MIN_CLEAR_M, 0.0, 1.0))

    # ── Term 3: Energy (normalised by maximum possible)
    energy_norm = (P_w * t_s) / (P_MAX_W * T_MAX_S)

    # ── Weighted sum
    w = OBJ_WEIGHTS
    cost = (w["undercoverage"] * under +
            w["oar_risk"]      * oar_risk +
            w["energy"]        * energy_norm)

    return cost, {
        "undercoverage": under,
        "oar_risk":      oar_risk,
        "energy":        energy_norm,
    }, zone_diam, per_hs, min_clear


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY A — GRID SEARCH
# ─────────────────────────────────────────────────────────────────────────────

def grid_search_optimizer(tumor_diam_cm, centroid_dists, vnames,
                           tissue_props, margin_cm=0.5, dose_sf=1.0,
                           verbose=True):
    """
    Exhaustive grid search over [P_MIN, P_MAX] × [T_MIN, T_MAX].

    Returns
    -------
    best_result   : dict with P_opt, t_opt, zone_diam_cm, cost, terms, etc.
    pareto_front  : list of non-dominated solutions (for multi-obj visualisation)
    grid_data     : full NxN numpy array of costs (for heatmap plotting)
    """
    P_vals = np.linspace(P_MIN_W, P_MAX_W, GRID_N)
    t_vals = np.linspace(T_MIN_S, T_MAX_S, GRID_N)

    cost_grid   = np.zeros((GRID_N, GRID_N))
    under_grid  = np.zeros_like(cost_grid)
    oar_grid    = np.zeros_like(cost_grid)
    energy_grid = np.zeros_like(cost_grid)
    diam_grid   = np.zeros_like(cost_grid)
    clr_grid    = np.zeros_like(cost_grid)

    all_solutions = []

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  GRID SEARCH OPTIMIZER  ({GRID_N}×{GRID_N} = {GRID_N**2} evaluations)")
        print(f"  P: [{P_MIN_W:.0f}, {P_MAX_W:.0f}] W   t: [{T_MIN_S:.0f}, {T_MAX_S:.0f}] s")
        print(f"  Weights: undercoverage={OBJ_WEIGHTS['undercoverage']:.2f}  "
              f"oar={OBJ_WEIGHTS['oar_risk']:.2f}  energy={OBJ_WEIGHTS['energy']:.2f}")
        print(f"{'─'*60}")

    for i, P in enumerate(P_vals):
        for j, t in enumerate(t_vals):
            cost, terms, diam, per_hs, min_cl = compute_cost(
                P, t, tumor_diam_cm, centroid_dists, vnames,
                tissue_props, margin_cm, dose_sf)
            cost_grid[i, j]   = cost
            under_grid[i, j]  = terms["undercoverage"]
            oar_grid[i, j]    = terms["oar_risk"]
            energy_grid[i, j] = terms["energy"]
            diam_grid[i, j]   = diam
            clr_grid[i, j]    = min_cl * 1000   # mm
            all_solutions.append({
                "P": P, "t": t, "cost": cost,
                "undercoverage": terms["undercoverage"],
                "oar_risk":      terms["oar_risk"],
                "energy":        terms["energy"],
                "zone_diam_cm":  diam,
                "min_clear_mm":  min_cl * 1000,
                "per_hs":        per_hs,
            })

    # ── Find global minimum
    flat_idx    = np.argmin(cost_grid)
    i_opt, j_opt = np.unravel_index(flat_idx, cost_grid.shape)
    P_opt = P_vals[i_opt]
    t_opt = t_vals[j_opt]

    _, best_terms, best_diam, best_per_hs, best_cl = compute_cost(
        P_opt, t_opt, tumor_diam_cm, centroid_dists, vnames,
        tissue_props, margin_cm, dose_sf)

    # ── Build clearance report (same format as other methods)
    zone_r = (best_diam / 2.0) / 100.0
    clearance_report = [
        {"vessel": vn,
         "wall_clear_mm": (centroid_dists[vn] - VESSEL_RADII[vn] - zone_r) * 1000}
        for vn in vnames
    ]
    constrained = best_cl < OAR_MIN_CLEAR_M

    # ── Pareto front (undercoverage vs oar_risk)
    pareto = _extract_pareto(all_solutions,
                             obj1="undercoverage", obj2="oar_risk")

    if verbose:
        print(f"  ✔  Grid search complete.")
        print(f"  Optimal:  P = {P_opt:.1f} W   t = {t_opt:.0f} s")
        print(f"  Zone diameter : {best_diam:.2f} cm")
        print(f"  Min wall clear: {best_cl*1000:.1f} mm")
        print(f"  Cost breakdown: undercoverage={best_terms['undercoverage']:.4f}  "
              f"oar={best_terms['oar_risk']:.4f}  energy={best_terms['energy']:.4f}")
        print(f"  Total cost    : {cost_grid[i_opt, j_opt]:.4f}")
        print(f"  Pareto front  : {len(pareto)} non-dominated solutions")
        print(f"{'─'*60}")

    best_result = {
        "method":           "GridSearch",
        "P_opt":            P_opt,
        "t_opt":            t_opt,
        "zone_diam_cm":     best_diam,
        "zone_fwd_cm":      best_diam * 1.25,   # approx forward extension
        "min_clear_mm":     best_cl * 1000,
        "cost":             float(cost_grid[i_opt, j_opt]),
        "cost_terms":       best_terms,
        "per_vessel_hs":    best_per_hs,
        "clearance_report": clearance_report,
        "constrained":      constrained,
        "converged":        True,
        "margin_cm":        margin_cm,
        "dose_sf":          dose_sf,
        "Q_sink_W":         sum(hs["Q_loss_W"] for hs in best_per_hs.values()),
        "P_net_W":          max(P_opt * dose_sf -
                                sum(hs["Q_loss_W"] for hs in best_per_hs.values()), 0.5),
    }

    grid_data = {
        "P_vals": P_vals, "t_vals": t_vals,
        "cost":   cost_grid,
        "under":  under_grid,
        "oar":    oar_grid,
        "energy": energy_grid,
        "diam":   diam_grid,
        "clear":  clr_grid,
    }

    return best_result, pareto, grid_data


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY B — GENETIC ALGORITHM
# ─────────────────────────────────────────────────────────────────────────────

class _Individual:
    """Chromosome = [P_w, t_s] in continuous space."""
    __slots__ = ["P", "t", "cost", "terms", "zone_diam", "per_hs", "min_clear"]

    def __init__(self, P=None, t=None):
        self.P  = P if P is not None else np.random.uniform(P_MIN_W, P_MAX_W)
        self.t  = t if t is not None else np.random.uniform(T_MIN_S, T_MAX_S)
        self.cost      = np.inf
        self.terms     = {}
        self.zone_diam = 0.0
        self.per_hs    = {}
        self.min_clear = 0.0

    def clip(self):
        self.P = float(np.clip(self.P, P_MIN_W, P_MAX_W))
        self.t = float(np.clip(self.t, T_MIN_S, T_MAX_S))

    def evaluate(self, tumor_diam_cm, centroid_dists, vnames,
                 tissue_props, margin_cm, dose_sf):
        c, terms, diam, per_hs, min_cl = compute_cost(
            self.P, self.t, tumor_diam_cm, centroid_dists, vnames,
            tissue_props, margin_cm, dose_sf)
        self.cost      = c
        self.terms     = terms
        self.zone_diam = diam
        self.per_hs    = per_hs
        self.min_clear = min_cl

def _tournament(population, k=GA_TOURNEY_K):
    """Tournament selection — returns best individual from k random contestants."""
    contestants = np.random.choice(population, k, replace=False)
    return min(contestants, key=lambda ind: ind.cost)

def _blx_crossover(p1, p2, alpha=GA_BLX_ALPHA):
    """BLX-α crossover: offspring drawn from extended interval [min-α·d, max+α·d]."""
    def _blx(a, b):
        lo, hi = min(a, b), max(a, b)
        d  = hi - lo
        return np.random.uniform(lo - alpha * d, hi + alpha * d)
    c1 = _Individual(P=_blx(p1.P, p2.P), t=_blx(p1.t, p2.t))
    c2 = _Individual(P=_blx(p1.P, p2.P), t=_blx(p1.t, p2.t))
    c1.clip(); c2.clip()
    return c1, c2

def _mutate(ind, sigma_P, sigma_t):
    """Gaussian mutation."""
    ind.P += np.random.normal(0, sigma_P)
    ind.t += np.random.normal(0, sigma_t)
    ind.clip()

def genetic_algorithm_optimizer(tumor_diam_cm, centroid_dists, vnames,
                                 tissue_props, margin_cm=0.5, dose_sf=1.0,
                                 verbose=True):
    """
    Genetic Algorithm optimizer for (P, t) regime selection.

    Algorithm:
        1. Initialise random population (GA_POP_SIZE individuals)
        2. Evaluate fitness = cost function J(P, t)
        3. Repeat for GA_GENERATIONS:
           a. Preserve top GA_ELITE_N (elitism)
           b. Fill rest via tournament selection + BLX-α crossover + Gaussian mutation
           c. Evaluate new individuals
           d. Track convergence (best cost over generations)
        4. Return best individual

    Returns
    -------
    best_result   : dict matching grid_search_optimizer output format
    history       : list of best cost per generation (convergence curve)
    final_pop     : final population for diversity analysis
    """
    np.random.seed(GA_SEED)
    sigma_P = GA_SIGMA_FRAC * (P_MAX_W - P_MIN_W)
    sigma_t = GA_SIGMA_FRAC * (T_MAX_S - T_MIN_S)

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  GENETIC ALGORITHM OPTIMIZER")
        print(f"  Population={GA_POP_SIZE}  Generations={GA_GENERATIONS}")
        print(f"  Selection: Tournament (k={GA_TOURNEY_K})")
        print(f"  Crossover: BLX-α (α={GA_BLX_ALPHA})")
        print(f"  Mutation:  Gaussian (σ_P={sigma_P:.1f}W, σ_t={sigma_t:.1f}s)")
        print(f"  Elitism:   top {GA_ELITE_N} preserved each generation")
        print(f"{'─'*60}")

    # ── Initialise population
    pop = [_Individual() for _ in range(GA_POP_SIZE)]
    for ind in pop:
        ind.evaluate(tumor_diam_cm, centroid_dists, vnames,
                     tissue_props, margin_cm, dose_sf)

    history = []
    best_ind = min(pop, key=lambda x: x.cost)

    if verbose:
        print(f"  {'Gen':>5}  {'Best Cost':>12}  {'P_opt(W)':>10}  "
              f"{'t_opt(s)':>10}  {'Zone(cm)':>9}  {'OAR_risk':>9}")
        print(f"  {'─'*65}")

    for gen in range(1, GA_GENERATIONS + 1):
        # ── Elitism: preserve top individuals
        pop_sorted = sorted(pop, key=lambda x: x.cost)
        new_pop    = pop_sorted[:GA_ELITE_N]

        # ── Fill rest with crossover + mutation
        while len(new_pop) < GA_POP_SIZE:
            p1 = _tournament(pop)
            p2 = _tournament(pop)
            c1, c2 = _blx_crossover(p1, p2)
            _mutate(c1, sigma_P, sigma_t)
            _mutate(c2, sigma_P, sigma_t)
            for child in (c1, c2):
                child.evaluate(tumor_diam_cm, centroid_dists, vnames,
                               tissue_props, margin_cm, dose_sf)
                new_pop.append(child)

        pop = new_pop[:GA_POP_SIZE]
        best_ind = min(pop, key=lambda x: x.cost)
        history.append(best_ind.cost)

        if verbose and (gen % 10 == 0 or gen == 1 or gen == GA_GENERATIONS):
            print(f"  {gen:>5}  {best_ind.cost:>12.5f}  "
                  f"{best_ind.P:>10.2f}  {best_ind.t:>10.1f}  "
                  f"{best_ind.zone_diam:>9.3f}  "
                  f"{best_ind.terms.get('oar_risk', 0):>9.4f}")

    # ── Build clearance report
    zone_r = (best_ind.zone_diam / 2.0) / 100.0
    clearance_report = [
        {"vessel": vn,
         "wall_clear_mm": (centroid_dists[vn] - VESSEL_RADII[vn] - zone_r) * 1000}
        for vn in vnames
    ]
    constrained = best_ind.min_clear < OAR_MIN_CLEAR_M

    Q_sink = sum(hs["Q_loss_W"] for hs in best_ind.per_hs.values())

    if verbose:
        print(f"\n  GA Optimal:  P = {best_ind.P:.2f} W   t = {best_ind.t:.1f} s")
        print(f"  Zone diam  : {best_ind.zone_diam:.2f} cm")
        print(f"  Min clear  : {best_ind.min_clear*1000:.1f} mm")
        print(f"  Final cost : {best_ind.cost:.5f}")
        print(f"  Convergence: final = {history[-1]:.5f}  "
              f"initial = {history[0]:.5f}  "
              f"improvement = {(history[0]-history[-1])/history[0]*100:.1f}%")
        print(f"{'─'*60}")

    best_result = {
        "method":           "GeneticAlgorithm",
        "P_opt":            best_ind.P,
        "t_opt":            best_ind.t,
        "zone_diam_cm":     best_ind.zone_diam,
        "zone_fwd_cm":      best_ind.zone_diam * 1.25,
        "min_clear_mm":     best_ind.min_clear * 1000,
        "cost":             best_ind.cost,
        "cost_terms":       best_ind.terms,
        "per_vessel_hs":    best_ind.per_hs,
        "clearance_report": clearance_report,
        "constrained":      constrained,
        "converged":        True,
        "margin_cm":        margin_cm,
        "dose_sf":          dose_sf,
        "Q_sink_W":         Q_sink,
        "P_net_W":          max(best_ind.P * dose_sf - Q_sink, 0.5),
    }

    return best_result, history, pop


# ─────────────────────────────────────────────────────────────────────────────
# PARETO FRONT EXTRACTION  (for grid search results)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_pareto(solutions, obj1="undercoverage", obj2="oar_risk"):
    """
    Extract Pareto-optimal solutions for two objectives (minimise both).
    A solution is non-dominated if no other solution is strictly better in both objectives.
    """
    pareto = []
    for s in solutions:
        dominated = False
        for other in solutions:
            if other is s:
                continue
            if (other[obj1] <= s[obj1] and other[obj2] <= s[obj2] and
               (other[obj1]  < s[obj1] or other[obj2]  < s[obj2])):
                dominated = True
                break
        if not dominated:
            pareto.append(s)
    # Sort by obj1
    pareto.sort(key=lambda s: s[obj1])
    return pareto


# ─────────────────────────────────────────────────────────────────────────────
# ASI COMPUTATION  (compatible with v9 / v11 format)
# ─────────────────────────────────────────────────────────────────────────────

def compute_asi_moo(opt_result, tumor_diam_cm, ray_losses=None):
    """
    Compute ASI score for an MOO optimizer result.
    Uses the same 4-component formula as v9 (no DAS — omnidirectional).

    Parameters
    ----------
    opt_result   : dict from grid_search or GA optimizer
    tumor_diam_cm: float
    ray_losses   : list of ray loss percentages (optional, for DRA)
    """
    per_hs       = opt_result["per_vessel_hs"]
    cr           = opt_result["clearance_report"]
    zone_diam_cm = opt_result["zone_diam_cm"]
    constrained  = opt_result["constrained"]

    # HSS
    max_loss  = max(hs["loss_pct"] for hs in per_hs.values())
    hss_score = float(np.clip(100.0 * (1.0 - max_loss / 50.0), 0, 100))

    # OCM
    min_cl_mm = min(c["wall_clear_mm"] for c in cr) if cr else 20.0
    ocm_score = float(np.clip(100.0 * min_cl_mm / 20.0, 0, 100))

    # CC
    margin_mm = (zone_diam_cm - tumor_diam_cm) * 10.0
    cc_score  = float(np.clip(100.0 * margin_mm / 10.0, 0, 100))
    if constrained:
        cc_score *= 0.60

    # DRA
    if ray_losses and len(ray_losses) > 1:
        spread    = float(np.max(ray_losses) - np.min(ray_losses))
        dra_score = float(np.clip(100.0 * (1.0 - spread / 30.0), 0, 100))
    else:
        dra_score = 50.0

    # Weighted composite (v9 weights — no DAS)
    w = {"hss": 0.35, "ocm": 0.30, "cc": 0.20, "dra": 0.15}
    asi = (w["hss"] * hss_score + w["ocm"] * ocm_score +
           w["cc"]  * cc_score  + w["dra"] * dra_score)

    risk = ("LOW"      if asi >= 75 else
            "MODERATE" if asi >= 50 else
            "HIGH"     if asi >= 30 else "CRITICAL")

    interp = {
        "LOW":      "MOO optimizer achieved coverage + OAR compliance with minimal energy.",
        "MODERATE": "Vessel proximity noted; optimizer found a locally safe compromise.",
        "HIGH":     "Heat sink significantly constrained feasible solution space.",
        "CRITICAL": "No feasible unconstrained solution found — constrained optimum returned.",
    }[risk]

    return {
        "asi":            round(asi, 1),
        "hss_score":      round(hss_score, 1),
        "ocm_score":      round(ocm_score, 1),
        "cc_score":       round(cc_score, 1),
        "dra_score":      round(dra_score, 1),
        "risk_label":     risk,
        "max_loss_pct":   round(max_loss, 2),
        "min_clear_mm":   round(min_cl_mm, 1),
        "margin_mm":      round(margin_mm, 1),
        "spread_pct":     round(float(np.max(ray_losses) - np.min(ray_losses))
                                if ray_losses and len(ray_losses) > 1 else 0.0, 2),
        "interpretation": interp,
        "method":         opt_result["method"],
        "cost":           opt_result["cost"],
        "cost_terms":     opt_result["cost_terms"],
    }

def print_asi_moo(asi):
    bar_len = 40
    filled  = int(round(asi["asi"] / 100.0 * bar_len))
    sym     = {"LOW":"🟢","MODERATE":"🟡","HIGH":"🟠","CRITICAL":"🔴"}[asi["risk_label"]]
    bar     = sym * filled + "⬜" * (bar_len - filled)
    print("\n" + "═"*70)
    print(f"  ASI — Method 3 [{asi['method']}]")
    print("═"*70)
    print(f"  Overall ASI : {asi['asi']:>5.1f} / 100   [{asi['risk_label']}]")
    print(f"  {bar}")
    print(f"\n  Sub-scores:")
    print(f"  {'Heat Sink Severity':<32} HSS = {asi['hss_score']:>5.1f}  "
          f"[max loss {asi['max_loss_pct']:.2f}%]")
    print(f"  {'OAR Clearance Margin':<32} OCM = {asi['ocm_score']:>5.1f}  "
          f"[min wall {asi['min_clear_mm']:.1f} mm]")
    print(f"  {'Coverage Confidence':<32}  CC = {asi['cc_score']:>5.1f}  "
          f"[margin {asi['margin_mm']:.1f} mm]")
    print(f"  {'Directional Risk Asymmetry':<32} DRA = {asi['dra_score']:>5.1f}  "
          f"[spread {asi['spread_pct']:.2f}%]")
    print(f"\n  MOO Objective breakdown:")
    ct = asi["cost_terms"]
    print(f"    Undercoverage term  : {ct.get('undercoverage', 0):.5f}  "
          f"(w={OBJ_WEIGHTS['undercoverage']:.2f})")
    print(f"    OAR risk term       : {ct.get('oar_risk', 0):.5f}  "
          f"(w={OBJ_WEIGHTS['oar_risk']:.2f})")
    print(f"    Energy term         : {ct.get('energy', 0):.5f}  "
          f"(w={OBJ_WEIGHTS['energy']:.2f})")
    print(f"    Total cost J        : {asi['cost']:.5f}")
    print(f"\n  ▶  {asi['interpretation']}")
    print("═"*70)


# ─────────────────────────────────────────────────────────────────────────────
# TEXT-BASED COST MAP VISUALISER  (no pyvista dependency)
# ─────────────────────────────────────────────────────────────────────────────

def print_cost_heatmap(grid_data, n_rows=15, n_cols=30):
    """
    Print a terminal ASCII heatmap of the cost surface.
    Rows = time (T_MIN → T_MAX), Cols = power (P_MIN → P_MAX).
    """
    cost = grid_data["cost"]
    P_v  = grid_data["P_vals"]
    t_v  = grid_data["t_vals"]

    # Downsample to n_rows × n_cols
    r_idx = np.linspace(0, len(t_v) - 1, n_rows, dtype=int)
    c_idx = np.linspace(0, len(P_v) - 1, n_cols, dtype=int)
    sub   = cost[np.ix_(c_idx, r_idx)]   # [P-idx, t-idx]

    c_min, c_max = sub.min(), sub.max()
    ramp = " ░▒▓█"

    print("\n  COST SURFACE  (rows=Power, cols=Time, darker=lower cost)")
    print(f"  P: {P_v[0]:.0f}→{P_v[-1]:.0f} W  |  t: {t_v[0]:.0f}→{t_v[-1]:.0f} s")
    print("  " + "─" * (n_cols + 4))
    for i, pi in enumerate(c_idx):
        row = ""
        for j, tj in enumerate(r_idx):
            v  = cost[pi, tj]
            nv = (v - c_min) / max(c_max - c_min, 1e-9)
            ch = ramp[int(nv * (len(ramp) - 1))]
            row += ch
        print(f"  {P_v[pi]:>5.0f}W |{row}|")
    print("  " + "─" * (n_cols + 4))
    print(f"  {'':>7}  {t_v[0]:.0f}s {'':>{n_cols//2-8}}→  {t_v[-1]:.0f}s")


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON SUMMARY PRINTER
# ─────────────────────────────────────────────────────────────────────────────

def print_method3_comparison(gs_result, gs_asi, ga_result, ga_asi):
    """Print a side-by-side comparison of grid search vs GA results."""
    print("\n" + "═"*70)
    print("  METHOD 3 — COMPARISON: Grid Search vs Genetic Algorithm")
    print("═"*70)
    print(f"  {'Metric':<30}  {'Grid Search':>15}  {'Genetic Alg.':>15}")
    print("  " + "─"*60)
    print(f"  {'Power (W)':<30}  {gs_result['P_opt']:>15.1f}  {ga_result['P_opt']:>15.2f}")
    print(f"  {'Time (s)':<30}  {gs_result['t_opt']:>15.0f}  {ga_result['t_opt']:>15.1f}")
    print(f"  {'Zone diameter (cm)':<30}  {gs_result['zone_diam_cm']:>15.2f}  {ga_result['zone_diam_cm']:>15.2f}")
    print(f"  {'Min wall clearance (mm)':<30}  {gs_result['min_clear_mm']:>15.1f}  {ga_result['min_clear_mm']:>15.1f}")
    print(f"  {'Total cost J':<30}  {gs_result['cost']:>15.5f}  {ga_result['cost']:>15.5f}")
    print(f"  {'ASI score':<30}  {gs_asi['asi']:>15.1f}  {ga_asi['asi']:>15.1f}")
    print(f"  {'ASI risk':<30}  {gs_asi['risk_label']:>15}  {ga_asi['risk_label']:>15}")
    print(f"  {'HSS':<30}  {gs_asi['hss_score']:>15.1f}  {ga_asi['hss_score']:>15.1f}")
    print(f"  {'OCM':<30}  {gs_asi['ocm_score']:>15.1f}  {ga_asi['ocm_score']:>15.1f}")
    print(f"  {'CC':<30}  {gs_asi['cc_score']:>15.1f}  {ga_asi['cc_score']:>15.1f}")
    print(f"  {'Constrained':<30}  {'YES' if gs_result['constrained'] else 'NO':>15}  "
          f"{'YES' if ga_result['constrained'] else 'NO':>15}")
    print("═"*70)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API  — call this from your main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_method3(tumor_diam_cm, centroid_dists, vnames,
                tissue_props=None, margin_cm=0.5, dose_sf=1.0,
                ray_losses=None, verbose=True):
    """
    Run Method 3: Multi-Objective Optimization.

    Parameters
    ----------
    tumor_diam_cm  : float — measured tumor diameter in cm
    centroid_dists : dict  — {vessel_name: distance_m} from tumor_metrics()
    vnames         : list  — vessel names present (subset of VESSEL_NAMES)
    tissue_props   : dict  — from TUMOR_TYPES[type_key] in hs1_directional_mwa.py
                             (optional; uses HCC defaults if None)
    margin_cm      : float — ablation margin (default 0.5 cm)
    dose_sf        : float — dose scale factor = k_factor × dose_factor
                             from histology + consistency (default 1.0)
    ray_losses     : list  — ray heat-loss percentages for DRA computation
    verbose        : bool  — print progress

    Returns
    -------
    gs_result, gs_asi, gs_pareto, gs_grid  — Grid Search outputs
    ga_result, ga_asi, ga_history           — GA outputs
    """
    if tissue_props is None:
        tissue_props = {
            "k_tissue": K_TISSUE_DEFAULT,
            "rho_cp":   RHO_CP_DEFAULT,
            "omega_b":  OMEGA_B_DEFAULT,
        }

    print("\n" + "╔" + "═"*66 + "╗")
    print("║  METHOD 3 — MULTI-OBJECTIVE OPTIMIZATION REGIME SELECTOR       ║")
    print("║  Strategy A: Grid Search  |  Strategy B: Genetic Algorithm     ║")
    print("╚" + "═"*66 + "╝")

    print(f"\n  Input:  Tumor diam = {tumor_diam_cm:.2f} cm  "
          f"Margin = {margin_cm} cm  Dose SF = {dose_sf:.3f}")
    print(f"  Vessels: {vnames}")

    # ── Grid Search
    gs_result, gs_pareto, gs_grid = grid_search_optimizer(
        tumor_diam_cm, centroid_dists, vnames, tissue_props,
        margin_cm, dose_sf, verbose=verbose)
    gs_asi = compute_asi_moo(gs_result, tumor_diam_cm, ray_losses)

    print_cost_heatmap(gs_grid)

    # ── Genetic Algorithm
    ga_result, ga_history, _ = genetic_algorithm_optimizer(
        tumor_diam_cm, centroid_dists, vnames, tissue_props,
        margin_cm, dose_sf, verbose=verbose)
    ga_asi = compute_asi_moo(ga_result, tumor_diam_cm, ray_losses)

    # ── Print ASI breakdowns
    print_asi_moo(gs_asi)
    print_asi_moo(ga_asi)

    # ── Side-by-side comparison
    print_method3_comparison(gs_result, gs_asi, ga_result, ga_asi)

    return (gs_result, gs_asi, gs_pareto, gs_grid,
            ga_result, ga_asi, ga_history)


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE DEMO  (run without VTK meshes for testing)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  Running Method 3 standalone demo (synthetic inputs)...\n")

    # Synthetic tumor and vessel geometry (replace with real tumor_metrics output)
    demo_tumor_diam = 3.5     # cm
    demo_dists = {
        "portal_vein":   0.018,   # 18 mm
        "hepatic_vein":  0.025,   # 25 mm
        "aorta":         0.060,   # 60 mm
        "ivc":           0.045,   # 45 mm
        "hepatic_artery":0.030,   # 30 mm
    }
    demo_vnames = list(demo_dists.keys())

    # HCC, firm tissue (dose_sf = 1.00 × 1.00 = 1.00)
    demo_tissue = {
        "k_tissue": 0.52,
        "rho_cp":   3.6e6,
        "omega_b":  0.0064,
    }
    demo_dose_sf = 1.00

    # Dummy ray losses (normally from ray tracing in heatsink_tumorselect.py)
    np.random.seed(7)
    demo_ray_losses = np.random.uniform(0.5, 8.0, 200).tolist()

    results = run_method3(
        tumor_diam_cm  = demo_tumor_diam,
        centroid_dists = demo_dists,
        vnames         = demo_vnames,
        tissue_props   = demo_tissue,
        margin_cm      = 0.5,
        dose_sf        = demo_dose_sf,
        ray_losses     = demo_ray_losses,
        verbose        = True,
    )

    gs_result, gs_asi = results[0], results[1]
    ga_result, ga_asi = results[4], results[5]

    print(f"\n  ✔  Method 3 demo complete.")
    print(f"  Grid Search:  {gs_result['P_opt']:.1f}W × {gs_result['t_opt']:.0f}s  "
          f"→ zone {gs_result['zone_diam_cm']:.2f}cm  ASI={gs_asi['asi']:.1f}")
    print(f"  GA:           {ga_result['P_opt']:.1f}W × {ga_result['t_opt']:.1f}s  "
          f"→ zone {ga_result['zone_diam_cm']:.2f}cm  ASI={ga_asi['asi']:.1f}")
