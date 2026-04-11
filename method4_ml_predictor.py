#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   METHOD 4 — ML-BASED REGIME PREDICTOR                                     ║
║   For Microwave Ablation Planning Comparison Study                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Author  : Veda Nunna (algorithm design)                                    ║
║  Version : 1.0                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  SCIENTIFIC BASIS                                                            ║
║  ──────────────                                                              ║
║  Trains a supervised regression model to directly predict the optimal       ║
║  (Power, Time) regime from patient and lesion features:                     ║
║                                                                              ║
║  Feature vector (9 inputs):                                                 ║
║    1.  tumor_diam_cm          — target ablation dimension                   ║
║    2.  min_vessel_dist_mm     — closest vessel (primary heat-sink)          ║
║    3.  max_vessel_dist_mm     — furthest vessel                             ║
║    4.  mean_vessel_dist_mm    — average vessel proximity                    ║
║    5.  k_factor               — histology-based energy scaling              ║
║    6.  dose_factor            — consistency-based dose correction           ║
║    7.  depth_cm               — needle insertion depth                      ║
║    8.  max_blood_velocity     — dominant flow speed (heat-sink proxy)       ║
║    9.  has_large_vessel       — binary: aorta/IVC within 30 mm             ║
║                                                                              ║
║  Targets (2 outputs):                                                       ║
║    • P_opt (W)                                                              ║
║    • t_opt (s)                                                               ║
║                                                                              ║
║  Training data:                                                              ║
║    Generated via physics simulation (Pennes bioheat + Gnielinski HS)       ║
║    using a Latin Hypercube Sampling (LHS) design over the feature space.   ║
║    N_TRAIN = 800 synthetic cases. Synthetic labels computed by the same     ║
║    physics engine as Method 3 (ensures internal consistency).               ║
║                                                                              ║
║  Three models compared:                                                      ║
║    A) Random Forest (scikit-learn) — robust ensemble, interpretable         ║
║       n_estimators=200, max_depth=12, min_samples_leaf=3                   ║
║       Ref: Breiman (2001) — handles non-linearity and interaction effects  ║
║                                                                              ║
║    B) XGBoost — gradient boosted trees, strong tabular baseline             ║
║       n_estimators=300, learning_rate=0.05, max_depth=6                    ║
║       Ref: Chen & Guestrin (2016) XGBoost                                  ║
║                                                                              ║
║    C) Gradient Boosted Trees (sklearn GBM) — if XGBoost unavailable        ║
║       Fallback when xgboost package is not installed.                       ║
║                                                                              ║
║  At inference:                                                               ║
║    1. Build feature vector from patient data                                 ║
║    2. Predict (P, t) via each model                                          ║
║    3. Clip to physical bounds [P_MIN, P_MAX] × [T_MIN, T_MAX]              ║
║    4. Run physics verification (heat-sink + zone radius)                    ║
║    5. Compute ASI score                                                      ║
║    6. Report predicted vs physics-verified regime                            ║
║                                                                              ║
║  OUTPUTS (same structure as Methods 1-3)                                    ║
║    • regime dict per model (RF, XGB, GBM)                                  ║
║    • per-vessel heat-sink (physics-verified)                                ║
║    • clearance report                                                        ║
║    • ASI score                                                               ║
║    • feature importance scores                                               ║
║    • training metrics (R², MAE)                                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL IMPORTS  — graceful fallback if not installed
# ─────────────────────────────────────────────────────────────────────────────
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_absolute_error, r2_score
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("  ⚠  scikit-learn not found — install with: pip install scikit-learn")

try:
    from xgboost import XGBRegressor
    XGBOOST_OK = True
except ImportError:
    XGBOOST_OK = False
    # Will use GBM fallback


# ─────────────────────────────────────────────────────────────────────────────
# SHARED PHYSICAL CONSTANTS  (keep in sync with Methods 1-3)
# ─────────────────────────────────────────────────────────────────────────────

RHO_B   = 1060.0;  MU_B    = 3.5e-3;  C_B     = 3700.0
K_B     = 0.52;    T_BLOOD = 37.0;    T_TISS  = 90.0;  T_ABL = 60.0
ALPHA_TISSUE = 70.0;  L_SEG = 0.01;  OAR_MIN_CLEAR_M = 5e-3

VESSEL_DIAMETERS = {
    "portal_vein":   12e-3, "hepatic_vein":   8e-3,
    "aorta":         25e-3, "ivc":           20e-3, "hepatic_artery": 4.5e-3,
}
VESSEL_VELOCITIES = {
    "portal_vein": 0.15, "hepatic_vein": 0.20, "aorta": 0.40,
    "ivc": 0.35, "hepatic_artery": 0.30,
}
VESSEL_RADII = {vn: d/2.0 for vn, d in VESSEL_DIAMETERS.items()}

P_MIN_W = 20.0;  P_MAX_W = 200.0;  T_MIN_S = 60.0;  T_MAX_S = 900.0

# Feature names (for importance plots)
FEATURE_NAMES = [
    "tumor_diam_cm",
    "min_vessel_dist_mm",
    "max_vessel_dist_mm",
    "mean_vessel_dist_mm",
    "k_factor",
    "dose_factor",
    "depth_cm",
    "max_blood_velocity",
    "has_large_vessel",
]
TARGET_NAMES = ["P_opt_W", "t_opt_s"]

# ─────────────────────────────────────────────────────────────────────────────
# PHYSICS ENGINE  (same as Method 3 — used for label generation and verification)
# ─────────────────────────────────────────────────────────────────────────────

def _nusselt(Re, Pr):
    if Re < 2300: return 4.36
    f  = (0.790 * np.log(Re) - 1.64) ** (-2)
    Nu = (f/8)*(Re-1000)*Pr / (1.0+12.7*np.sqrt(f/8)*(Pr**(2/3)-1))
    if Re >= 10000: Nu = 0.023*Re**0.8*Pr**0.4
    return max(Nu, 4.36)

def _wall_corr(Re, D):
    if Re < 2300: return 1.0
    f = (0.790*np.log(Re)-1.64)**(-2)
    nu = MU_B/RHO_B; u_tau = 0.25*np.sqrt(f/8)
    dv = 5.0*nu/(u_tau+1e-9); Pr = (C_B*MU_B)/K_B
    return max(0.90, 1.0 - dv*Pr**(-1/3)/(D/2.0))

def vessel_hs(d_m, vn, P, t):
    D=VESSEL_DIAMETERS[vn]; u=VESSEL_VELOCITIES[vn]
    Re=(RHO_B*u*D)/MU_B; Pr=(C_B*MU_B)/K_B
    Nu=_nusselt(Re,Pr); eta=_wall_corr(Re,D)
    hb=(Nu*K_B)/D; hw=hb*eta
    Ac=(D/2)*(np.pi/3)*L_SEG; Af=np.pi*D*L_SEG
    dTw=max(T_TISS-T_BLOOD,0.1); dTb=max((T_TISS+T_BLOOD)/2-T_BLOOD,0.1)
    Qw=hw*Ac*dTw; Qb=(0.30 if Re>=2300 else 0.05)*hb*Af*dTb
    Qv=min(Qw+Qb,P); d=max(d_m,1e-4)
    Ql=min(Qv*np.exp(-ALPHA_TISSUE*d),P)
    Ei=P*t; El=min(Ql*t,Ei)
    return {"vessel":vn,"dist_mm":d*1000,"Re":Re,"Pr":Pr,"Nu":Nu,
            "flow_regime":("Laminar" if Re<2300 else "Turbulent"),
            "Q_loss_W":Ql,"E_loss_J":El,
            "loss_pct":100.0*El/max(Ei,1e-9)}

def total_hs(dists, vnames, P, t):
    total=0.0; per={}
    for vn in vnames:
        hs=vessel_hs(dists[vn],vn,P,t)
        per[vn]=hs; total+=hs["Q_loss_W"]
    return min(total,P*0.85), per

def pennes_radius(Pnet, t, tissue):
    kt=tissue.get("k_tissue",0.52); rcp=tissue.get("rho_cp",3.6e6)
    omega=tissue.get("omega_b",0.0064)
    gam=np.sqrt(omega*RHO_B*C_B/kt); tau=rcp/max(omega*RHO_B*C_B,1e-6)
    eff=1.0-np.exp(-t/max(tau,1e-3))
    Peff=max(Pnet*eff,0.1); den=4*np.pi*kt*(T_ABL-T_BLOOD)*max(gam,1e-3)
    return float(np.clip(np.sqrt(max(Peff/den,1e-6)),0.005,0.080))

def physics_optimal_regime(tumor_diam_cm, dists, vnames, tissue,
                             margin_cm=0.5, dose_sf=1.0):
    """
    Solve for (P, t) that achieves required coverage with minimum energy.
    Grid search at coarse resolution (10×10) for label generation.
    Used to build the ML training set.
    """
    r_req = ((tumor_diam_cm + margin_cm)/2.0)/100.0
    P_vals= np.linspace(P_MIN_W, P_MAX_W, 10)
    t_vals= np.linspace(T_MIN_S, T_MAX_S, 10)
    best_P, best_t, best_cost = P_MAX_W, T_MAX_S, np.inf

    for P in P_vals:
        for t in t_vals:
            Pe = P * dose_sf
            Qs, _ = total_hs(dists, vnames, Pe, t)
            Pnet  = max(Pe - Qs, 0.5)
            r_abl = pennes_radius(Pnet, t, tissue)
            zone_r= r_abl
            clrs  = [dists[vn]-VESSEL_RADII[vn]-zone_r for vn in vnames]
            min_cl= min(clrs)
            under = max(r_req-r_abl, 0.0) / max(r_req, 1e-3)
            oar   = float(np.clip(1.0 - min_cl/OAR_MIN_CLEAR_M, 0.0, 1.0))
            energy= (P*t)/(P_MAX_W*T_MAX_S)
            cost  = 0.50*under + 0.35*oar + 0.15*energy
            if cost < best_cost:
                best_cost=cost; best_P=P; best_t=t
    return best_P, best_t


# ─────────────────────────────────────────────────────────────────────────────
# LATIN HYPERCUBE SAMPLING  (for training data generation)
# ─────────────────────────────────────────────────────────────────────────────

def _lhs(n, d, seed=0):
    """
    Simple Latin Hypercube Sampling.
    Returns (n, d) array with values in [0, 1].
    Each column has exactly one sample per stratum.
    """
    rng = np.random.default_rng(seed)
    result = np.zeros((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        result[:, j] = (perm + rng.random(n)) / n
    return result


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING DATA GENERATION
# ─────────────────────────────────────────────────────────────────────────────

N_TRAIN = 800

# Feature ranges for LHS sampling
FEATURE_RANGES = {
    "tumor_diam_cm":       (2.0,  5.5),    # cm
    "min_vessel_dist_mm":  (5.0,  80.0),   # mm
    "max_vessel_dist_mm":  (30.0, 150.0),  # mm
    "mean_vessel_dist_mm": (15.0, 100.0),  # mm
    "k_factor":            (0.90, 1.30),   # histology scaling
    "dose_factor":         (0.90, 1.20),   # consistency scaling
    "depth_cm":            (3.0,  20.0),   # cm
    "max_blood_velocity":  (0.15, 0.40),   # m/s
    "has_large_vessel":    (0.0,  1.0),    # binary (threshold at 0.5)
}

# Default tissue (HCC) for training — varied via k_factor/dose_factor
_BASE_TISSUE = {"k_tissue": 0.52, "rho_cp": 3.6e6, "omega_b": 0.0064}
_ALL_VNAMES  = list(VESSEL_DIAMETERS.keys())

def _features_to_inputs(row):
    """
    Convert a feature row to physics-simulator inputs.
    Reconstructs approximate vessel distances from summary statistics.
    """
    diam     = float(row[0])
    d_min    = float(row[1]) / 1000.0   # mm → m
    d_max    = float(row[2]) / 1000.0
    d_mean   = float(row[3]) / 1000.0
    k_fac    = float(row[4])
    dose_fac = float(row[5])
    depth    = float(row[6])
    v_max    = float(row[7])
    has_lg   = float(row[8]) >= 0.5

    # Reconstruct per-vessel distances
    # Strategy: spread 5 vessels between d_min and d_max, mean ≈ d_mean
    # Use a simple 5-point distribution: [d_min, d_min+(d_mean-d_min)*0.5,
    # d_mean, d_mean+(d_max-d_mean)*0.5, d_max]
    dists_raw = np.array([d_min,
                          d_min + (d_mean-d_min)*0.5,
                          d_mean,
                          d_mean + (d_max-d_mean)*0.5,
                          d_max])
    dists_raw = np.clip(dists_raw, 0.005, 0.200)
    dists  = {vn: float(dists_raw[i]) for i, vn in enumerate(_ALL_VNAMES)}
    tissue = {"k_tissue": 0.52 * k_fac,
              "rho_cp":   3.6e6,
              "omega_b":  0.0064}
    dose_sf = k_fac * dose_fac
    return diam, dists, _ALL_VNAMES, tissue, dose_sf

def generate_training_data(n=N_TRAIN, seed=42, verbose=True):
    """
    Generate synthetic training data via LHS + physics simulation.

    Returns
    -------
    X : (n, 9) float array — features
    y : (n, 2) float array — [P_opt, t_opt] labels
    """
    if verbose:
        print(f"\n  Generating {n} training samples via LHS + physics simulation...")

    ranges = list(FEATURE_RANGES.values())
    lhs    = _lhs(n, len(ranges), seed=seed)

    X = np.zeros((n, len(ranges)))
    for j, (lo, hi) in enumerate(ranges):
        X[:, j] = lo + lhs[:, j] * (hi - lo)

    y = np.zeros((n, 2))
    failed = 0
    for i in range(n):
        try:
            diam, dists, vnames, tissue, dose_sf = _features_to_inputs(X[i])
            P_opt, t_opt = physics_optimal_regime(
                diam, dists, vnames, tissue, margin_cm=0.5, dose_sf=dose_sf)
            y[i, 0] = P_opt
            y[i, 1] = t_opt
        except Exception:
            y[i, 0] = 80.0    # fallback
            y[i, 1] = 300.0
            failed  += 1

    if verbose:
        print(f"  ✔ Generated: {n-failed}/{n} valid samples  "
              f"({'no' if failed==0 else failed} simulation failures)")
        print(f"  P range: {y[:,0].min():.0f}–{y[:,0].max():.0f} W  "
              f"t range: {y[:,1].min():.0f}–{y[:,1].max():.0f} s")

    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITIONS AND TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def build_models():
    """
    Return dict of model objects.
    XGBoost is used if available; otherwise sklearn GBM.
    """
    if not SKLEARN_OK:
        return {}

    models = {}

    # Random Forest (Breiman 2001)
    models["RandomForest"] = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=3,
            n_jobs=-1, random_state=42))

    # XGBoost (Chen & Guestrin 2016) or GBM fallback
    if XGBOOST_OK:
        models["XGBoost"] = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=0))
    else:
        models["GradBoost"] = MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=5,
                subsample=0.8, random_state=42))

    return models

def train_models(X, y, verbose=True):
    """
    Train all models. Returns dict of fitted models and training metrics.

    80/20 split for metrics; models are retrained on full data for inference.
    """
    if not SKLEARN_OK:
        return {}, {}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 80/20 train/val split (fixed seed)
    n_tr  = int(0.80 * len(X))
    idx   = np.random.default_rng(7).permutation(len(X))
    tr_i  = idx[:n_tr]; va_i = idx[n_tr:]
    X_tr, X_va = X_scaled[tr_i], X_scaled[va_i]
    y_tr, y_va = y[tr_i],        y[va_i]

    models       = build_models()
    fitted_models = {}
    metrics       = {}

    if verbose:
        print(f"\n  Training ML models on {n_tr} samples  "
              f"(val: {len(va_i)} samples)")
        print(f"  Features: {FEATURE_NAMES}")
        print(f"  Targets : {TARGET_NAMES}")
        print(f"  {'─'*60}")
        print(f"  {'Model':<20}  {'R²(P)':>8}  {'R²(t)':>8}  "
              f"{'MAE(P,W)':>10}  {'MAE(t,s)':>10}")
        print(f"  {'─'*60}")

    for name, model in models.items():
        # Fit on training split for metrics
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_va)
        y_pred = np.clip(y_pred, [P_MIN_W, T_MIN_S], [P_MAX_W, T_MAX_S])

        r2_P   = r2_score(y_va[:, 0], y_pred[:, 0])
        r2_t   = r2_score(y_va[:, 1], y_pred[:, 1])
        mae_P  = mean_absolute_error(y_va[:, 0], y_pred[:, 0])
        mae_t  = mean_absolute_error(y_va[:, 1], y_pred[:, 1])

        metrics[name] = {"r2_P": r2_P, "r2_t": r2_t,
                         "mae_P": mae_P, "mae_t": mae_t}

        if verbose:
            print(f"  {name:<20}  {r2_P:>8.4f}  {r2_t:>8.4f}  "
                  f"{mae_P:>10.2f}  {mae_t:>10.1f}")

        # Refit on full data for inference
        model.fit(X_scaled, y)
        fitted_models[name] = model

    if verbose:
        print(f"  {'─'*60}")
        print(f"  ✔  All models trained and validated.")

    return fitted_models, metrics, scaler

def get_feature_importances(fitted_models):
    """Extract feature importances for Random Forest and GBM-type models."""
    importances = {}
    for name, model in fitted_models.items():
        try:
            # MultiOutputRegressor wraps the base estimator
            estimators = model.estimators_
            # Average importance across output targets
            imp = np.mean([est.feature_importances_
                           for est in estimators], axis=0)
            importances[name] = dict(zip(FEATURE_NAMES, imp))
        except AttributeError:
            pass   # XGBoost or others without sklearn-style importances
    return importances


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE VECTOR BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_vector(tumor_diam_cm, centroid_dists, vnames,
                          k_factor=1.0, dose_factor=1.0,
                          depth_cm=10.0):
    """
    Build the 9-element feature vector from clinical inputs.

    Parameters
    ----------
    tumor_diam_cm  : float — measured tumor diameter in cm
    centroid_dists : dict  — {vessel_name: distance_m}
    vnames         : list  — vessel names present
    k_factor       : float — histology scaling factor from TUMOR_TYPES
    dose_factor    : float — consistency scaling factor
    depth_cm       : float — insertion depth (from tumor_metrics)
    """
    dists_mm = [centroid_dists[vn] * 1000.0 for vn in vnames]
    d_min  = min(dists_mm)
    d_max  = max(dists_mm)
    d_mean = float(np.mean(dists_mm))

    # Maximum blood velocity among present vessels
    v_max = max(VESSEL_VELOCITIES.get(vn, 0.15) for vn in vnames)

    # Large vessel flag: aorta or IVC within 30 mm of centroid
    large_vessels = {"aorta", "ivc"}
    has_large = float(any(
        centroid_dists.get(vn, 999.) < 0.030
        for vn in vnames if vn in large_vessels))

    feat = np.array([
        tumor_diam_cm,
        d_min,
        d_max,
        d_mean,
        k_factor,
        dose_factor,
        depth_cm,
        v_max,
        has_large,
    ], dtype=float)
    return feat


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE + PHYSICS VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def predict_and_verify(model, model_name, scaler, feat,
                        tumor_diam_cm, centroid_dists, vnames,
                        tissue_props, margin_cm=0.5, dose_sf=1.0,
                        verbose=True):
    """
    Run inference, clip predictions, then verify with physics engine.

    Returns a result dict compatible with Methods 1-3 ASI computation.
    """
    # ── ML prediction
    feat_scaled = scaler.transform(feat.reshape(1, -1))
    pred        = model.predict(feat_scaled)[0]
    P_pred      = float(np.clip(pred[0], P_MIN_W, P_MAX_W))
    t_pred      = float(np.clip(pred[1], T_MIN_S, T_MAX_S))

    # ── Physics verification
    Pe          = P_pred * dose_sf
    Q_sink, per_hs = total_hs(centroid_dists, vnames, Pe, t_pred)
    P_net       = max(Pe - Q_sink, 0.5)
    r_abl       = pennes_radius(P_net, t_pred, tissue_props)
    zone_diam   = r_abl * 2.0 * 100.0   # cm
    target_req  = tumor_diam_cm + margin_cm

    # Clearance
    zone_r = r_abl
    clrs   = {vn: centroid_dists[vn] - VESSEL_RADII[vn] - zone_r
              for vn in vnames}
    min_cl = min(clrs.values())

    clearance_report = [
        {"vessel": vn, "wall_clear_mm": v * 1000}
        for vn, v in clrs.items()
    ]
    constrained = min_cl < OAR_MIN_CLEAR_M

    if verbose:
        print(f"\n  [{model_name}] Prediction:  P = {P_pred:.1f} W   t = {t_pred:.0f} s")
        print(f"  [{model_name}] Physics verify: zone = {zone_diam:.2f} cm  "
              f"(need {target_req:.2f} cm)  min_clear = {min_cl*1000:.1f} mm")
        if zone_diam < target_req:
            short = target_req - zone_diam
            print(f"  [{model_name}] ⚠  Under-coverage by {short:.2f} cm — "
                  f"prediction bias or training gap")
        if constrained:
            print(f"  [{model_name}] ⚠  OAR clearance insufficient ({min_cl*1000:.1f} mm < 5 mm)")

    return {
        "method":           f"ML_{model_name}",
        "P_opt":            P_pred,
        "t_opt":            t_pred,
        "zone_diam_cm":     zone_diam,
        "zone_fwd_cm":      zone_diam * 1.25,
        "min_clear_mm":     min_cl * 1000,
        "per_vessel_hs":    per_hs,
        "clearance_report": clearance_report,
        "constrained":      constrained,
        "converged":        zone_diam >= target_req,
        "margin_cm":        margin_cm,
        "dose_sf":          dose_sf,
        "Q_sink_W":         Q_sink,
        "P_net_W":          P_net,
        "feature_vector":   feat,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ASI COMPUTATION  (same v9 formula as Method 3)
# ─────────────────────────────────────────────────────────────────────────────

def compute_asi_ml(opt_result, tumor_diam_cm, ray_losses=None):
    per_hs = opt_result["per_vessel_hs"]
    cr     = opt_result["clearance_report"]
    zone   = opt_result["zone_diam_cm"]
    const  = opt_result["constrained"]

    max_loss  = max(hs["loss_pct"] for hs in per_hs.values())
    hss_score = float(np.clip(100.0*(1.0 - max_loss/50.0), 0, 100))

    min_cl_mm = min(c["wall_clear_mm"] for c in cr) if cr else 20.0
    ocm_score = float(np.clip(100.0*min_cl_mm/20.0, 0, 100))

    margin_mm = (zone - tumor_diam_cm)*10.0
    cc_score  = float(np.clip(100.0*margin_mm/10.0, 0, 100))
    if const: cc_score *= 0.60

    if ray_losses and len(ray_losses) > 1:
        spread    = float(np.max(ray_losses) - np.min(ray_losses))
        dra_score = float(np.clip(100.0*(1.0 - spread/30.0), 0, 100))
    else:
        dra_score = 50.0

    w   = {"hss": 0.35, "ocm": 0.30, "cc": 0.20, "dra": 0.15}
    asi = w["hss"]*hss_score + w["ocm"]*ocm_score + w["cc"]*cc_score + w["dra"]*dra_score
    risk = ("LOW" if asi>=75 else "MODERATE" if asi>=50 else "HIGH" if asi>=30 else "CRITICAL")
    interp = {
        "LOW":      "ML predictor identified an efficient regime with good coverage and OAR safety.",
        "MODERATE": "ML prediction acceptable; vessel proximity noted in physics verification.",
        "HIGH":     "Heat sink effect reduced zone significantly after ML prediction.",
        "CRITICAL": "ML predicted regime insufficient; clinical override recommended.",
    }[risk]

    return {
        "asi": round(asi,1), "hss_score": round(hss_score,1),
        "ocm_score": round(ocm_score,1), "cc_score": round(cc_score,1),
        "dra_score": round(dra_score,1), "risk_label": risk,
        "max_loss_pct": round(max_loss,2), "min_clear_mm": round(min_cl_mm,1),
        "margin_mm": round(margin_mm,1),
        "spread_pct": round(float(np.max(ray_losses)-np.min(ray_losses))
                            if ray_losses and len(ray_losses)>1 else 0.0, 2),
        "interpretation": interp, "method": opt_result["method"],
    }

def print_asi_ml(asi):
    bar_len=40; filled=int(round(asi["asi"]/100.0*bar_len))
    sym={"LOW":"🟢","MODERATE":"🟡","HIGH":"🟠","CRITICAL":"🔴"}[asi["risk_label"]]
    bar=sym*filled+"⬜"*(bar_len-filled)
    print("\n"+"═"*70)
    print(f"  ASI — {asi['method']}")
    print("═"*70)
    print(f"  Overall ASI : {asi['asi']:>5.1f} / 100   [{asi['risk_label']}]")
    print(f"  {bar}")
    print(f"  HSS={asi['hss_score']:.1f}  OCM={asi['ocm_score']:.1f}  "
          f"CC={asi['cc_score']:.1f}  DRA={asi['dra_score']:.1f}")
    print(f"  ▶  {asi['interpretation']}")
    print("═"*70)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE PRINTER
# ─────────────────────────────────────────────────────────────────────────────

def print_feature_importances(importances):
    if not importances:
        return
    print("\n" + "═"*70)
    print("  FEATURE IMPORTANCES  (Method 4 — ML)")
    print("═"*70)
    for model_name, imp_dict in importances.items():
        print(f"\n  [{model_name}]")
        sorted_feats = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)
        for feat, imp in sorted_feats:
            bar = "█" * int(imp * 40)
            print(f"  {feat:<30}  {imp:.4f}  {bar}")
    print("═"*70)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING METRICS PRINTER
# ─────────────────────────────────────────────────────────────────────────────

def print_training_metrics(metrics):
    print("\n" + "═"*70)
    print("  ML MODEL TRAINING METRICS  (80/20 hold-out split)")
    print("═"*70)
    print(f"  {'Model':<22}  {'R²(P)':>8}  {'R²(t)':>8}  "
          f"{'MAE(P,W)':>10}  {'MAE(t,s)':>10}")
    print("  " + "─"*60)
    for name, m in metrics.items():
        print(f"  {name:<22}  {m['r2_P']:>8.4f}  {m['r2_t']:>8.4f}  "
              f"{m['mae_P']:>10.2f}  {m['mae_t']:>10.1f}")
    print("═"*70)


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON TABLE PRINTER
# ─────────────────────────────────────────────────────────────────────────────

def print_method4_comparison(results_per_model, asis_per_model):
    """Print cross-model comparison table."""
    names = list(results_per_model.keys())
    print("\n" + "═"*70)
    print("  METHOD 4 — COMPARISON: All ML Models")
    print("═"*70)
    print(f"  {'Metric':<30}", end="")
    for n in names:
        print(f"  {n:>13}", end="")
    print()
    print("  " + "─"*(30 + len(names)*15))

    def row(label, fn):
        print(f"  {label:<30}", end="")
        for n in names:
            print(f"  {fn(results_per_model[n], asis_per_model[n]):>13}", end="")
        print()

    row("Power (W)",              lambda r,a: f"{r['P_opt']:.1f}")
    row("Time (s)",               lambda r,a: f"{r['t_opt']:.0f}")
    row("Zone diameter (cm)",     lambda r,a: f"{r['zone_diam_cm']:.2f}")
    row("Min wall clear (mm)",    lambda r,a: f"{r['min_clear_mm']:.1f}")
    row("ASI score",              lambda r,a: f"{a['asi']:.1f}")
    row("ASI risk",               lambda r,a: a["risk_label"])
    row("HSS",                    lambda r,a: f"{a['hss_score']:.1f}")
    row("OCM",                    lambda r,a: f"{a['ocm_score']:.1f}")
    row("CC",                     lambda r,a: f"{a['cc_score']:.1f}")
    row("Constrained",            lambda r,a: "YES" if r["constrained"] else "NO")
    row("Coverage OK",            lambda r,a: "YES" if r["converged"] else "NO")
    print("═"*70)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

# Global cache — avoids retraining on repeated calls in same session
_CACHE = {"trained": False, "models": {}, "metrics": {}, "scaler": None,
          "X": None, "y": None}

def run_method4(tumor_diam_cm, centroid_dists, vnames,
                tissue_props=None, k_factor=1.0, dose_factor=1.0,
                depth_cm=10.0, margin_cm=0.5,
                ray_losses=None, retrain=False, verbose=True):
    """
    Run Method 4: ML-Based Regime Predictor.

    Parameters
    ----------
    tumor_diam_cm  : float — measured tumor diameter in cm
    centroid_dists : dict  — {vessel_name: distance_m}
    vnames         : list  — vessel names present
    tissue_props   : dict  — from TUMOR_TYPES[type_key] (optional)
    k_factor       : float — histology k_factor (from TUMOR_TYPES)
    dose_factor    : float — consistency dose_factor
    depth_cm       : float — insertion depth in cm
    margin_cm      : float — ablation margin (default 0.5 cm)
    ray_losses     : list  — for DRA sub-score (optional)
    retrain        : bool  — force retraining even if cached
    verbose        : bool  — print progress

    Returns
    -------
    results_per_model  : dict {model_name: result_dict}
    asis_per_model     : dict {model_name: asi_dict}
    fitted_models      : dict {model_name: model}
    metrics            : dict {model_name: {r2, mae}}
    importances        : dict {model_name: {feature: importance}}
    """
    if not SKLEARN_OK:
        print("  ✘ scikit-learn required for Method 4.  "
              "Install with: pip install scikit-learn")
        return {}, {}, {}, {}, {}

    if tissue_props is None:
        tissue_props = {"k_tissue": 0.52*k_factor, "rho_cp": 3.6e6, "omega_b": 0.0064}

    dose_sf = k_factor * dose_factor

    print("\n" + "╔" + "═"*66 + "╗")
    print("║  METHOD 4 — ML-BASED REGIME PREDICTOR                          ║")
    print(f"║  Models: RandomForest  {'XGBoost' if XGBOOST_OK else 'GradBoost'}                                 ║")
    print("╚" + "═"*66 + "╝")

    print(f"\n  Input:  Tumor = {tumor_diam_cm:.2f} cm  "
          f"k_factor = {k_factor:.2f}  dose_factor = {dose_factor:.2f}  "
          f"depth = {depth_cm:.1f} cm")

    # ── Training (cached)
    if not _CACHE["trained"] or retrain:
        X, y = generate_training_data(N_TRAIN, seed=42, verbose=verbose)
        fitted_models, metrics, scaler = train_models(X, y, verbose=verbose)
        _CACHE.update({"trained": True, "models": fitted_models,
                       "metrics": metrics, "scaler": scaler, "X": X, "y": y})
    else:
        if verbose:
            print(f"\n  Using cached trained models ({N_TRAIN} samples).")
        fitted_models = _CACHE["models"]
        metrics       = _CACHE["metrics"]
        scaler        = _CACHE["scaler"]

    print_training_metrics(metrics)

    # ── Feature vector
    feat = build_feature_vector(
        tumor_diam_cm, centroid_dists, vnames,
        k_factor=k_factor, dose_factor=dose_factor, depth_cm=depth_cm)

    if verbose:
        print(f"\n  Feature vector:")
        for name, val in zip(FEATURE_NAMES, feat):
            print(f"    {name:<30} = {val:.4f}")

    # ── Inference per model
    results_per_model = {}
    asis_per_model    = {}

    print(f"\n{'─'*60}")
    print(f"  ML INFERENCE + PHYSICS VERIFICATION")
    print(f"{'─'*60}")

    for model_name, model in fitted_models.items():
        res = predict_and_verify(
            model, model_name, scaler, feat,
            tumor_diam_cm, centroid_dists, vnames,
            tissue_props, margin_cm=margin_cm, dose_sf=dose_sf,
            verbose=verbose)
        asi = compute_asi_ml(res, tumor_diam_cm, ray_losses)
        print_asi_ml(asi)
        results_per_model[model_name] = res
        asis_per_model[model_name]    = asi

    # ── Feature importances
    importances = get_feature_importances(fitted_models)
    print_feature_importances(importances)

    # ── Cross-model comparison
    print_method4_comparison(results_per_model, asis_per_model)

    return results_per_model, asis_per_model, fitted_models, metrics, importances


# ─────────────────────────────────────────────────────────────────────────────
# FOUR-METHOD MASTER COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_master_comparison(method_results, method_asis):
    """
    Print the master 4-method comparison table.

    Parameters
    ----------
    method_results : dict — key = method label, value = result dict
    method_asis    : dict — key = method label, value = ASI dict
    """
    labels = list(method_results.keys())
    print("\n" + "╔" + "═"*76 + "╗")
    print("║  MASTER COMPARISON — ALL METHODS                                           ║")
    print("╚" + "═"*76 + "╝")
    hdr = f"  {'Metric':<32}"
    for lbl in labels:
        hdr += f"  {lbl[:12]:>12}"
    print(hdr)
    print("  " + "─"*(32 + len(labels)*14))

    def row(label, fn):
        line = f"  {label:<32}"
        for lbl in labels:
            try:
                val = fn(method_results[lbl], method_asis[lbl])
            except Exception:
                val = "N/A"
            line += f"  {str(val):>12}"
        print(line)

    row("Power (W)",              lambda r,a: f"{r['P_opt']:.1f}")
    row("Time (s)",               lambda r,a: f"{r['t_opt']:.0f}")
    row("Zone diam (cm)",         lambda r,a: f"{r['zone_diam_cm']:.2f}")
    row("Min wall clear (mm)",    lambda r,a: f"{r['min_clear_mm']:.1f}")
    row("Q_sink (W)",             lambda r,a: f"{r['Q_sink_W']:.2f}")
    row("ASI score (/100)",       lambda r,a: f"{a['asi']:.1f}")
    row("ASI risk",               lambda r,a: a["risk_label"])
    row("HSS",                    lambda r,a: f"{a['hss_score']:.1f}")
    row("OCM",                    lambda r,a: f"{a['ocm_score']:.1f}")
    row("CC",                     lambda r,a: f"{a['cc_score']:.1f}")
    row("DRA",                    lambda r,a: f"{a['dra_score']:.1f}")
    row("Constrained",            lambda r,a: "YES" if r["constrained"] else "NO")
    row("Coverage met",           lambda r,a: "YES" if r.get("converged") else "NO")
    print("╚" + "═"*76 + "╝")


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  Running Method 4 standalone demo (synthetic inputs)...\n")

    demo_tumor_diam = 3.5
    demo_dists = {
        "portal_vein":   0.018,
        "hepatic_vein":  0.025,
        "aorta":         0.060,
        "ivc":           0.045,
        "hepatic_artery":0.030,
    }
    demo_vnames  = list(demo_dists.keys())
    demo_k       = 1.12    # colorectal
    demo_df      = 1.00    # firm
    demo_depth   = 12.0    # cm

    np.random.seed(7)
    demo_rays = np.random.uniform(0.5, 8.0, 200).tolist()

    results, asis, models, metrics, importances = run_method4(
        tumor_diam_cm  = demo_tumor_diam,
        centroid_dists = demo_dists,
        vnames         = demo_vnames,
        k_factor       = demo_k,
        dose_factor    = demo_df,
        depth_cm       = demo_depth,
        margin_cm      = 0.5,
        ray_losses     = demo_rays,
        retrain        = False,
        verbose        = True,
    )

    print(f"\n  ✔  Method 4 demo complete.")
    for name, res in results.items():
        asi = asis[name]
        print(f"  [{name}]  {res['P_opt']:.1f}W × {res['t_opt']:.0f}s  "
              f"→ {res['zone_diam_cm']:.2f}cm  ASI={asi['asi']:.1f} [{asi['risk_label']}]")
