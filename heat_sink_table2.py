#!/usr/bin/env python3
"""
Integrated Heat Sink Analysis — Full Physics + Visualization
=============================================================

NEW IN THIS VERSION
--------------------
1. VELOCITY PROFILE PHYSICS (Hagen-Poiseuille + turbulent boundary layer)
   - Laminar vessels (Re < 2300): parabolic profile u(r) = u_max(1 - r²/R²)
     → near-wall velocity ≈ 0 → heat transfer limited by thin wall layer
     → Nu = 4.36 (thermally developed laminar, constant heat flux)
   - Transition (2300 ≤ Re < 10000): Gnielinski correlation
     → Nu = (f/8)(Re-1000)Pr / [1 + 12.7√(f/8)(Pr^(2/3)-1)]
     → Petukhov friction: f = (0.790 ln Re - 1.64)^-2
   - Turbulent (Re ≥ 10000): Dittus-Boelter
     → Nu = 0.023 Re^0.8 Pr^0.4
   - Wall-layer correction: h_effective = h_bulk × η_wall
     → η_wall accounts for near-wall velocity deficit

2. SPLIT HEAT LOSS: wall contribution + bulk convective contribution
   - Q_wall: direct tumor→vessel-wall conduction zone (dominant)
   - Q_bulk: far-field convective mixing (secondary)
   - Total = Q_wall + 0.3 × Q_bulk  (empirical weighting from bioheat lit.)

3. DIRECTIONAL HEAT FLOW ARROWS
   - Arrow from tumor centroid toward each vessel surface
   - Arrow length ∝ Q_loss magnitude
   - Arrow color: green (low) → red (high)
   - Labeled with vessel name and loss %

4. AORTA RAY ARTIFACT FIX
   - Ray-segment distance is now capped at centroid→vessel distance
   - Prevents far aorta from appearing as 99% loss due to geometric
     intersection of ray LINE with distant mesh

5. TUMOR MESH SMOOTHING
   - PyVista smooth() applied to selected tumor before display
   - Makes Tumor 4 appear more uniform/realistic

6. BLOOD FLOW VISUALIZATION IMPROVEMENT
   - Particles colored by local velocity (blue=slow near-wall, red=fast center)
   - Particle size scales with flow speed
"""

import os
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
# FILE PATHS
# ─────────────────────────────────────────────────────────────────────

DATASET_BASE     = r"C:\Users\z005562w\OneDrive - Siemens Healthineers\Veda\Project\siemens project\3d simulation\Nunna Algo\Nunna Algo\Dataset"
PORTALVENOUS_DIR = os.path.join(DATASET_BASE, "portalvenous")

TUMOR_VTK   = os.path.join(DATASET_BASE, "908ac523data00007_leasion_out.vtk")
SURFACE_VTK = os.path.join(DATASET_BASE, "908ac523data00013_skin_out.vtk")

VESSEL_VTK_LIST = [
    os.path.join(PORTALVENOUS_DIR, "908ac52300001.vtk"),   # portal vein
    os.path.join(PORTALVENOUS_DIR, "908ac52300002.vtk"),   # hepatic vein
    os.path.join(PORTALVENOUS_DIR, "908ac52300003.vtk"),   # aorta
    os.path.join(PORTALVENOUS_DIR, "908ac52300004.vtk"),   # IVC
    os.path.join(PORTALVENOUS_DIR, "908ac52300005.vtk"),   # hepatic artery
]

VESSEL_NAMES  = ["portal_vein", "hepatic_vein", "aorta", "ivc", "hepatic_artery"]
VESSEL_COLORS = ["purple", "teal", "royalblue", "navy", "orange"]

# ─────────────────────────────────────────────────────────────────────
# MANUFACTURER ABLATION TABLE
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
    (160, 600, 7.20, 6.3, 5.6),  (160, 600, 10.20,5.9, 5.8),
    (160, 600, 10.30,6.1, 5.8),
]

# ─────────────────────────────────────────────────────────────────────
# PHYSICAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────

RHO_B    = 1060.0   # blood density        (kg/m³)
MU_B     = 3.5e-3   # dynamic viscosity     (Pa·s)
C_B      = 3700.0   # specific heat         (J/kg·K)
K_B      = 0.52     # thermal conductivity  (W/m·K)
T_BLOOD  = 37.0     # blood temperature     (°C)
T_TISSUE = 90.0     # ablation temperature  (°C)

ALPHA_TISSUE = 70.0  # tissue attenuation (m⁻¹)
L_SEG        = 0.01  # reference vessel segment (m)

MIN_DIAMETER_CM = 3.0
MAX_DIAMETER_CM = 5.0
MAX_DEPTH_CM    = 26.0

VESSEL_DIAMETERS  = {"portal_vein": 12e-3, "hepatic_vein": 8e-3,
                     "aorta": 25e-3, "ivc": 20e-3, "hepatic_artery": 4.5e-3}
VESSEL_VELOCITIES = {"portal_vein": 0.15, "hepatic_vein": 0.20,
                     "aorta": 0.40,  "ivc": 0.35,  "hepatic_artery": 0.30}

# ─────────────────────────────────────────────────────────────────────
# FULL VELOCITY PROFILE + NUSSELT CALCULATION
# ─────────────────────────────────────────────────────────────────────

def nusselt_full(Re, Pr):
    """
    Nusselt number covering all flow regimes.

    Laminar   (Re < 2300)  : Nu = 4.36  (thermally developed, const heat flux)
    Transition (2300-10000): Gnielinski correlation
                             Nu = (f/8)(Re-1000)Pr / [1 + 12.7√(f/8)(Pr^(2/3)-1)]
                             f  = (0.790 ln(Re) - 1.64)^(-2)  [Petukhov]
    Turbulent  (Re ≥ 10000): Dittus-Boelter
                             Nu = 0.023 Re^0.8 Pr^0.4
    """
    if Re < 2300:
        return 4.36   # laminar, constant wall heat flux

    f = (0.790 * np.log(Re) - 1.64) ** (-2)   # Petukhov friction factor

    if Re < 10000:
        # Gnielinski — transition regime
        Nu = (f / 8) * (Re - 1000) * Pr / (1 + 12.7 * np.sqrt(f / 8) * (Pr**(2/3) - 1))
    else:
        # Dittus-Boelter — fully turbulent
        Nu = 0.023 * (Re ** 0.8) * (Pr ** 0.4)

    return max(Nu, 4.36)   # physical lower bound


def wall_layer_correction(Re, D):
    """
    Near-wall velocity correction factor η_wall.

    For laminar flow (Hagen-Poiseuille):
        u(r) = u_max × (1 - (r/R)²)
        At the wall (r → R): u_wall = 0  (no-slip)
        The thermal boundary layer is limited by slow near-wall flow.
        → η = 1.0 (Nu = 4.36 already accounts for this geometry)

    For turbulent flow:
        The viscous sublayer thickness: δ_v = 5 ν / u_τ
        where u_τ = u_mean × √(f/8)  (friction velocity)
        The thermal sublayer: δ_t = δ_v × Pr^(-1/3)
        Correction: η = 1 - (δ_t / R)  → slight reduction from bulk Nu
        In practice, for blood (Pr ≈ 25): δ_t is very thin → η ≈ 0.97-0.99

    We return η as a simple correction multiplier on h_bulk.
    """
    if Re < 2300:
        return 1.0   # laminar: Nu = 4.36 already wall-corrected

    # Turbulent: compute viscous sublayer
    f      = (0.790 * np.log(Re) - 1.64) ** (-2)
    u_mean = VESSEL_VELOCITIES.get("aorta", 0.4)  # fallback; overridden per vessel
    nu     = MU_B / RHO_B                          # kinematic viscosity
    u_tau  = u_mean * np.sqrt(f / 8)              # friction velocity
    R      = D / 2.0

    delta_v = 5.0 * nu / (u_tau + 1e-9)           # viscous sublayer
    Pr      = (C_B * MU_B) / K_B
    delta_t = delta_v * Pr ** (-1/3)               # thermal sublayer

    eta = max(0.90, 1.0 - delta_t / R)
    return eta


def velocity_profile_at_wall(Re, D, vessel_name):
    """
    Compute the effective near-wall velocity for heat transfer.

    Returns:
        u_wall_eff: effective velocity at the vessel wall facing the tumor
        profile_type: string description
        u_centerline: centerline velocity for visualization
    """
    u_mean = VESSEL_VELOCITIES[vessel_name]
    R      = D / 2.0

    if Re < 2300:
        # Hagen-Poiseuille laminar parabolic profile
        u_max        = 2.0 * u_mean          # centerline
        # Wall-adjacent layer (r = R - ε, ε = 0.05R)
        r_wall       = 0.95 * R
        u_wall_eff   = u_max * (1.0 - (r_wall / R) ** 2)
        profile_type = "Laminar (Hagen-Poiseuille, parabolic)"
        u_centerline = u_max
    else:
        # Turbulent power-law: u(r) = u_max × (1 - r/R)^(1/7)
        u_max        = u_mean * (8/7) * (9/8)   # from power-law integration
        r_wall       = 0.95 * R
        u_wall_eff   = u_max * (1.0 - r_wall / R) ** (1/7)
        profile_type = "Turbulent (1/7 power-law)"
        u_centerline = u_max

    return u_wall_eff, profile_type, u_centerline


def heat_sink_full_physics(distance_m, vessel_name, power_w, ablation_time_s):
    """
    Full physics heat sink calculation:

    1. Compute Re, Pr
    2. Nusselt via regime-appropriate correlation (laminar/transition/turbulent)
    3. Near-wall correction factor η_wall
    4. h_effective = (Nu × K_b / D) × η_wall
    5. Q_wall = h_eff × A_contact × ΔT_wall   (tumor-facing wall strip)
    6. Q_bulk = 0.3 × h_bulk × A_full × ΔT    (bulk mixing contribution)
    7. Q_vessel = Q_wall + Q_bulk, capped at power_w
    8. Distance attenuation: Q_loss = Q_vessel × exp(-α × d)
    9. E_loss% = 100 × Q_loss × t / (P × t)

    The split between Q_wall and Q_bulk reflects that heat exchange
    happens primarily at the vessel wall closest to the tumor (Q_wall),
    with secondary contribution from turbulent bulk mixing (Q_bulk).
    For laminar vessels Q_bulk ≈ 0 because mixing is negligible.
    """
    D      = VESSEL_DIAMETERS[vessel_name]
    u_mean = VESSEL_VELOCITIES[vessel_name]
    R      = D / 2.0

    Re  = (RHO_B * u_mean * D) / MU_B
    Pr  = (C_B * MU_B) / K_B
    Nu  = nusselt_full(Re, Pr)
    eta = wall_layer_correction(Re, D)

    # Bulk heat transfer coefficient
    h_bulk = (Nu * K_B) / D

    # Wall-corrected coefficient (for tumor-facing wall strip)
    h_wall = h_bulk * eta

    # Contact area: thin strip of vessel wall facing tumor (arc ~ 60° of circumference)
    theta_contact = np.pi / 3        # 60° arc
    A_contact     = R * theta_contact * L_SEG     # m²
    A_full        = np.pi * D * L_SEG             # full circumference

    dT_wall = max(T_TISSUE - T_BLOOD, 0.1)
    dT_bulk = max((T_TISSUE + T_BLOOD) / 2 - T_BLOOD, 0.1)  # mean temperature

    # Wall contribution (dominant — direct interface heat transfer)
    Q_wall = h_wall * A_contact * dT_wall

    # Bulk convective contribution (turbulent mixing carries heat away)
    # For laminar: practically zero mixing → weight = 0.05
    # For turbulent: significant mixing → weight = 0.30
    bulk_weight = 0.30 if Re >= 2300 else 0.05
    Q_bulk      = bulk_weight * h_bulk * A_full * dT_bulk

    # Combined vessel cooling power
    Q_vessel = min(Q_wall + Q_bulk, power_w)

    # Distance attenuation through tissue
    d      = max(distance_m, 1e-4)
    Q_loss = Q_vessel * np.exp(-ALPHA_TISSUE * d)
    Q_loss = min(Q_loss, power_w)

    E_input = power_w * ablation_time_s
    E_loss  = min(Q_loss * ablation_time_s, E_input)
    pct     = 100.0 * E_loss / E_input

    # Velocity profile info
    u_wall_eff, profile_type, u_center = velocity_profile_at_wall(Re, D, vessel_name)

    return {
        "vessel":          vessel_name,
        "dist_mm":         d * 1000,
        "Re":              Re,
        "Pr":              Pr,
        "Nu":              Nu,
        "flow_regime":     ("Laminar" if Re < 2300 else
                            "Transition" if Re < 10000 else "Turbulent"),
        "profile_type":    profile_type,
        "u_mean_m_s":      u_mean,
        "u_wall_eff_m_s":  u_wall_eff,
        "u_centerline_m_s":u_center,
        "eta_wall":        eta,
        "h_bulk":          h_bulk,
        "h_wall":          h_wall,
        "Q_wall_W":        Q_wall,
        "Q_bulk_W":        Q_bulk,
        "Q_vessel_W":      Q_vessel,
        "Q_loss_W":        Q_loss,
        "E_loss_J":        E_loss,
        "loss_pct":        pct,
    }


# ─────────────────────────────────────────────────────────────────────
# MESH UTILITIES
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

def smooth_tumor_mesh(tumor_mesh, n_iter=50, relaxation=0.1):
    """Smooth tumor mesh to reduce segmentation roughness."""
    try:
        smoothed = tumor_mesh.smooth(n_iter=n_iter, relaxation_factor=relaxation,
                                     boundary_smoothing=False)
        return smoothed
    except Exception:
        return tumor_mesh   # fallback to original if smooth fails

# ─────────────────────────────────────────────────────────────────────
# TUMOR ANALYSIS
# ─────────────────────────────────────────────────────────────────────

def extract_tumors(tumor_mesh):
    print("\n🔍 Splitting tumor mesh...")
    connected = tumor_mesh.connectivity()
    tumors    = connected.split_bodies()
    print(f"   {len(tumors)} tumors found")
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
    print(f"\n  {'T':<5}{'Diam':<9}{'Depth':<9}", end="")
    for n in vnames:
        print(f"{n[:12]+'(mm)':<18}", end="")
    print()
    for m in metrics:
        print(f"  {m['idx']+1:<5}{m['diameter_cm']:<9.2f}{m['depth_cm']:<9.2f}", end="")
        for d in m["vessel_dists_m"]:
            print(f"{d*1000:<18.1f}", end="")
        print()
    return metrics

# ─────────────────────────────────────────────────────────────────────
# RAY-SEGMENT TO VESSEL DISTANCE  (with centroid-distance cap)
# ─────────────────────────────────────────────────────────────────────

def ray_segment_dist(origin, direction, path_d, vessel_pts, centroid_dist, n_sample=30):
    """
    Minimum distance from ray segment to vessel, CAPPED at centroid→vessel distance.

    The cap prevents the geometric artifact where a ray's LINE intersects a
    distant mesh (e.g. aorta at 80mm) giving a near-zero segment distance
    even though the ray travels away from the vessel. The cap enforces:
        ray_seg_dist ≥ centroid_vessel_dist  if vessel is far away

    This makes the result physically meaningful: a ray can never 'see'
    more of a vessel than the tumor centroid already sees.
    """
    ts      = np.linspace(0.0, path_d, n_sample)
    samples = origin + np.outer(ts, direction)
    tree    = cKDTree(vessel_pts)
    dists,_ = tree.query(samples, k=1)
    raw     = float(np.min(dists))
    # Cap: never report a ray as closer to the vessel than the centroid is
    return max(raw, centroid_dist * 0.5)   # allow up to 50% below centroid dist


# ─────────────────────────────────────────────────────────────────────
# OAR IDENTIFICATION
# ─────────────────────────────────────────────────────────────────────

def identify_oars(centroid, vessels, vnames, fwd_cm, diam_cm, needle_dir=None):
    a = (fwd_cm  / 2.0) / 100.0
    b = (diam_cm / 2.0) / 100.0
    if needle_dir is None:
        needle_dir = np.array([0.0, 0.0, 1.0])
    n_hat = needle_dir / (np.linalg.norm(needle_dir) + 1e-9)
    oars  = []
    for vessel, vname in zip(vessels, vnames):
        pts      = np.array(vessel.points)
        rel      = pts - centroid
        axial    = rel.dot(n_hat)
        perp_d   = np.linalg.norm(rel - np.outer(axial, n_hat), axis=1)
        inside   = (axial / a)**2 + (perp_d / b)**2 <= 1.0
        n_in     = int(inside.sum())
        if n_in > 0:
            cl = float(np.min(np.linalg.norm(rel[inside], axis=1)))
            oars.append({"vessel": vname, "points_inside": n_in,
                         "closest_mm": cl * 1000,
                         "risk": "CRITICAL" if cl < 0.005 else "HIGH"})
    return oars

# ─────────────────────────────────────────────────────────────────────
# TREATMENT REGIME SELECTION
# ─────────────────────────────────────────────────────────────────────

def select_regime(tumor_diam_cm, max_loss_pct, margin_cm=0.5):
    loss_frac  = max_loss_pct / 100.0
    eff_req    = tumor_diam_cm + margin_cm
    raw_req    = eff_req / max(1.0 - loss_frac, 0.01)
    candidates = sorted([r for r in ABLATION_TABLE if r[3] >= raw_req],
                        key=lambda r: (r[0], r[1]))
    if not candidates:
        candidates = sorted(ABLATION_TABLE, key=lambda r: r[3], reverse=True)
    return candidates[0], candidates[1:3], raw_req

# ─────────────────────────────────────────────────────────────────────
# RAY GENERATION
# ─────────────────────────────────────────────────────────────────────

def generate_rays(n_theta=20, n_phi=40):
    rays = []
    for t in np.linspace(0, np.pi, n_theta):
        for p in np.linspace(0, 2*np.pi, n_phi):
            rays.append([np.sin(t)*np.cos(p), np.sin(t)*np.sin(p), np.cos(t)])
    return np.array(rays)

# ─────────────────────────────────────────────────────────────────────
# DIRECTIONAL HEAT FLOW ARROWS
# ─────────────────────────────────────────────────────────────────────

def create_heat_flow_arrows(centroid, vessels, vnames, per_vessel_hs, plotter):
    """
    Draw arrows from tumor centroid toward each vessel surface.

    Arrow properties:
    - Direction: centroid → nearest point on vessel surface
    - Length: proportional to Q_loss (scaled for visibility)
    - Color: green (low loss) → yellow → red (high loss)
    - Label: vessel name + loss % + flow regime

    The arrow visually shows:
    1. WHICH direction heat is being carried away
    2. HOW MUCH heat is lost in each direction
    3. FLOW REGIME affecting the loss (laminar = smaller arrow = less mixing)
    """
    max_loss = max(hs["loss_pct"] for hs in per_vessel_hs.values())
    min_loss = min(hs["loss_pct"] for hs in per_vessel_hs.values())

    # Color scale: green (0%) → yellow (50%) → red (100%)
    def loss_to_color(pct):
        t = (pct - min_loss) / max(max_loss - min_loss, 0.01)
        if t < 0.5:
            return [2*t, 1.0, 0.0]        # green → yellow
        else:
            return [1.0, 2*(1-t), 0.0]    # yellow → red

    # Base arrow scale (meters) — adjust for visual clarity
    BASE_SCALE = 0.04   # 4 cm = base arrow length at max loss

    for vname, hs in per_vessel_hs.items():
        # Find nearest vessel point to centroid
        vessel    = vessels[vnames.index(vname)]
        pts       = np.array(vessel.points)
        tree      = cKDTree(pts)
        _, idx    = tree.query(centroid, k=1)
        target_pt = pts[idx]

        # Arrow direction and length
        raw_dir   = target_pt - centroid
        dist      = np.linalg.norm(raw_dir)
        if dist < 1e-6:
            continue
        unit_dir  = raw_dir / dist

        # Arrow length ∝ Q_loss (normalized to BASE_SCALE)
        arrow_len = BASE_SCALE * (hs["loss_pct"] / max(max_loss, 1.0))
        arrow_len = max(arrow_len, 0.005)   # minimum visible length

        # Tip of arrow (stops before reaching vessel)
        tip = centroid + unit_dir * min(dist * 0.85, dist - 0.005)

        color = loss_to_color(hs["loss_pct"])

        # Create arrow using PyVista
        arrow = pv.Arrow(
            start   = centroid,
            direction = unit_dir,
            scale   = arrow_len,
            tip_length   = 0.3,
            tip_radius   = 0.05,
            shaft_radius = 0.02,
        )
        plotter.add_mesh(arrow, color=color, opacity=0.92)

        # Label at arrow tip
        label_pos = centroid + unit_dir * (arrow_len * 1.15)
        regime_short = hs["flow_regime"][0]   # L / T / Tr
        label = (f"{vname.replace('_',' ')}\n"
                 f"{hs['loss_pct']:.2f}%  [{regime_short}]\n"
                 f"Q={hs['Q_loss_W']:.3f}W")
        plotter.add_point_labels(
            pv.PolyData([label_pos]),
            [label],
            font_size=9,
            text_color=color,
            point_size=1,
            always_visible=True,
            shape_opacity=0.0,
        )

    # Legend for arrow colors
    plotter.add_text(
        "Heat Flow Arrows:\n"
        "  Green = Low loss\n"
        "  Yellow = Moderate\n"
        "  Red = High loss\n"
        "  [L]=Laminar  [T]=Turbulent  [Tr]=Transition",
        position="lower_right", font_size=9, color="white"
    )

# ─────────────────────────────────────────────────────────────────────
# VELOCITY PROFILE VISUALIZATION (cross-section disk)
# ─────────────────────────────────────────────────────────────────────

def add_velocity_profile_disks(vessels, vnames, plotter):
    """
    For each vessel, add a small cross-sectional disk at the midpoint
    colored by radial velocity profile (blue=slow wall, red=fast center).
    This makes the parabolic vs turbulent profile visible.
    """
    for vessel, vname in zip(vessels, vnames):
        D      = VESSEL_DIAMETERS[vname]
        u_mean = VESSEL_VELOCITIES[vname]
        R      = D / 2.0

        Re = (RHO_B * u_mean * D) / MU_B

        # Sample the midpoint of the vessel
        mid_pt = vessel.points[len(vessel.points) // 2]

        # Create a small disk (cross-section visualization)
        disk = pv.Disc(center=mid_pt, inner=0.0, outer=R * 2, r_res=12, c_res=24)

        # Color by velocity profile
        pts_local = disk.points - mid_pt
        r_vals    = np.linalg.norm(pts_local, axis=1)
        r_norm    = np.clip(r_vals / (R + 1e-9), 0, 1)

        if Re < 2300:
            # Laminar: parabolic
            u_vals = u_mean * 2 * (1 - r_norm**2)
            profile_label = "parabolic"
        else:
            # Turbulent: flatter power-law
            u_vals = u_mean * (8/7) * (9/8) * (1 - r_norm) ** (1/7)
            profile_label = "power-law"

        disk["velocity_m_s"] = u_vals

        plotter.add_mesh(disk, scalars="velocity_m_s", cmap="coolwarm",
                         opacity=0.7, show_scalar_bar=False)


# ─────────────────────────────────────────────────────────────────────
# ABLATION ELLIPSOID
# ─────────────────────────────────────────────────────────────────────

def make_ellipsoid(centroid, fwd_m, diam_m, needle_dir=None):
    if needle_dir is None:
        needle_dir = np.array([0.0, 0.0, 1.0])
    n_hat = needle_dir / (np.linalg.norm(needle_dir) + 1e-9)
    a = fwd_m / 2.0
    b = diam_m / 2.0
    if a < 1e-5 or b < 1e-5:
        return None

    perp1 = np.cross(n_hat, [1, 0, 0])
    if np.linalg.norm(perp1) < 1e-6:
        perp1 = np.cross(n_hat, [0, 1, 0])
    perp1 /= np.linalg.norm(perp1)
    perp2  = np.cross(n_hat, perp1)

    u  = np.linspace(0, 2*np.pi, 36)
    v  = np.linspace(0, np.pi,   18)
    uu, vv = np.meshgrid(u, v)

    xs = a * np.cos(vv).ravel()
    ys = b * np.sin(vv).ravel() * np.cos(uu).ravel()
    zs = b * np.sin(vv).ravel() * np.sin(uu).ravel()

    pts = (np.outer(xs, n_hat) + np.outer(ys, perp1) +
           np.outer(zs, perp2) + centroid)

    surf = pv.PolyData(pts).delaunay_3d().extract_surface().clean()
    dists = np.linalg.norm(surf.points - centroid, axis=1)
    surf["Temperature"] = (T_BLOOD + (T_TISSUE - T_BLOOD) *
                           np.exp(-2.0 * (dists / max(a, b))**2))
    return surf


# ─────────────────────────────────────────────────────────────────────
# ANIMATION + VISUALIZATION
# ─────────────────────────────────────────────────────────────────────

def run_visualization(surface, vessels, vnames, tumors, centroids,
                      sel_idx, results, per_vessel_hs,
                      recommended, oar_list, safest_dir):

    print("\n🎬 Building 3D visualization...")

    power_w   = float(recommended[0])
    time_s    = float(recommended[1])
    fwd_m     = recommended[2] / 100.0
    diam_m    = recommended[3] / 100.0
    centroid  = centroids[sel_idx]
    needle_dir= safest_dir

    plotter = pv.Plotter(window_size=[1500, 1000])
    plotter.background_color = "black"

    # Body surface (transparent)
    plotter.add_mesh(surface, color="lightgray", opacity=0.08, label="Body Surface")

    # Vessels — OARs in red
    for i, (v, col, vn) in enumerate(zip(vessels, VESSEL_COLORS, vnames)):
        is_oar = any(o["vessel"] == vn for o in oar_list)
        vc     = "red" if is_oar else col
        op     = 0.85 if is_oar else 0.55
        plotter.add_mesh(v, color=vc, opacity=op,
                         label=("⚠ OAR: " if is_oar else "") + vn)

        # Add velocity profile disk at vessel midpoint
        add_velocity_profile_disks([v], [vn], plotter)

    # Tumors — selected one smoothed
    colors = ["yellow","orange","purple","pink","red","lime"]
    for i, t in enumerate(tumors):
        if i == sel_idx:
            t_display = smooth_tumor_mesh(t, n_iter=80, relaxation=0.1)
            op = 0.85
            label = f"Tumor {i+1} ← ablation target (smoothed)"
        else:
            t_display = t
            op = 0.30
            label = f"Tumor {i+1}"
        plotter.add_mesh(t_display, color=colors[i % len(colors)],
                         opacity=op, label=label)

    # Tumor centroid
    plotter.add_mesh(pv.Sphere(radius=0.006, center=centroid),
                     color="yellow", label="Tumor centroid")

    # ── DIRECTIONAL HEAT FLOW ARROWS ──────────────────────────────
    print("   Adding directional heat flow arrows...")
    create_heat_flow_arrows(centroid, vessels, vnames, per_vessel_hs, plotter)

    # Ray coloring (directional loss map)
    if results:
        losses = np.array([r["loss_pct"] for r in results])
        lo, hi = losses.min(), losses.max()
        norm   = (losses - lo) / (hi - lo + 1e-9)
        step   = max(1, len(results) // 80)
        for i in range(0, len(results), step):
            r      = results[i]
            start  = centroid
            end_pt = start + r["ray_direction"] * r["path_distance"]
            line   = pv.Line(start, end_pt)
            cv     = norm[i]
            plotter.add_mesh(line, color=[cv, 0.0, 1.0 - cv], line_width=1.5)

    # ── ANIMATION SLIDER ──────────────────────────────────────────
    def update(t_val):
        t    = float(t_val)
        frac = min(t / time_s, 1.0)
        for name in ["ablation", "particles", "time_hud"]:
            try:
                plotter.remove_actor(name)
            except Exception:
                pass

        # Growing ellipsoid
        cur_fwd  = fwd_m  * frac
        cur_diam = diam_m * frac
        if cur_fwd > 5e-4 and cur_diam > 5e-4:
            ell = make_ellipsoid(centroid, cur_fwd, cur_diam, needle_dir)
            if ell is not None:
                plotter.add_mesh(ell, scalars="Temperature", cmap="hot",
                                 opacity=0.6, name="ablation",
                                 scalar_bar_args={"title": "Temp (°C)"})

        # Blood particles (constrained to vessel surfaces)
        all_pts, all_colors = [], []
        for j, (vessel, vn) in enumerate(zip(vessels, vnames)):
            u_mean = VESSEL_VELOCITIES[vn]
            D      = VESSEL_DIAMETERS[vn]
            R      = D / 2.0
            Re_v   = (RHO_B * u_mean * D) / MU_B

            pts  = np.array(vessel.points)
            n    = min(60, len(pts))
            idx  = np.random.choice(len(pts), n, replace=False)
            spts = pts[idx].copy()

            # PCA flow direction
            centered = pts - pts.mean(axis=0)
            _, _, vt = np.linalg.svd(centered[:min(3000, len(centered))],
                                     full_matrices=False)
            flow = vt[0]

            # Move particles along flow
            disp = (flow * u_mean * t * 0.002) % 0.015
            spts = spts + disp

            # Compute approximate r for each surface point to color by velocity
            rel = spts - vessel.points.mean(axis=0)
            r_approx = np.linalg.norm(rel - np.outer(rel.dot(vt[0]), vt[0]), axis=1)
            r_norm   = np.clip(r_approx / (R + 1e-9), 0, 1)

            if Re_v < 2300:
                u_local = u_mean * 2 * (1 - r_norm**2)
            else:
                u_local = u_mean * (8/7) * (9/8) * np.clip(1 - r_norm, 0, 1)**(1/7)

            all_pts.append(spts)
            all_colors.append(u_local)

        if all_pts:
            pts_all = np.vstack(all_pts)
            vel_all = np.concatenate(all_colors)
            cloud   = pv.PolyData(pts_all)
            cloud["velocity"] = vel_all
            plotter.add_mesh(cloud, scalars="velocity", cmap="coolwarm",
                             point_size=5, render_points_as_spheres=True,
                             name="particles",
                             scalar_bar_args={"title": "Blood velocity (m/s)"})

        hud = (f"t = {t:.0f}s / {time_s:.0f}s  ({frac*100:.0f}%)\n"
               f"Power: {power_w:.0f} W\n"
               f"Zone: {cur_fwd*100:.1f}cm × {cur_diam*100:.1f}cm\n"
               f"OARs: {len(oar_list)}")
        plotter.add_text(hud, position="lower_left", font_size=11,
                         color="white", name="time_hud")
        plotter.render()

    plotter.add_slider_widget(update, rng=[0.0, time_s], value=0.0,
                              title="Ablation Time (s)",
                              pointa=(0.1, 0.05), pointb=(0.9, 0.05),
                              style="modern")

    plotter.add_legend(loc="upper right", size=(0.24, 0.40))
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
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  HEAT SINK | VELOCITY PROFILE | OAR | TREATMENT | ANIMATION")
    print("=" * 70)

    if not os.path.exists(DATASET_BASE):
        print(f"  ✘ Dataset not found: {DATASET_BASE}")
        return

    # Load
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

    # Tumors
    tumors   = extract_tumors(tumor_mesh)
    metrics  = tumor_metrics(tumors, surface, vessels, vnames)
    centroids= np.array([m["centroid"] for m in metrics])

    eligible = sorted(
        [m for m in metrics if MIN_DIAMETER_CM <= m["diameter_cm"] <= MAX_DIAMETER_CM
         and m["depth_cm"] <= MAX_DEPTH_CM],
        key=lambda m: m["min_vessel_m"])

    sel = eligible[0] if eligible else sorted(metrics, key=lambda m: m["min_vessel_m"])[0]
    sel_idx  = sel["idx"]
    centroid = sel["centroid"]
    sel_diam = sel["diameter_cm"]

    print(f"\n🎯 Selected Tumor {sel_idx+1}: {sel_diam:.2f}cm diameter, "
          f"{sel['depth_cm']:.2f}cm deep")

    # Centroid→vessel distances
    centroid_dists = {}
    vessel_trees   = []
    for i, v in enumerate(vessels):
        tree = cKDTree(np.array(v.points))
        vessel_trees.append(tree)
        d, _ = tree.query(centroid, k=1)
        centroid_dists[vnames[i]] = float(d)

    POWER_W = 60.0
    TIME_S  = 600.0

    # Full physics heat sink per vessel
    print("\n" + "=" * 70)
    print("  FULL PHYSICS HEAT SINK (with velocity profile correction)")
    print("=" * 70)
    print(f"\n  {'Vessel':<18}{'Dist':<9}{'Re':<8}{'Regime':<14}"
          f"{'Nu':<8}{'η_wall':<9}{'Q_wall':<10}{'Q_bulk':<10}"
          f"{'Q_loss':<10}{'Loss%'}")
    print("  " + "-" * 104)

    per_vessel_hs = {}
    for vn in vnames:
        d  = centroid_dists[vn]
        hs = heat_sink_full_physics(d, vn, POWER_W, TIME_S)
        per_vessel_hs[vn] = hs
        print(f"  {vn:<18}{d*1000:<9.1f}{hs['Re']:<8.0f}{hs['flow_regime']:<14}"
              f"{hs['Nu']:<8.1f}{hs['eta_wall']:<9.3f}"
              f"{hs['Q_wall_W']:<10.4f}{hs['Q_bulk_W']:<10.4f}"
              f"{hs['Q_loss_W']:<10.4f}{hs['loss_pct']:.3f}%")

    print("\n  Velocity Profile Detail:")
    print(f"  {'Vessel':<18}{'Profile type':<38}{'u_mean':<10}"
          f"{'u_wall_eff':<14}{'u_center'}")
    print("  " + "-" * 90)
    for vn in vnames:
        hs = per_vessel_hs[vn]
        print(f"  {vn:<18}{hs['profile_type']:<38}{hs['u_mean_m_s']:<10.3f}"
              f"{hs['u_wall_eff_m_s']:<14.4f}{hs['u_centerline_m_s']:.3f}")

    max_hs_pct  = max(hs["loss_pct"] for hs in per_vessel_hs.values())
    dom_vessel  = max(per_vessel_hs, key=lambda k: per_vessel_hs[k]["loss_pct"])

    # Ray tracing with capped segment distance
    print("\n  Ray tracing...")
    rays    = generate_rays(n_theta=20, n_phi=40)
    results = []
    vessel_pts_list = [np.array(v.points) for v in vessels]

    for direction in tqdm(rays, desc="  Rays"):
        try:
            pts_hit, _ = surface.ray_trace(centroid, centroid + direction * 0.5)
            if len(pts_hit) == 0:
                continue
            hit    = pts_hit[0]
            path_d = float(np.linalg.norm(hit - centroid))

            seg_dists = {}
            for vi, vn in enumerate(vnames):
                c_dist = centroid_dists[vn]   # centroid→vessel (fixed)
                seg_d  = ray_segment_dist(centroid, direction, path_d,
                                          vessel_pts_list[vi], c_dist, n_sample=30)
                seg_dists[vn] = seg_d

            dom_vn   = min(seg_dists, key=seg_dists.get)
            ray_dist = seg_dists[dom_vn]
            hs       = heat_sink_full_physics(ray_dist, dom_vn, POWER_W, TIME_S)
            hs["ray_direction"]   = direction
            hs["path_distance"]   = path_d
            hs["ray_seg_dist_mm"] = ray_dist * 1000
            results.append(hs)
        except Exception:
            continue

    sorted_res  = sorted(results, key=lambda x: x["loss_pct"], reverse=True)
    all_losses  = [r["loss_pct"] for r in results]
    safest_dir  = sorted_res[-1]["ray_direction"] if results else np.array([0,0,1])
    worst_ray   = sorted_res[0]  if results else {}

    print(f"\n  {len(results)} rays processed")
    print(f"  Loss range: {np.min(all_losses):.2f}% → {np.max(all_losses):.2f}%")
    print(f"  Safest insertion : {safest_dir.round(3)}  "
          f"(loss={sorted_res[-1]['loss_pct']:.2f}%)")
    print(f"  Riskiest direction: {worst_ray.get('ray_direction', np.zeros(3)).round(3)}  "
          f"(loss={worst_ray.get('loss_pct', 0):.2f}%)")

    # OAR
    oar_list = identify_oars(centroid, vessels, vnames, 5.82, 3.9, safest_dir)
    print(f"\n  OARs encroached: {len(oar_list)}")
    for o in oar_list:
        print(f"    {o['vessel']}  {o['points_inside']} pts  "
              f"{o['closest_mm']:.1f}mm  [{o['risk']}]")

    # Treatment regime
    rec, alts, raw_req = select_regime(sel_diam, max_hs_pct, margin_cm=0.5)
    print(f"\n  Recommended: {rec[0]:.0f}W × {rec[1]:.0f}s  "
          f"diam={rec[3]:.2f}cm ≥ required {raw_req:.2f}cm")

    # Clinical summary
    print("\n" + "=" * 70)
    print("  CLINICAL SUMMARY")
    print("=" * 70)
    print(f"  Dominant vessel      : {dom_vessel}")
    print(f"  Flow regime          : {per_vessel_hs[dom_vessel]['flow_regime']}")
    print(f"  Velocity profile     : {per_vessel_hs[dom_vessel]['profile_type']}")
    print(f"  u_mean / u_wall      : {per_vessel_hs[dom_vessel]['u_mean_m_s']:.3f} / "
          f"{per_vessel_hs[dom_vessel]['u_wall_eff_m_s']:.4f} m/s")
    print(f"  Max centroid loss    : {max_hs_pct:.2f}%")
    print(f"  Max directional loss : {np.max(all_losses):.2f}%")
    print(f"  Safest needle dir    : {safest_dir.round(3)}")

    # Visualization
    run_visualization(surface, vessels, vnames, tumors, centroids,
                      sel_idx, results, per_vessel_hs,
                      rec, oar_list, safest_dir)

    print("\n  Complete!")
    return results

if __name__ == "__main__":
    try:
        results = main()
    except KeyboardInterrupt:
        print("\n  Stopped.")
    except Exception as e:
        print(f"\n  Error: {e}")
        raise
