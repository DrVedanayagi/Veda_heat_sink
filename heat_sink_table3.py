#!/usr/bin/env python3
"""
Integrated Heat Sink Analysis — v4 (Bug Fixes)
===============================================

FIXES IN THIS VERSION vs v3
----------------------------
FIX 1 — ELLIPSOID: Replace delaunay_3d point-cloud approach with
         pv.ParametricEllipsoid() → eliminates ALL VTK degenerate-triangle
         warnings ("Unable to factor linear system", "mesh quality suspect").

FIX 2 — BLOOD PARTICLES: Replace erratic modulo-wrap with proper looping
         flow along the vessel PCA axis. Each particle has a fixed phase
         offset so they spread evenly along the vessel. At t=0 they start
         at random positions; as the slider advances they march smoothly
         toward the flow direction and loop back to the start continuously.
         Colored blue (slow, near-wall) → red (fast, centerline).

FIX 3 — ABLATION COLORMAP: Change from "hot" (white center = confusing) to
         "plasma" with explicit clim=[37, 90] and a labeled scalar bar showing
         °C. Add a text annotation explaining what the growing zone means.

FIX 4 — Re/Nu EXPLANATION: Added detailed print block showing why Re and Nu
         are what they are, so the values are transparent and auditable.
         Re = ρuD/μ — fixed by anatomy/physiology, not tunable.
         Nu switches by regime (laminar/transition/turbulent).

NOTE on Pylance errors in .ipynb:
  These are NOT Python errors. Pylance misreads Jupyter notebook JSON cells
  as raw Python. To suppress in VS Code:
    settings.json → add: "python.analysis.exclude": ["**/*.ipynb"]
  OR just ignore them — they do not affect script execution.
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
    os.path.join(PORTALVENOUS_DIR, "908ac52300001.vtk"),
    os.path.join(PORTALVENOUS_DIR, "908ac52300002.vtk"),
    os.path.join(PORTALVENOUS_DIR, "908ac52300003.vtk"),
    os.path.join(PORTALVENOUS_DIR, "908ac52300004.vtk"),
    os.path.join(PORTALVENOUS_DIR, "908ac52300005.vtk"),
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

# ─────────────────────────────────────────────────────────────────────
# FIX 1 — CLEAN ELLIPSOID using pv.ParametricEllipsoid
# No more delaunay_3d → no degenerate triangles → no VTK warnings
# ─────────────────────────────────────────────────────────────────────

def make_ellipsoid_clean(centroid, fwd_m, diam_m, needle_dir=None):
    """
    Build ablation ellipsoid using pv.ParametricEllipsoid — clean tessellation,
    zero VTK warnings.

    The ParametricEllipsoid is created in local space with:
        xradius = diam_m / 2   (transverse)
        yradius = diam_m / 2   (transverse)
        zradius = fwd_m  / 2   (along needle)

    Then rotated to align with needle_dir and translated to centroid.

    Temperature mapped as Gaussian decay from center:
        T(r_norm) = T_blood + (T_tissue - T_blood) × exp(-2 × r_norm²)
    where r_norm = normalised distance (0=center, 1=surface).
    """
    if fwd_m < 1e-4 or diam_m < 1e-4:
        return None

    a = diam_m / 2.0   # transverse semi-axis
    c = fwd_m  / 2.0   # axial semi-axis (along needle)

    # Build ellipsoid centred at origin using ParametricEllipsoid
    ell = pv.ParametricEllipsoid(xradius=a, yradius=a, zradius=c,
                                 u_res=30, v_res=30, w_res=10)

    # Rotate so z-axis aligns with needle_dir
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

    # Translate to centroid
    ell.points += centroid

    # Temperature scalar: Gaussian from centre
    r_norm = np.linalg.norm(ell.points - centroid, axis=1) / (max(a, c) + 1e-9)
    ell["Temperature_C"] = T_BLOOD + (T_TISSUE - T_BLOOD) * np.exp(-2.0 * r_norm**2)

    return ell


# ─────────────────────────────────────────────────────────────────────
# FIX 2 — SMOOTH LOOPING BLOOD PARTICLES
# Particles march along PCA flow axis, loop seamlessly
# Colored by radial position (blue=wall, red=center)
# ─────────────────────────────────────────────────────────────────────

class VesselParticleSystem:
    """
    Pre-computed particle system for one vessel.
    Call update(t) to get current positions and velocity colors.

    Each particle i has:
        base_pos  : fixed start position on vessel surface
        phase_i   : offset in [0, vessel_length] so particles spread evenly
        r_norm_i  : normalised radial distance from vessel axis (0=center, 1=wall)
        u_local_i : local velocity at that radial position (for coloring)
    """

    def __init__(self, vessel, vessel_name, n_particles=80):
        pts   = np.array(vessel.points)
        D     = VESSEL_DIAMETERS[vessel_name]
        R     = D / 2.0
        u_mean= VESSEL_VELOCITIES[vessel_name]
        Re    = (RHO_B * u_mean * D) / MU_B

        # PCA → principal flow axis (vessel long axis)
        centered = pts - pts.mean(axis=0)
        _, _, vt = np.linalg.svd(centered[:min(5000, len(centered))],
                                  full_matrices=False)
        self.flow_dir = vt[0]                        # unit vector along vessel
        self.origin   = pts.mean(axis=0)

        # Project all points onto flow axis to find vessel extent
        proj    = centered.dot(self.flow_dir)
        self.L  = float(proj.max() - proj.min())     # vessel length along axis
        self.L  = max(self.L, 0.02)                  # minimum 2 cm

        # Sample n particles on vessel surface
        idx  = np.random.choice(len(pts), min(n_particles, len(pts)), replace=False)
        spts = pts[idx]

        # Radial distance from vessel axis for each particle
        rel      = spts - self.origin
        axial_c  = np.outer(rel.dot(self.flow_dir), self.flow_dir)
        perp     = rel - axial_c
        r_vals   = np.linalg.norm(perp, axis=1)
        self.r_norm = np.clip(r_vals / (R + 1e-9), 0.0, 1.0)

        # Velocity at each particle's radial position
        if Re < 2300:
            # Laminar parabolic: u(r) = 2u_mean(1 - r²/R²)
            self.u_local = u_mean * 2.0 * (1.0 - self.r_norm**2)
        else:
            # Turbulent power-law: u(r) = u_max(1 - r/R)^(1/7)
            u_max = u_mean * (8/7) * (9/8)
            self.u_local = u_max * (1.0 - self.r_norm)**(1.0/7.0)

        # Phase offset: spread particles evenly along vessel length
        self.phase = np.random.uniform(0.0, self.L, len(idx))

        # Base positions projected onto axis start
        axial_pos_i  = rel.dot(self.flow_dir)                  # current axial coord
        self.base_pts= spts - np.outer(axial_pos_i, self.flow_dir)  # zero-out axial

        self.vessel_name = vessel_name
        self.u_mean      = u_mean
        self.Re          = Re
        self.n           = len(idx)
        # Scale factor: 1 unit of t → this many meters of flow
        # We slow it down 500× for visual clarity
        self.speed_scale = u_mean / 500.0

    def update(self, t):
        """Return current particle positions and velocity colors for time t."""
        # Each particle advances by speed_scale × t + its phase, mod vessel length
        axial_advance = (self.phase + self.speed_scale * t) % self.L
        # Reconstruct 3D positions
        pts = self.base_pts + np.outer(axial_advance, self.flow_dir)
        return pts, self.u_local   # positions, scalar for colormap


# ─────────────────────────────────────────────────────────────────────
# PHYSICS — Nu, Re, heat sink (unchanged from v3, with explanation print)
# ─────────────────────────────────────────────────────────────────────

def nusselt_full(Re, Pr):
    """
    Re < 2300    → Laminar  : Nu = 4.36 (const heat flux, fully developed)
    2300-10000   → Transition: Gnielinski with Petukhov friction factor
    ≥ 10000      → Turbulent : Dittus-Boelter
    """
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
    # approximate u_mean from any vessel — correction is weakly dependent on it
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
    """
    FIX 4 — Print a transparent explanation of why Re and Nu have
    their specific values, so they can be audited.
    """
    print("\n" + "=" * 70)
    print("  Re AND Nu EXPLANATION — why these values, not others")
    print("=" * 70)
    print("""
  Re = ρ × u_mean × D / μ     (Reynolds number — fixed by anatomy+physiology)
  ─────────────────────────────────────────────────────────────────
  ρ (blood density)  = 1060 kg/m³       ← literature constant
  μ (blood viscosity)= 3.5×10⁻³ Pa·s   ← non-Newtonian average
  D (vessel diameter)= anatomical value per vessel (see table)
  u_mean             = mean blood velocity per vessel (see table)

  These are NOT adjustable — they reflect the vessel's real biology.
  Re changes ONLY if diameter or flow speed changes (e.g. vasospasm,
  hypertension, or portal hypertension would alter these values).

  Nu = heat transfer effectiveness (dimensionless)
  ─────────────────────────────────────────────────────────────────
  Re < 2300   → LAMINAR    → Nu = 4.36  (constant, parabolic profile,
                                          heat exchange limited by slow
                                          near-wall blood)
  2300-10000  → TRANSITION → Gnielinski: Nu = f(Re,Pr)  ← higher mixing
  ≥ 10000     → TURBULENT  → Dittus-Boelter: Nu = 0.023 Re^0.8 Pr^0.4
""")
    print(f"  {'Vessel':<18} {'D(mm)':<8} {'u(m/s)':<9} {'Re':<8} "
          f"{'Regime':<14} {'Nu':<8} {'Why Nu is this value'}")
    print("  " + "-" * 90)
    for vn, hs in per_vessel_hs.items():
        D = VESSEL_DIAMETERS[vn] * 1000
        u = VESSEL_VELOCITIES[vn]
        why = {
            "Laminar":    "Re<2300 → parabolic profile → Nu fixed at 4.36",
            "Transition": "2300<Re<10000 → Gnielinski correlation",
            "Turbulent":  "Re≥10000 → Dittus-Boelter, strong mixing",
        }[hs["flow_regime"]]
        print(f"  {vn:<18} {D:<8.1f} {u:<9.2f} {hs['Re']:<8.0f} "
              f"{hs['flow_regime']:<14} {hs['Nu']:<8.1f} {why}")


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
    print(f"\n  {'T':<5}{'Diam(cm)':<11}{'Depth(cm)':<11}", end="")
    for n in vnames:
        print(f"{n[:10]+'(mm)':<17}", end="")
    print()
    for m in metrics:
        print(f"  {m['idx']+1:<5}{m['diameter_cm']:<11.2f}{m['depth_cm']:<11.2f}", end="")
        for d in m["vessel_dists_m"]:
            print(f"{d*1000:<17.1f}", end="")
        print()
    return metrics


def ray_segment_dist(origin, direction, path_d, vessel_pts, centroid_dist, n_sample=30):
    ts      = np.linspace(0.0, path_d, n_sample)
    samples = origin + np.outer(ts, direction)
    dists,_ = cKDTree(vessel_pts).query(samples, k=1)
    raw     = float(np.min(dists))
    return max(raw, centroid_dist * 0.5)


def identify_oars(centroid, vessels, vnames, fwd_cm, diam_cm, needle_dir=None):
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
            cl = float(np.min(np.linalg.norm(rel[inside], axis=1)))
            oars.append({"vessel": vname, "points_inside": n_in,
                         "closest_mm": cl * 1000,
                         "risk": "CRITICAL" if cl < 0.005 else "HIGH"})
    return oars


def select_regime(tumor_diam_cm, max_loss_pct, margin_cm=0.5):
    lf  = max_loss_pct / 100.0
    req = (tumor_diam_cm + margin_cm) / max(1.0 - lf, 0.01)
    cands = sorted([r for r in ABLATION_TABLE if r[3] >= req],
                   key=lambda r: (r[0], r[1]))
    if not cands:
        cands = sorted(ABLATION_TABLE, key=lambda r: r[3], reverse=True)
    return cands[0], cands[1:3], req


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
# MAIN VISUALIZATION — with all 3 animation fixes applied
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

    # Body surface
    plotter.add_mesh(surface, color="lightgray", opacity=0.08, label="Body Surface")

    # Vessels — OARs in red
    for i, (v, col, vn) in enumerate(zip(vessels, VESSEL_COLORS, vnames)):
        is_oar = any(o["vessel"] == vn for o in oar_list)
        plotter.add_mesh(v, color="red" if is_oar else col,
                         opacity=0.85 if is_oar else 0.55,
                         label=("⚠ OAR: " if is_oar else "") + vn)

    # Tumors — selected one smoothed
    colors = ["yellow", "orange", "purple", "pink", "red", "lime"]
    for i, t in enumerate(tumors):
        td    = smooth_tumor(t) if i == sel_idx else t
        op    = 0.85 if i == sel_idx else 0.30
        label = (f"Tumor {i+1} ablation target (smoothed)"
                 if i == sel_idx else f"Tumor {i+1}")
        plotter.add_mesh(td, color=colors[i % len(colors)], opacity=op, label=label)

    plotter.add_mesh(pv.Sphere(radius=0.006, center=centroid),
                     color="yellow", label="Tumor centroid")

    # Heat flow arrows
    create_heat_flow_arrows(centroid, vessels, vnames, per_vessel_hs, plotter)

    # Ray directional loss lines
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
                             color=[cv, 0.0, 1.0 - cv], line_width=1.5)

    # ── SLIDER CALLBACK ──────────────────────────────────────────────

    def update(t_val):
        t    = float(t_val)
        frac = min(t / time_s, 1.0)

        for name in ["ablation", "particles", "hud"]:
            try:
                plotter.remove_actor(name)
            except Exception:
                pass

        # FIX 1: Clean ellipsoid — no VTK warnings
        cur_fwd  = fwd_m  * frac
        cur_diam = diam_m * frac
        ell = make_ellipsoid_clean(centroid, cur_fwd, cur_diam, needle_dir)
        if ell is not None:
            # FIX 3: plasma colormap, explicit clim, labeled scalar bar
            plotter.add_mesh(
                ell,
                scalars="Temperature_C",
                cmap="plasma",
                clim=[T_BLOOD, T_TISSUE],        # 37–90 °C explicit range
                opacity=0.62,
                name="ablation",
                scalar_bar_args={
                    "title": "Temperature (°C)",
                    "n_labels": 5,
                    "label_font_size": 11,
                    "title_font_size": 12,
                    "position_x": 0.02,
                    "position_y": 0.25,
                    "width": 0.08,
                    "height": 0.40,
                    "color": "white",
                }
            )

        # FIX 2: Smooth looping particles
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
                cloud,
                scalars="blood_velocity_m_s",
                cmap="coolwarm",
                clim=[0.0, max(VESSEL_VELOCITIES.values()) * 2.0],
                point_size=5,
                render_points_as_spheres=True,
                name="particles",
                scalar_bar_args={
                    "title": "Blood velocity (m/s)",
                    "n_labels": 3,
                    "label_font_size": 10,
                    "title_font_size": 11,
                    "position_x": 0.12,
                    "position_y": 0.25,
                    "width": 0.08,
                    "height": 0.40,
                    "color": "white",
                }
            )

        # HUD
        zone_str = (f"Zone: {cur_fwd*100:.1f}cm × {cur_diam*100:.1f}cm"
                    if frac > 0.01 else "Zone: growing...")
        hud = (f"t = {t:.0f}s / {time_s:.0f}s  ({frac*100:.0f}%)\n"
               f"Power: {power_w:.0f} W\n"
               f"{zone_str}\n"
               f"OARs encroached: {len(oar_list)}\n"
               f"─────────────────────────\n"
               f"Ablation zone colour:\n"
               f"  Purple=cool(37°C)\n"
               f"  Yellow=warm\n"
               f"  White=hot(90°C)\n"
               f"Blood dots:\n"
               f"  Blue=slow(wall)\n"
               f"  Red=fast(centre)")
        plotter.add_text(hud, position="lower_left", font_size=10,
                         color="white", name="hud")
        plotter.render()

    plotter.add_slider_widget(
        update, rng=[0.0, time_s], value=0.0,
        title="Ablation Time (s)",
        pointa=(0.10, 0.05), pointb=(0.90, 0.05),
        style="modern")

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
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  HEAT SINK v4 — ELLIPSOID FIX + SMOOTH PARTICLES + COLORMAP FIX")
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

    # Tumors
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

    # Centroid→vessel distances
    centroid_dists = {}
    vessel_trees   = []
    for i, v in enumerate(vessels):
        tree = cKDTree(np.array(v.points))
        vessel_trees.append(tree)
        centroid_dists[vnames[i]] = float(tree.query(centroid, k=1)[0])

    POWER_W = 60.0
    TIME_S  = 600.0

    # Heat sink physics
    print("\n  Computing heat sink...")
    per_vessel_hs = {}
    for vn in vnames:
        per_vessel_hs[vn] = heat_sink_full_physics(
            centroid_dists[vn], vn, POWER_W, TIME_S)

    # FIX 4: Print Re/Nu explanation
    print_re_nu_explanation(per_vessel_hs)

    # Summary table
    print(f"\n  {'Vessel':<18}{'Dist(mm)':<11}{'Regime':<14}"
          f"{'Nu':<8}{'Q_loss(W)':<12}{'Loss%'}")
    print("  " + "-" * 68)
    for vn, hs in per_vessel_hs.items():
        print(f"  {vn:<18}{hs['dist_mm']:<11.1f}{hs['flow_regime']:<14}"
              f"{hs['Nu']:<8.1f}{hs['Q_loss_W']:<12.4f}{hs['loss_pct']:.3f}%")

    max_hs_pct = max(hs["loss_pct"] for hs in per_vessel_hs.values())
    dom_vessel = max(per_vessel_hs, key=lambda k: per_vessel_hs[k]["loss_pct"])

    # Ray tracing
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

    # OAR
    oar_list = identify_oars(centroid, vessels, vnames, 5.82, 3.9, safest_dir)
    print(f"\n  OARs: {len(oar_list)}")
    for o in oar_list:
        print(f"    {o['vessel']}  {o['points_inside']} pts  "
              f"{o['closest_mm']:.1f}mm  [{o['risk']}]")

    # Treatment
    rec, alts, raw_req = select_regime(sel_diam, max_hs_pct, margin_cm=0.5)
    print(f"\n  Recommended: {rec[0]:.0f}W × {rec[1]:.0f}s  "
          f"diam={rec[3]:.2f}cm (need {raw_req:.2f}cm)")

    # FIX 2: Pre-build particle systems (once, not per frame)
    print("\n  Building particle systems...")
    particle_systems = []
    for v, vn in zip(vessels, vnames):
        ps = VesselParticleSystem(v, vn, n_particles=80)
        particle_systems.append(ps)
        print(f"   {vn}: {ps.n} particles, Re={ps.Re:.0f} "
              f"({'Laminar' if ps.Re < 2300 else 'Turbulent/Transition'}), "
              f"L={ps.L*100:.1f}cm")

    # Visualize
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
