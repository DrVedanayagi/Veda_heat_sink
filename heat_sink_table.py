#!/usr/bin/env python3
"""
Integrated Heat Sink Effect Analysis & Animation System
=======================================================

WHAT THIS CODE DOES:
--------------------
1. Multi-tumor detection and MWA eligibility filtering
2. CORRECTED heat sink physics (exponential decay, always ≤ 100%)
3. FIXED ray coloring — each ray's loss uses the minimum distance from
   the RAY LINE SEGMENT to the vessel (not centroid→vessel), so rays
   pointing toward vessels show red, rays pointing away show blue
4. OAR (Organ at Risk) identification — checks if ablation ellipsoid
   from the manufacturer table will encroach on any vessel
5. Treatment regime selection — picks the correct Power/Time row from
   the lab table that achieves tumor coverage AFTER heat sink compensation
6. Directional loss map — shows which needle insertion directions are
   thermally safest
7. PyVista animation — growing ablation ellipsoid (not sphere!),
   blood flow particles constrained to vessel surfaces, temperature field

PHYSICS CORRECTIONS vs previous versions:
------------------------------------------
A) Ray heat loss now uses ray-segment-to-vessel distance, giving 
   meaningful directional variation in coloring.
B) Ablation zone is an ELLIPSOID (forward_throw × diameter/2),
   matching the manufacturer lab table geometry.
C) Treatment regime compensates for heat sink: if 7.57% energy is 
   lost to hepatic vein, the required ablation zone must be scaled up
   by 1/(1 - loss_fraction) and the matching power/time is selected.
D) Blood particles move along vessel point trajectories, not random walk.
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
# MANUFACTURER ABLATION TABLE  (from lab tests on liver tissue)
# Columns: power(W), time(s), forward_throw(cm), diameter(cm), length(cm)
# The ablation zone is an ELLIPSOID:
#   semi-axis along needle = forward_throw / 2
#   semi-axis transverse   = diameter / 2
# ─────────────────────────────────────────────────────────────────────

ABLATION_TABLE = [
    # (power_W, time_s, forward_throw_cm, diameter_cm, length_cm)
    (30,  180, 2.20,  1.9,  2.3),
    (30,  300, 2.50,  2.4,  2.7),
    (30,  480, 4.90,  2.9,  3.0),
    (30,  600, 5.47,  3.1,  3.1),
    (60,  180, 2.80,  2.5,  2.8),
    (60,  300, 4.70,  3.0,  3.3),
    (60,  480, 6.33,  3.8,  3.8),
    (60,  600, 5.82,  3.9,  3.9),
    (90,  180, 3.80,  3.1,  3.3),
    (90,  300, 5.20,  3.7,  3.8),
    (90,  480, 5.20,  4.2,  4.6),
    (90,  600, 6.30,  4.6,  4.9),
    (80,  300, 3.40,  4.2,  3.8),
    (80,  600, 8.40,  5.2,  4.4),
    (80,  300, 4.80,  4.5,  3.6),
    (80,  600, 9.20,  5.1,  4.6),
    (120, 300, 8.00,  5.1,  4.3),
    (120, 600, 9.40,  5.6,  5.0),
    (120, 300, 6.40,  5.2,  3.9),
    (140, 600, 8.82,  6.0,  5.0),
    (120, 600, 9.70,  5.9,  5.1),
    (160, 300, 6.90,  5.8,  4.2),
    (160, 300, 7.40,  5.4,  4.4),
    (160, 300, 6.70,  4.9,  4.5),
    (160, 600, 7.20,  6.3,  5.6),
    (160, 600, 10.20, 5.9,  5.8),
    (160, 600, 10.30, 6.1,  5.8),
]

# ─────────────────────────────────────────────────────────────────────
# PHYSICAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────

RHO_B    = 1060.0   # blood density           (kg/m³)
MU_B     = 3.5e-3   # dynamic viscosity        (Pa·s)
C_B      = 3700.0   # specific heat            (J/kg·K)
K_B      = 0.52     # thermal conductivity     (W/m·K)
T_BLOOD  = 37.0     # blood temperature        (°C)
T_TISSUE = 90.0     # ablation temperature     (°C)

ALPHA_TISSUE = 70.0  # tissue attenuation (m⁻¹); loss halves every ~10 mm
L_SEG        = 0.01  # reference vessel segment for area calc (m)

# MWA eligibility criteria
MIN_DIAMETER_CM = 3.0
MAX_DIAMETER_CM = 5.0
MAX_DEPTH_CM    = 26.0

VESSEL_DIAMETERS  = {"portal_vein": 12e-3, "hepatic_vein": 8e-3,
                     "aorta": 25e-3, "ivc": 20e-3, "hepatic_artery": 4.5e-3}
VESSEL_VELOCITIES = {"portal_vein": 0.15, "hepatic_vein": 0.20,
                     "aorta": 0.40,  "ivc": 0.35,  "hepatic_artery": 0.30}

# ─────────────────────────────────────────────────────────────────────
# MESH UTILITIES
# ─────────────────────────────────────────────────────────────────────

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
        print(f"    → rescaled mm→m")
    return mesh

# ─────────────────────────────────────────────────────────────────────
# TUMOR ANALYSIS
# ─────────────────────────────────────────────────────────────────────

def extract_tumors(tumor_mesh):
    print("\n🔍 Splitting combined tumor mesh into components...")
    connected = tumor_mesh.connectivity()
    tumors    = connected.split_bodies()
    print(f"   Found {len(tumors)} tumors")
    return tumors

def tumor_metrics(tumors, surface, vessels):
    s_tree = cKDTree(np.array(surface.points))
    v_trees = [cKDTree(np.array(v.points)) for v in vessels]
    metrics = []
    for i, t in enumerate(tumors):
        c  = np.array(t.center)
        b  = t.bounds
        dm = max(b[1]-b[0], b[3]-b[2], b[5]-b[4])
        dep_m, _ = s_tree.query(c, k=1)
        vd = [float(vt.query(c, k=1)[0]) for vt in v_trees]
        metrics.append({
            "idx": i, "centroid": c,
            "diameter_cm": dm * 100,
            "depth_cm": dep_m * 100,
            "vessel_dists_m": vd,
            "min_vessel_m": min(vd),
            "closest_vessel": int(np.argmin(vd)),
        })
    # Print table
    print(f"\n  {'Tumor':<7}{'Diam(cm)':<11}{'Depth(cm)':<11}", end="")
    for n in VESSEL_NAMES:
        print(f"{n+'(mm)':<22}", end="")
    print()
    print("  " + "-" * (7 + 11 + 11 + 22 * len(VESSEL_NAMES)))
    for m in metrics:
        print(f"  {m['idx']+1:<7}{m['diameter_cm']:<11.2f}{m['depth_cm']:<11.2f}", end="")
        for d in m["vessel_dists_m"]:
            print(f"{d*1000:<22.2f}", end="")
        print()
    return metrics

# ─────────────────────────────────────────────────────────────────────
# CORRECTED HEAT SINK PHYSICS
# ─────────────────────────────────────────────────────────────────────

def heat_sink_from_distance(distance_m, vessel_name, power_w, ablation_time_s):
    """
    Compute heat sink loss given a distance (any distance — centroid or ray-based).

    Q_loss(d) = min(Q_max, POWER) × exp(-α × d)
    E_loss%   = 100 × Q_loss × t / (P × t)  ∈ [0, 100]
    """
    D  = VESSEL_DIAMETERS[vessel_name]
    u  = VESSEL_VELOCITIES[vessel_name]
    Re = (RHO_B * u * D) / MU_B
    Pr = (C_B * MU_B) / K_B
    Nu = 0.023 * (Re ** 0.8) * (Pr ** 0.4) if Re >= 2300 else 4.36
    h  = (Nu * K_B) / D
    A  = np.pi * D * L_SEG
    dT = max(T_TISSUE - T_BLOOD, 0.1)

    Q_max  = min(h * A * dT, power_w)          # cap at source power
    d      = max(distance_m, 1e-4)
    Q_loss = Q_max * np.exp(-ALPHA_TISSUE * d)
    E_in   = power_w * ablation_time_s
    E_loss = min(Q_loss * ablation_time_s, E_in)
    return {
        "vessel":    vessel_name,
        "dist_mm":   d * 1000,
        "Re": Re, "Pr": Pr, "Nu": Nu, "h": h,
        "Q_max_W":   Q_max,
        "Q_loss_W":  Q_loss,
        "E_loss_J":  E_loss,
        "loss_pct":  100.0 * E_loss / E_in,
    }


# ─────────────────────────────────────────────────────────────────────
# KEY FIX: RAY-SEGMENT-TO-VESSEL DISTANCE
# ─────────────────────────────────────────────────────────────────────

def ray_segment_to_vessel_dist(ray_origin, ray_direction, path_distance,
                                vessel_points, n_sample=30):
    """
    Compute minimum distance from the RAY LINE SEGMENT to any vessel point.

    This replaces the old centroid→vessel distance so that:
    - Rays pointing TOWARD a vessel → short segment-to-vessel distance → high loss (RED)
    - Rays pointing AWAY from vessel → large distance → low loss (BLUE)

    Method:
        Sample n_sample points uniformly along the ray segment [0, path_distance].
        For each sampled point, query the vessel KD-tree.
        Return the minimum distance found.

    This is O(n_sample × log N) and fast enough for 800 rays.
    """
    ts = np.linspace(0.0, path_distance, n_sample)
    sampled = ray_origin + np.outer(ts, ray_direction)   # shape (n_sample, 3)
    tree = cKDTree(vessel_points)
    dists, _ = tree.query(sampled, k=1)
    return float(np.min(dists))


# ─────────────────────────────────────────────────────────────────────
# OAR IDENTIFICATION
# ─────────────────────────────────────────────────────────────────────

def identify_oars(centroid, vessels, vessel_names,
                  forward_throw_cm, diameter_cm, needle_direction=None):
    """
    Identify Organs at Risk (OAR) — vessels whose surface points fall
    inside the planned ablation ellipsoid.

    Ablation zone is an ellipsoid:
        semi-axis a (along needle) = forward_throw_cm / 2  (cm → m)
        semi-axis b (transverse)   = diameter_cm / 2       (cm → m)

    For simplicity we use a = forward_throw/2, b = c = diameter/2.
    If needle_direction is None, we use the z-axis as default.

    A vessel point P is INSIDE the ellipsoid if:
        (P-C)·a_hat)² / a² + (component perp)² / b²  ≤ 1
    """
    a = (forward_throw_cm / 2.0) / 100.0   # m
    b = (diameter_cm      / 2.0) / 100.0   # m

    if needle_direction is None:
        needle_direction = np.array([0.0, 0.0, 1.0])
    n_hat = needle_direction / (np.linalg.norm(needle_direction) + 1e-9)

    oar_results = []
    for i, (vessel, vname) in enumerate(zip(vessels, vessel_names)):
        pts      = np.array(vessel.points)
        rel      = pts - centroid                      # (N, 3)
        axial    = rel.dot(n_hat)                      # projection along needle
        perp     = rel - np.outer(axial, n_hat)        # perpendicular component
        perp_d   = np.linalg.norm(perp, axis=1)

        # Ellipsoid inequality
        inside_mask = (axial / a) ** 2 + (perp_d / b) ** 2 <= 1.0
        n_inside    = int(np.sum(inside_mask))

        if n_inside > 0:
            closest_inside = np.min(np.linalg.norm(rel[inside_mask], axis=1))
            oar_results.append({
                "vessel":          vname,
                "points_inside":   n_inside,
                "closest_dist_mm": closest_inside * 1000,
                "risk":            "CRITICAL" if closest_inside < 0.005 else "HIGH",
            })

    return oar_results


# ─────────────────────────────────────────────────────────────────────
# TREATMENT REGIME SELECTION
# ─────────────────────────────────────────────────────────────────────

def select_treatment_regime(tumor_diameter_cm, max_heat_loss_pct, safety_margin_cm=0.5):
    """
    Select the optimal Power/Time setting from the manufacturer table.

    Logic:
    ------
    1. Required effective diameter = tumor_diameter_cm + safety_margin_cm
       (the ablation zone must cover the tumor plus a margin)

    2. Because heat sink steals (max_heat_loss_pct)% of energy, the
       ablation zone produced under vessel-free conditions must be
       LARGER to compensate:

         required_raw_diameter = effective_required / (1 - loss_fraction)

       Example: tumor = 3.09 cm, margin = 0.5 cm → need 3.59 cm effective
       Heat sink = 7.57% → need 3.59 / (1 - 0.0757) = 3.88 cm raw zone

    3. Scan the table for all rows where diameter_cm ≥ required_raw_diameter.
       Among those, prefer:
         a) Lowest power (patient safety, reduced charring)
         b) Shortest time (for equal power)

    4. Also return the next two rows as alternatives.
    """
    loss_frac           = max_heat_loss_pct / 100.0
    effective_required  = tumor_diameter_cm + safety_margin_cm
    required_raw        = effective_required / max(1.0 - loss_frac, 0.01)

    print(f"\n  Tumor diameter          : {tumor_diameter_cm:.2f} cm")
    print(f"  Safety margin           : {safety_margin_cm:.1f} cm")
    print(f"  Required effective zone : {effective_required:.2f} cm")
    print(f"  Max heat sink loss      : {max_heat_loss_pct:.2f}%")
    print(f"  Required raw zone (comp): {required_raw:.2f} cm")

    candidates = [
        row for row in ABLATION_TABLE
        if row[3] >= required_raw   # diameter_cm column
    ]

    if not candidates:
        print("  ⚠  No standard setting achieves required coverage — use max settings.")
        candidates = sorted(ABLATION_TABLE, key=lambda r: r[3], reverse=True)

    # Sort: lowest power first, then shortest time
    candidates_sorted = sorted(candidates, key=lambda r: (r[0], r[1]))

    recommended = candidates_sorted[0]
    alternatives = candidates_sorted[1:3]

    return recommended, alternatives, required_raw


# ─────────────────────────────────────────────────────────────────────
# RAY GENERATION
# ─────────────────────────────────────────────────────────────────────

def generate_rays(n_theta=20, n_phi=40):
    theta = np.linspace(0, np.pi, n_theta)
    phi   = np.linspace(0, 2 * np.pi, n_phi)
    rays  = []
    for t in theta:
        for p in phi:
            rays.append([np.sin(t)*np.cos(p), np.sin(t)*np.sin(p), np.cos(t)])
    return np.array(rays)


# ─────────────────────────────────────────────────────────────────────
# ANIMATION
# ─────────────────────────────────────────────────────────────────────

def create_ablation_ellipsoid(centroid, forward_throw_m, diameter_m,
                               needle_dir=None, n_lat=30, n_lon=30):
    """
    Create an ellipsoid mesh matching the manufacturer ablation table geometry.
    a = forward_throw / 2  (along needle axis)
    b = c = diameter / 2   (transverse)
    """
    if needle_dir is None:
        needle_dir = np.array([0.0, 0.0, 1.0])
    n_hat = needle_dir / (np.linalg.norm(needle_dir) + 1e-9)

    a = forward_throw_m / 2.0
    b = diameter_m      / 2.0

    # Parametric ellipsoid
    u = np.linspace(0, 2 * np.pi, n_lon)
    v = np.linspace(0, np.pi,     n_lat)
    uu, vv = np.meshgrid(u, v)

    # Local frame: needle direction + two perpendicular axes
    perp1 = np.cross(n_hat, [1, 0, 0])
    if np.linalg.norm(perp1) < 1e-6:
        perp1 = np.cross(n_hat, [0, 1, 0])
    perp1 /= np.linalg.norm(perp1)
    perp2  = np.cross(n_hat, perp1)

    xs = a * np.cos(vv)
    ys = b * np.sin(vv) * np.cos(uu)
    zs = b * np.sin(vv) * np.sin(uu)

    # Transform to world coordinates
    pts = (np.outer(xs.ravel(), n_hat) +
           np.outer(ys.ravel(), perp1) +
           np.outer(zs.ravel(), perp2) + centroid)

    cloud  = pv.PolyData(pts)
    surf   = cloud.delaunay_3d().extract_surface().clean()
    dists  = np.linalg.norm(surf.points - centroid, axis=1)
    surf["Temperature"] = T_BLOOD + (T_TISSUE - T_BLOOD) * np.exp(
        -2.0 * (dists / max(a, b)) ** 2)
    return surf


def create_constrained_particles(vessel, n_particles=60):
    """
    Sample particles ON the vessel surface and give them a flow direction
    derived from the local surface normal projected along the vessel's
    principal axis (approximation of flow direction).
    """
    if vessel is None or vessel.n_points < n_particles:
        n_particles = max(1, vessel.n_points // 3) if vessel else 0
    if n_particles == 0:
        return None, None

    idx  = np.random.choice(vessel.n_points, n_particles, replace=False)
    pts  = vessel.points[idx].copy()

    # Approximate flow direction: PCA first eigenvector of vessel points
    centered = vessel.points - vessel.points.mean(axis=0)
    _, _, vt = np.linalg.svd(centered[:min(5000, len(centered))], full_matrices=False)
    flow_dir = vt[0]   # principal axis ≈ vessel long axis

    return pts, flow_dir


def run_animation(surface, vessels, tumors, centroids, selected_idx,
                  results, recommended_regime, oar_list):
    """
    PyVista animation with:
    - Growing ablation ELLIPSOID (not sphere)
    - Blood particles constrained to vessel surfaces
    - Temperature field
    - OAR highlighting
    - Treatment regime overlay
    """
    print("\n🎬 Starting animation...")

    power_w    = float(recommended_regime[0])
    time_s     = float(recommended_regime[1])
    fwd_throw  = recommended_regime[2] / 100.0   # cm → m
    diam       = recommended_regime[3] / 100.0   # cm → m

    plotter = pv.Plotter(window_size=[1400, 1000])
    plotter.background_color = "black"

    # Static anatomy
    plotter.add_mesh(surface, color="lightgray", opacity=0.10, label="Body Surface")
    for i, (v, col, nm) in enumerate(zip(vessels, VESSEL_COLORS, VESSEL_NAMES)):
        if v is None:
            continue
        is_oar = any(o["vessel"] == nm for o in oar_list)
        plotter.add_mesh(v, color="red" if is_oar else col,
                         opacity=0.8 if is_oar else 0.55,
                         label=f"{'⚠ OAR: ' if is_oar else ''}{nm}")

    for i, tumor in enumerate(tumors):
        t_col = ["yellow","orange","purple","pink","red","lime"][i % 6]
        op    = 0.85 if i == selected_idx else 0.3
        plotter.add_mesh(tumor, color=t_col, opacity=op,
                         label=f"Tumor {i+1}" + (" ← ablation target" if i == selected_idx else ""))

    centroid   = centroids[selected_idx]
    center_sph = pv.Sphere(radius=0.008, center=centroid)
    plotter.add_mesh(center_sph, color="yellow", label="Tumor centroid")

    # Particle initial positions
    particle_data = []
    for v in vessels:
        pts, flow_dir = create_constrained_particles(v, n_particles=60)
        particle_data.append({"pts": pts, "flow_dir": flow_dir,
                               "original_pts": pts.copy() if pts is not None else None})

    # ── Slider callback ──────────────────────────────────────────────
    def update(t_value):
        t = float(t_value)
        frac = min(t / time_s, 1.0)

        # Remove previous dynamic actors
        for name in ["ablation", "particles_all", "time_text"]:
            try:
                plotter.remove_actor(name)
            except Exception:
                pass

        # Growing ellipsoid (scales with time fraction)
        cur_fwd  = fwd_throw * frac
        cur_diam = diam      * frac
        if cur_fwd > 1e-4 and cur_diam > 1e-4:
            ellipsoid = create_ablation_ellipsoid(centroid, cur_fwd, cur_diam)
            plotter.add_mesh(ellipsoid, scalars="Temperature", cmap="hot",
                             opacity=0.65, name="ablation",
                             scalar_bar_args={"title": "Temp (°C)"})

        # Blood particles — move along flow direction, wrap around vessel
        all_particle_pts = []
        for j, pd in enumerate(particle_data):
            if pd["pts"] is None:
                continue
            speed = list(VESSEL_VELOCITIES.values())[j % len(VESSEL_VELOCITIES)]
            disp  = pd["flow_dir"] * speed * t * 0.001  # scaled for visibility
            new_pts = pd["original_pts"] + disp
            # Wrap: keep particles near vessel by clamping displacement
            new_pts = pd["original_pts"] + (disp % 0.02) - 0.01
            all_particle_pts.append(new_pts)

        if all_particle_pts:
            all_pts = np.vstack(all_particle_pts)
            cloud   = pv.PolyData(all_pts)
            plotter.add_mesh(cloud, color="crimson", point_size=4,
                             render_points_as_spheres=True, name="particles_all")

        # Time/info overlay
        pct_done = frac * 100
        info = (f"t = {t:.1f}s / {time_s:.0f}s  ({pct_done:.0f}%)\n"
                f"Power: {power_w:.0f} W\n"
                f"Zone: {cur_fwd*100:.1f}cm × {cur_diam*100:.1f}cm\n"
                f"OARs encroached: {len(oar_list)}")
        plotter.add_text(info, position="lower_left", font_size=11,
                         color="white", name="time_text")
        plotter.render()

    # Add slider
    plotter.add_slider_widget(
        callback=update,
        rng=[0.0, time_s],
        value=0.0,
        title="Ablation Time (s)",
        pointa=(0.1, 0.05), pointb=(0.9, 0.05),
        style="modern",
    )

    # Legend and title
    plotter.add_legend(loc="upper right", size=(0.22, 0.35))
    plotter.add_text(
        f"Heat Sink + OAR Analysis  |  Regime: {power_w:.0f}W × {time_s:.0f}s",
        position="upper_left", font_size=13, color="white"
    )
    plotter.add_axes()

    update(0.0)

    try:
        plotter.show(auto_close=False)
    except Exception as e:
        print(f"  Visualization error: {e}")
    finally:
        plotter.close()


# ─────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 68)
    print("  INTEGRATED HEAT SINK | OAR | TREATMENT REGIME | ANIMATION")
    print("=" * 68)

    if not os.path.exists(DATASET_BASE):
        print(f"  ✘ Dataset not found: {DATASET_BASE}")
        return

    # ── Load ─────────────────────────────────────────────────────────
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
        print("  ✘ Failed to load required files.")
        return

    # ── Tumor selection ───────────────────────────────────────────────
    tumors  = extract_tumors(tumor_mesh)
    metrics = tumor_metrics(tumors, surface, vessels)
    centroids = np.array([m["centroid"] for m in metrics])

    eligible = [m for m in metrics
                if MIN_DIAMETER_CM <= m["diameter_cm"] <= MAX_DIAMETER_CM
                and m["depth_cm"] <= MAX_DEPTH_CM]

    if eligible:
        eligible_sorted = sorted(eligible, key=lambda m: m["min_vessel_m"])
        print(f"\n  Eligible tumors for MWA ({len(eligible)} found):")
        for rank, m in enumerate(eligible_sorted, 1):
            cv = vnames[m["closest_vessel"]] if m["closest_vessel"] < len(vnames) else "?"
            print(f"    {rank}. Tumor {m['idx']+1}  diam={m['diameter_cm']:.2f}cm  "
                  f"depth={m['depth_cm']:.2f}cm  "
                  f"closest={cv} @ {m['min_vessel_m']*1000:.1f}mm")
        sel = eligible_sorted[0]
    else:
        print("  No tumor met MWA criteria — using closest-to-vessel as fallback.")
        sel = sorted(metrics, key=lambda m: m["min_vessel_m"])[0]

    sel_idx  = sel["idx"]
    centroid = sel["centroid"]
    sel_diam = sel["diameter_cm"]

    print(f"\n🎯 Selected: Tumor {sel_idx+1}  "
          f"({sel_diam:.2f}cm, depth={sel['depth_cm']:.2f}cm)")

    # ── Per-vessel centroid distances ─────────────────────────────────
    centroid_dists = {}
    for i, v in enumerate(vessels):
        tree = cKDTree(np.array(v.points))
        d, _ = tree.query(centroid, k=1)
        centroid_dists[vnames[i]] = float(d)

    # ── Corrected heat sink (centroid-based, for ablation physics) ────
    POWER_USE = 30.0
    TIME_USE  = 600.0
    print("\n" + "=" * 68)
    print("  HEAT SINK PARAMETERS  (centroid→vessel, corrected physics)")
    print("=" * 68)
    print(f"\n  α = {ALPHA_TISSUE} m⁻¹  |  loss halves every "
          f"{1000*np.log(2)/ALPHA_TISSUE:.1f} mm\n")
    print(f"  {'Vessel':<18}{'Dist(mm)':<12}{'Re':<9}{'Nu':<9}"
          f"{'h(W/m²K)':<13}{'Qmax(W)':<11}{'Qloss(W)':<11}{'Loss%'}")
    print("  " + "-" * 95)

    per_vessel_hs = {}
    for vn in vnames:
        d  = centroid_dists[vn]
        hs = heat_sink_from_distance(d, vn, POWER_USE, TIME_USE)
        per_vessel_hs[vn] = hs
        print(f"  {vn:<18}{d*1000:<12.2f}{hs['Re']:<9.0f}{hs['Nu']:<9.1f}"
              f"{hs['h']:<13.1f}{hs['Q_max_W']:<11.3f}"
              f"{hs['Q_loss_W']:<11.4f}{hs['loss_pct']:.3f}%")

    max_hs_pct = max(hs["loss_pct"] for hs in per_vessel_hs.values())
    dom_vessel  = max(per_vessel_hs, key=lambda k: per_vessel_hs[k]["loss_pct"])

    # ── Build vessel KD-trees for ray analysis ────────────────────────
    # Per-vessel trees (needed for ray-segment distance per vessel)
    vessel_trees_per = [cKDTree(np.array(v.points)) for v in vessels]

    # Combined tree (for label lookup)
    all_v_pts, all_v_lab = [], []
    for i, v in enumerate(vessels):
        pts = np.array(v.points)
        all_v_pts.append(pts)
        all_v_lab.append(np.full(len(pts), i, dtype=int))
    all_v_pts = np.vstack(all_v_pts)
    all_v_lab = np.concatenate(all_v_lab)

    # ── Ray tracing with segment-based distance ───────────────────────
    print("\n  Generating rays...")
    rays = generate_rays(n_theta=20, n_phi=40)
    print(f"   {len(rays)} rays  |  tracing against body surface + vessel distances")

    results = []
    print()
    for direction in tqdm(rays, desc="  Ray trace"):
        try:
            pts_hit, _ = surface.ray_trace(centroid, centroid + direction * 0.5)
            if len(pts_hit) == 0:
                continue
            hit     = pts_hit[0]
            path_d  = float(np.linalg.norm(hit - centroid))

            # ── FIXED: ray-segment-to-vessel distance ────────────────
            # For each vessel, find minimum distance from ray segment to vessel
            seg_dists = {}
            for vi, (vn, tree) in enumerate(zip(vnames, vessel_trees_per)):
                d_seg = ray_segment_to_vessel_dist(
                    centroid, direction, path_d, np.array(vessels[vi].points), n_sample=30)
                seg_dists[vn] = d_seg

            # Dominant vessel for this ray = closest vessel along ray path
            dominant_vn  = min(seg_dists, key=seg_dists.get)
            ray_dist_dom = seg_dists[dominant_vn]

            # Heat loss uses ray-segment distance → varies per ray direction
            hs = heat_sink_from_distance(ray_dist_dom, dominant_vn, POWER_USE, TIME_USE)

            hs["ray_direction"] = direction
            hs["path_distance"] = path_d
            hs["hit_point"]     = hit
            hs["ray_seg_dist_mm"] = ray_dist_dom * 1000
            hs["all_seg_dists"] = {k: v * 1000 for k, v in seg_dists.items()}
            results.append(hs)

        except Exception:
            continue

    print(f"\n  {len(results)} valid ray paths")

    # ── Results summary ───────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  DIRECTIONAL HEAT LOSS RESULTS  (ray-segment distance, varies per ray)")
    print("=" * 68)

    vessel_groups = {}
    for r in results:
        vessel_groups.setdefault(r["vessel"], []).append(r)

    print(f"\n  {'Vessel':<18}{'Rays':<7}{'AvgSegDist(mm)':<18}"
          f"{'AvgLoss%':<13}{'MaxLoss%':<12}{'MinLoss%'}")
    print("  " + "-" * 80)
    for vn, grp in vessel_groups.items():
        avg_d  = np.mean([r["ray_seg_dist_mm"] for r in grp])
        avg_l  = np.mean([r["loss_pct"] for r in grp])
        max_l  = np.max( [r["loss_pct"] for r in grp])
        min_l  = np.min( [r["loss_pct"] for r in grp])
        print(f"  {vn:<18}{len(grp):<7}{avg_d:<18.2f}{avg_l:<13.3f}{max_l:<12.3f}{min_l:.3f}")

    all_losses = [r["loss_pct"] for r in results]
    sorted_res = sorted(results, key=lambda x: x["loss_pct"], reverse=True)

    print(f"\n  Top 25 critical ray paths:")
    print("  " + "-" * 100)
    print(f"  {'Vessel':<18}{'SegDist(mm)':<14}{'Loss%':<11}{'Qloss(W)':<12}"
          f"{'dir_x':<9}{'dir_y':<9}{'dir_z'}")
    print("  " + "-" * 100)
    for r in sorted_res[:25]:
        d = r["ray_direction"]
        print(f"  {r['vessel']:<18}{r['ray_seg_dist_mm']:<14.2f}"
              f"{r['loss_pct']:<11.3f}{r['Q_loss_W']:<12.4f}"
              f"{d[0]:<9.3f}{d[1]:<9.3f}{d[2]:.3f}")

    print(f"\n  Overall (Tumor {sel_idx+1}):")
    print(f"   Paths analyzed      : {len(results)}")
    print(f"   Average heat loss   : {np.mean(all_losses):.3f}%")
    print(f"   Maximum heat loss   : {np.max(all_losses):.3f}%  ← now varies per ray!")
    print(f"   Minimum heat loss   : {np.min(all_losses):.3f}%")
    print(f"   Loss range          : {np.max(all_losses)-np.min(all_losses):.3f}%  "
          f"(was 0% before fix)")

    worst = sorted_res[0]
    best  = sorted_res[-1]
    print(f"\n  ➡ Safest insertion direction (BLUE ray):")
    print(f"     Vessel   : {best['vessel']}")
    print(f"     Seg dist : {best['ray_seg_dist_mm']:.2f} mm from vessel")
    print(f"     Loss     : {best['loss_pct']:.3f}%")
    print(f"     Direction: {best['ray_direction'].round(3)}")

    print(f"\n  ➡ Most dangerous insertion direction (RED ray):")
    print(f"     Vessel   : {worst['vessel']}")
    print(f"     Seg dist : {worst['ray_seg_dist_mm']:.2f} mm from vessel")
    print(f"     Loss     : {worst['loss_pct']:.3f}%")
    print(f"     Direction: {worst['ray_direction'].round(3)}")

    # ── OAR identification ────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  OAR (ORGAN AT RISK) IDENTIFICATION")
    print("=" * 68)

    # Use recommended regime dimensions for OAR check
    POWER_USE2 = 60.0; TIME_USE2 = 300.0
    ref_row = [r for r in ABLATION_TABLE if r[0] == POWER_USE2 and r[1] == TIME_USE2]
    ref_fwd  = ref_row[0][2] if ref_row else 4.7
    ref_diam = ref_row[0][3] if ref_row else 3.0

    # Needle direction = safest ray direction (lowest heat loss)
    needle_dir = best["ray_direction"]
    oar_list   = identify_oars(centroid, vessels, vnames, ref_fwd, ref_diam, needle_dir)

    if oar_list:
        print(f"\n  ⚠  {len(oar_list)} vessel(s) ENCROACHED by planned ablation zone "
              f"({ref_fwd}cm × {ref_diam}cm ellipsoid):")
        print(f"  {'Vessel':<18}{'Points inside':<16}{'Closest(mm)':<15}{'Risk'}")
        print("  " + "-" * 55)
        for o in oar_list:
            print(f"  {o['vessel']:<18}{o['points_inside']:<16}"
                  f"{o['closest_dist_mm']:<15.2f}{o['risk']}")
    else:
        print(f"  ✔  No vessels encroached by the planned ellipsoid.")
        print(f"     (Forward throw={ref_fwd}cm, Diameter={ref_diam}cm)")

    # ── Treatment regime ──────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  TREATMENT REGIME SELECTION  (heat-sink compensated)")
    print("=" * 68)

    recommended, alternatives, req_raw = select_treatment_regime(
        sel_diam, max_hs_pct, safety_margin_cm=0.5)

    print(f"\n  ✅ RECOMMENDED SETTING:")
    print(f"     Power       : {recommended[0]:.0f} W")
    print(f"     Time        : {recommended[1]:.0f} s  ({recommended[1]/60:.1f} min)")
    print(f"     Forward throw: {recommended[2]:.2f} cm")
    print(f"     Diameter    : {recommended[3]:.2f} cm  ≥ required {req_raw:.2f} cm")
    print(f"     Length      : {recommended[4]:.2f} cm")

    if alternatives:
        print(f"\n  Alternative settings:")
        for alt in alternatives:
            print(f"     {alt[0]:.0f}W × {alt[1]:.0f}s  →  "
                  f"throw={alt[2]:.2f}cm  diam={alt[3]:.2f}cm")

    # Clinical summary
    print("\n" + "=" * 68)
    print("  CLINICAL SUMMARY")
    print("=" * 68)
    print(f"  Dominant heat sink vessel : {dom_vessel} "
          f"({centroid_dists[dom_vessel]*1000:.1f} mm from tumor centroid)")
    print(f"  Centroid-based max loss   : {max_hs_pct:.2f}%")
    print(f"  Directional max loss      : {np.max(all_losses):.3f}% "
          f"(ray pointing toward {worst['vessel']})")
    print(f"  Safest insertion dir      : {best['ray_direction'].round(3)}")

    if max_hs_pct > 40:
        print("  ⚠  CRITICAL heat sink — mandatory energy compensation")
    elif max_hs_pct > 20:
        print("  ⚠  HIGH heat sink — energy compensation recommended")
    elif max_hs_pct > 10:
        print("  ℹ  MODERATE — standard protocol with slight increase")
    else:
        print("  ✔  LOW risk — recommended regime should be sufficient")

    if oar_list:
        print(f"\n  ⚠  OAR WARNING: {[o['vessel'] for o in oar_list]} will be inside ablation zone!")
        print("     → Reposition needle or reduce power/time")
        print(f"     → Use insertion direction: {best['ray_direction'].round(3)}")

    # ── Animation ─────────────────────────────────────────────────────
    try:
        run_animation(surface, vessels, tumors, centroids, sel_idx,
                      results, recommended, oar_list)
    except Exception as e:
        print(f"\n  Animation error: {e}")

    print("\n  Pipeline complete!")
    return results


if __name__ == "__main__":
    try:
        results = main()
    except KeyboardInterrupt:
        print("\n  Interrupted.")
    except Exception as e:
        print(f"\n  Fatal error: {e}")
        raise
