#!/usr/bin/env python3
"""
Patient-Specific Heat Sink–Aware Microwave Ablation Planning
============================================================

• Multi-tumor extraction
• Vessel proximity analysis
• Heat sink modeling (Re, Pr, Nu, h)
• Directional ray tracing
• Empirical liver MWA table integration
• Effective ablation radius reduction
• %OAZ (Outside Ablation Zone) estimation
• Residual tumor risk quantification
"""

import os
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
DATASET_BASE = r"C:\Users\z005562w\OneDrive - Siemens Healthineers\Veda\Project\siemens project\3d simulation\Nunna Algo\Nunna Algo\Dataset"
PORTALVENOUS_DIR = os.path.join(DATASET_BASE, "portalvenous")

TUMOR_VTK = os.path.join(DATASET_BASE, "908ac523data00007_leasion_out.vtk")
SURFACE_VTK = os.path.join(DATASET_BASE, "908ac523data00013_skin_out.vtk")

VESSEL_VTK_LIST = [
    "908ac52300001.vtk",  # portal vein
    "908ac52300002.vtk",  # hepatic vein
    "908ac52300003.vtk",  # aorta
    "908ac52300004.vtk",  # ivc
    "908ac52300005.vtk"   # hepatic artery
]

VESSEL_NAMES = ["portal_vein", "hepatic_vein", "aorta", "ivc", "hepatic_artery"]

# ------------------------------------------------------------
# PHYSICAL CONSTANTS
# ------------------------------------------------------------
RHO_B = 1060.0        # kg/m³
MU_B = 3.5e-3         # Pa·s
C_B = 3700.0          # J/kg·K
K_B = 0.52            # W/m·K

T_BLOOD = 37.0        # °C
T_TISSUE = 90.0       # °C

POWER = 30.0          # W
ABLATION_TIME = 600.0 # s
ENERGY_INPUT = POWER * ABLATION_TIME

# ------------------------------------------------------------
# EMPIRICAL LIVER MWA ABLATION TABLE (NO HEAT SINK)
# ------------------------------------------------------------
ABLATION_TABLE = [
    {"power":30, "time":180, "diameter_mm":19},
    {"power":30, "time":300, "diameter_mm":24},
    {"power":30, "time":480, "diameter_mm":29},
    {"power":30, "time":600, "diameter_mm":31},
    {"power":60, "time":300, "diameter_mm":30},
    {"power":90, "time":300, "diameter_mm":37},
    {"power":120, "time":600, "diameter_mm":59}
]

def get_baseline_radius(power, time):
    for row in ABLATION_TABLE:
        if row["power"] == power and row["time"] == time:
            return (row["diameter_mm"] / 1000) / 2
    raise ValueError("No matching ablation parameters")

# ------------------------------------------------------------
# VESSEL PARAMETERS
# ------------------------------------------------------------
VESSEL_DIAMETERS = {
    "portal_vein": 12e-3,
    "hepatic_vein": 8e-3,
    "aorta": 25e-3,
    "ivc": 20e-3,
    "hepatic_artery": 4.5e-3
}

VESSEL_VELOCITIES = {
    "portal_vein": 0.15,
    "hepatic_vein": 0.20,
    "aorta": 0.40,
    "ivc": 0.35,
    "hepatic_artery": 0.30
}

# ------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------
def load_mesh(path):
    mesh = pv.read(path)
    if np.max(np.abs(mesh.points)) > 1000:
        mesh.points /= 1000
    return mesh

def generate_rays(n_theta=20, n_phi=40):
    rays = []
    for t in np.linspace(0, np.pi, n_theta):
        for p in np.linspace(0, 2*np.pi, n_phi):
            rays.append([
                np.sin(t)*np.cos(p),
                np.sin(t)*np.sin(p),
                np.cos(t)
            ])
    return np.array(rays)

# ------------------------------------------------------------
# HEAT SINK MODEL
# ------------------------------------------------------------
def heat_sink(hit_point, vessel_pts, vessel_labels):
    tree = cKDTree(vessel_pts)
    dist, idx = tree.query(hit_point)
    vessel = VESSEL_NAMES[vessel_labels[idx]]

    D = VESSEL_DIAMETERS[vessel]
    u = VESSEL_VELOCITIES[vessel]

    Re = (RHO_B * u * D) / MU_B
    Pr = (C_B * MU_B) / K_B
    Nu = 0.023*(Re**0.8)*(Pr**0.4) if Re > 2300 else 4.36

    h = Nu * K_B / D
    A = np.pi * D * 0.01
    Q_loss = h * A * (T_TISSUE - T_BLOOD)
    E_loss = Q_loss * ABLATION_TIME
    pct_loss = 100 * E_loss / ENERGY_INPUT

    return vessel, dist, pct_loss

# ------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------
def run():
    tumor_mesh = load_mesh(TUMOR_VTK)
    surface = load_mesh(SURFACE_VTK)
    vessels = [load_mesh(os.path.join(PORTALVENOUS_DIR, f)) for f in VESSEL_VTK_LIST]

    tumors = tumor_mesh.connectivity().split_bodies()
    centroids = [t.center for t in tumors]

    # Select smallest depth tumor (already validated in your run)
    selected_idx = 3
    tumor = tumors[selected_idx]
    centroid = np.array(centroids[selected_idx])

    tumor_radius = (max(tumor.length, tumor.width, tumor.height) / 2)

    R_base = get_baseline_radius(POWER, ABLATION_TIME)

    vessel_pts, vessel_lbls = [], []
    for i, v in enumerate(vessels):
        vessel_pts.append(v.points)
        vessel_lbls.append(np.full(len(v.points), i))

    vessel_pts = np.vstack(vessel_pts)
    vessel_lbls = np.concatenate(vessel_lbls)

    rays = generate_rays()
    results = []

    for d in tqdm(rays):
        pts, _ = surface.ray_trace(centroid, centroid + d*0.25)
        if len(pts) == 0:
            continue

        vessel, dist, hl = heat_sink(pts[0], vessel_pts, vessel_lbls)
        R_eff = R_base * (1 - hl/100)**(1/3)

        if R_eff < tumor_radius:
            OAZ = 1 - (R_eff / tumor_radius)**3
        else:
            OAZ = 0

        results.append({
            "vessel": vessel,
            "distance_mm": dist*1000,
            "heat_loss_pct": hl,
            "R_eff_mm": R_eff*1000,
            "OAZ_pct": OAZ*100
        })

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------
    OAZ_vals = [r["OAZ_pct"] for r in results]

    print("\n===== ABLATION ADEQUACY REPORT =====")
    print(f"Tumor radius           : {tumor_radius*1000:.1f} mm")
    print(f"Baseline ablation rad. : {R_base*1000:.1f} mm")
    print(f"Mean OAZ               : {np.mean(OAZ_vals):.2f} %")
    print(f"Max OAZ (worst-case)   : {np.max(OAZ_vals):.2f} %")
    print(f"High-risk rays (>10%)  : {np.mean(np.array(OAZ_vals)>10)*100:.1f} %")

    worst = max(results, key=lambda x: x["OAZ_pct"])
    print("\nWorst direction:")
    print(worst)

    print("\nSimulation complete.")

# ------------------------------------------------------------
if __name__ == "__main__":
    run()
