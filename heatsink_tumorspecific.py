#!/usr/bin/env python3
"""
Multi-Tumor Heat Sink Effect Analysis for Liver Ablation
========================================================

- Single tumor VTK contains multiple tumors (connected components)
- Each tumor's centroid, diameter, depth, and vessel distances are computed
- Tumors are filtered for microwave ablation eligibility:
    * Diameter between 3–5 cm
    * Depth from body surface <= 26 cm   (change this if needed)
- Among eligible tumors, the one closest to any vessel is selected
- Heat sink effect is analyzed from this tumor using ray tracing
- Full statistics (per-vessel summary, top 25 critical paths, overall stats) are printed
- 3D visualization shows all tumors and rays from the selected tumor,
  colored blue→red by heat-loss intensity
"""

import os
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# --------------------------------------------------------
# Configuration and File Paths
# --------------------------------------------------------

DATASET_BASE = r"C:\Users\z005562w\OneDrive - Siemens Healthineers\Veda\Project\siemens project\3d simulation\Nunna Algo\Nunna Algo\Dataset"
PORTALVENOUS_DIR = os.path.join(DATASET_BASE, "portalvenous")

TUMOR_VTK = os.path.join(DATASET_BASE, "908ac523data00007_leasion_out.vtk")
SURFACE_VTK = os.path.join(DATASET_BASE, "908ac523data00013_skin_out.vtk")

VESSEL_VTK_LIST = [
    os.path.join(PORTALVENOUS_DIR, "908ac52300001.vtk"),    # portal vein
    os.path.join(PORTALVENOUS_DIR, "908ac52300002.vtk"),    # hepatic vein
    os.path.join(PORTALVENOUS_DIR, "908ac52300003.vtk"),    # aorta
    os.path.join(PORTALVENOUS_DIR, "908ac52300004.vtk"),    # IVC
    os.path.join(PORTALVENOUS_DIR, "908ac52300005.vtk")     # hepatic artery
]

VESSEL_NAMES = ["portal_vein", "hepatic_vein", "aorta", "ivc", "hepatic_artery"]
TUMOR_COLORS = ["yellow", "orange", "purple", "pink", "red", "lime"]

# --------------------------------------------------------
# Physical Constants and Parameters
# --------------------------------------------------------

# Blood properties
RHO_B = 1060.0         # blood density (kg/m³)
MU_B = 3.5e-3          # blood dynamic viscosity (Pa·s)
C_B = 3700.0           # blood specific heat (J/kg·K)
K_B = 0.52             # blood thermal conductivity (W/m·K)
T_BLOOD = 37.0         # blood core temperature (°C)
T_TISSUE = 90.0        # tissue ablation temperature (°C)

# Ablation settings
POWER = 30.0           # microwave power (W)
ABLATION_TIME = 60.0   # ablation time (s)
ENERGY_INPUT = POWER * ABLATION_TIME  # total energy (J)

# Tumor selection criteria for MWA
MIN_DIAMETER_CM = 3.0
MAX_DIAMETER_CM = 5.0
MAX_DEPTH_CM = 26.0    # you can reduce to 2.6 if you meant 26 mm

# Vessel parameters (diameters in m, velocities in m/s)
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

# --------------------------------------------------------
# Utility Functions
# --------------------------------------------------------

def check_file_exists(filepath):
    """Check if file exists and print status."""
    if os.path.exists(filepath):
        print(f" Found: {os.path.basename(filepath)}")
        return True
    else:
        print(f" Missing: {filepath}")
        return False

def load_vtk_mesh(filepath):
    """Load VTK mesh with error handling."""
    try:
        if not check_file_exists(filepath):
            return None
        mesh = pv.read(filepath)
        print(f"   Loaded mesh with {mesh.n_points} points and {mesh.n_cells} cells")
        return mesh
    except Exception as e:
        print(f" Error loading {filepath}: {e}")
        return None

def rescale_if_needed(mesh):
    """Rescale mesh from mm to m if needed."""
    if mesh is None:
        return None
    pts = np.array(mesh.points)
    if np.max(np.abs(pts)) > 1000:
        print(f"   Rescaling mesh from mm to m...")
        mesh.points = pts / 1000.0
    return mesh

# --------------------------------------------------------
# Tumor Handling (multiple lesions in one mesh)
# --------------------------------------------------------

def extract_tumors_from_mesh(tumor_mesh):
    """
    Split the input tumor mesh into connected components (individual tumors).
    Returns a list of PyVista meshes, one per tumor.
    """
    print("\n🔍 Extracting individual tumors from combined tumor mesh...")
    connected = tumor_mesh.connectivity()
    tumors = connected.split_bodies()
    print(f"   Detected {len(tumors)} separate tumors")
    return tumors

def compute_tumor_metrics(tumors, surface, vessels):
    """
    For each tumor:
      - centroid
      - approximate diameter (bounding box max extent, in cm)
      - depth from body surface (shortest distance, in cm)
      - distance from centroid to each vessel (m)
      - min distance and index of closest vessel
    Returns list of dicts.
    """
    surface_pts = np.array(surface.points)
    surface_tree = cKDTree(surface_pts)

    vessel_trees = []
    vessel_pts_per_vessel = []
    for v in vessels:
        pts = np.array(v.points)
        vessel_pts_per_vessel.append(pts)
        vessel_trees.append(cKDTree(pts))

    metrics = []

    for i, tumor in enumerate(tumors):
        c = np.array(tumor.center)

        # Diameter from bounding box
        xmin, xmax, ymin, ymax, zmin, zmax = tumor.bounds
        dx = xmax - xmin
        dy = ymax - ymin
        dz = zmax - zmin
        diameter_m = max(dx, dy, dz)
        diameter_cm = diameter_m * 100.0

        # Depth from surface (shortest distance)
        depth_m, _ = surface_tree.query(c, k=1)
        depth_cm = depth_m * 100.0

        # Distance to each vessel
        vessel_distances_m = []
        for tree in vessel_trees:
            d, _ = tree.query(c, k=1)
            vessel_distances_m.append(float(d))

        min_vessel_distance_m = min(vessel_distances_m)
        closest_vessel_idx = int(np.argmin(vessel_distances_m))

        metrics.append({
            "tumor_index": i,
            "centroid": c,
            "diameter_cm": diameter_cm,
            "depth_cm": depth_cm,
            "vessel_distances_m": vessel_distances_m,
            "min_vessel_distance_m": min_vessel_distance_m,
            "closest_vessel_idx": closest_vessel_idx
        })

    # Print per-tumor metrics
    print("\n Tumor geometric and vessel distance metrics:")
    header = f"{'Tumor':<8}{'Diameter (cm)':<15}{'Depth (cm)':<12}"
    for name in VESSEL_NAMES:
        header += f"{name + ' dist (mm)':<18}"
    print(header)
    print("-" * len(header))
    for m in metrics:
        line = f"{m['tumor_index']+1:<8}{m['diameter_cm']:<15.2f}{m['depth_cm']:<12.2f}"
        for d in m["vessel_distances_m"]:
            line += f"{d*1000:<18.2f}"
        print(line)

    return metrics

# --------------------------------------------------------
# Heat Sink Analysis Functions
# --------------------------------------------------------

def compute_heat_sink_parameters(hit_point, vessel_points, vessel_labels, vessel_names):
    """
    Compute heat sink parameters for a given point.
    """
    vessel_tree = cKDTree(vessel_points)
    dist, idx = vessel_tree.query(hit_point, k=1)
    if np.isinf(dist):
        return None
    
    vessel_idx = vessel_labels[idx]
    vessel_name = vessel_names[vessel_idx]
    
    D = VESSEL_DIAMETERS[vessel_name]    # diameter (m)
    u = VESSEL_VELOCITIES[vessel_name]   # velocity (m/s)
    
    Re = (RHO_B * u * D) / MU_B
    Pr = (C_B * MU_B) / K_B
    
    Nu = 0.023 * (Re**0.8) * (Pr**0.4) if Re >= 2300 else 4.36
    h = (Nu * K_B) / D
    
    L_seg = 0.01
    A = np.pi * D * L_seg
    
    dT = max(T_TISSUE - T_BLOOD, 0.1)
    
    Q_loss = h * A * dT
    E_loss = Q_loss * ABLATION_TIME
    pct_loss = 100.0 * E_loss / ENERGY_INPUT
    
    return {
        "vessel_name": vessel_name,
        "distance_to_vessel_m": float(dist),
        "vessel_diameter_mm": float(D * 1000),
        "velocity_m_s": float(u),
        "reynolds_number": float(Re),
        "prandtl_number": float(Pr),
        "nusselt_number": float(Nu),
        "heat_transfer_coeff": float(h),
        "heat_loss_power_W": float(Q_loss),
        "energy_loss_J": float(E_loss),
        "energy_loss_percent": float(pct_loss)
    }

def generate_3d_rays(centroid, n_theta=20, n_phi=40):
    """Generate 3D rays for path planning analysis."""
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi)
    
    rays = []
    for t in theta:
        for p in phi:
            direction = np.array([
                np.sin(t) * np.cos(p),
                np.sin(t) * np.sin(p),
                np.cos(t)
            ])
            rays.append(direction)
    
    return np.array(rays)

# --------------------------------------------------------
# Visualization
# --------------------------------------------------------

def create_visualization_multi(surface, vessels, tumors, centroids, selected_idx, results):
    """Create 3D visualization of all tumors, vessels, and rays for selected tumor."""
    
    print(f"\n Creating 3D visualization (multi-tumor)...")
    
    plotter = pv.Plotter(window_size=[1200, 800])
    
    # Body surface
    plotter.add_mesh(surface, color='lightgray', opacity=0.15, label='Body Surface')
    
    # Vessels
    vessel_colors = ['magenta', 'cyan', 'blue', 'teal', 'orange']
    for i, vessel in enumerate(vessels):
        color = vessel_colors[i % len(vessel_colors)]
        plotter.add_mesh(vessel, color=color, opacity=0.6, label=VESSEL_NAMES[i])
    
    # Tumors and centroids
    for i, tumor in enumerate(tumors):
        t_color = TUMOR_COLORS[i % len(TUMOR_COLORS)]
        opacity = 0.9 if i == selected_idx else 0.4
        label = f"Tumor {i+1}" + (" (selected)" if i == selected_idx else "")
        plotter.add_mesh(tumor, color=t_color, opacity=opacity, label=label)
        
        centroid_sphere = pv.Sphere(radius=0.01, center=centroids[i])
        plotter.add_mesh(centroid_sphere,
                        color='yellow' if i == selected_idx else 'white',
                        label=f"Tumor {i+1} center")
    
    # Rays colored by heat loss (only for selected tumor)
    if results:
        heat_losses = np.array([r["energy_loss_percent"] for r in results])
        if len(heat_losses) > 0 and np.max(heat_losses) > np.min(heat_losses):
            norm_losses = (heat_losses - np.min(heat_losses)) / (np.max(heat_losses) - np.min(heat_losses))
            step = max(1, len(results)//60)
            for i in range(0, len(results), step):
                result = results[i]
                direction = result["ray_direction"]
                distance = result["path_distance"]
                color_val = norm_losses[i]
                
                start = centroids[selected_idx]
                end_point = start + direction * distance
                line = pv.Line(start, end_point)
                
                # Blue (low loss) → Red (high loss)
                color = [color_val, 0, 1 - color_val]
                plotter.add_mesh(line, color=color, line_width=2)
    
    plotter.add_legend(loc='upper right')
    plotter.add_text(
        "Heat Sink Effect Analysis\n(Rays from highest-priority tumor, colored by heat loss intensity)",
        position='upper_left',
        font_size=12
    )
    
    plotter.show_axes()
    plotter.show_grid()
    
    try:
        plotter.show(auto_close=False)
    except:
        print("   Display not available - skipping interactive visualization")

# --------------------------------------------------------
# Main Analysis Function
# --------------------------------------------------------

def analyze_heat_sink_effect_multi():
    """Full pipeline: multi-tumor metrics + selection + heat-sink analysis."""
    
    print("="*60)
    print(" HEAT SINK EFFECT ANALYSIS FOR LIVER ABLATION (MULTI-TUMOR)")
    print("="*60)
    
    if not os.path.exists(DATASET_BASE):
        print(f" Dataset directory not found: {DATASET_BASE}")
        return
    
    print(f"\n Dataset location: {DATASET_BASE}")
    print("\n Checking required files...")
    
    # Load meshes
    print("\n Loading VTK meshes...")
    tumor_mesh = load_vtk_mesh(TUMOR_VTK)
    surface = load_vtk_mesh(SURFACE_VTK)
    
    if tumor_mesh is None or surface is None:
        print(" Failed to load required meshes. Check file paths.")
        return
    
    # Load vessel meshes
    vessels = []
    valid_vessel_names = []
    for i, vessel_path in enumerate(VESSEL_VTK_LIST):
        vessel = load_vtk_mesh(vessel_path)
        if vessel is not None:
            vessels.append(vessel)
            valid_vessel_names.append(VESSEL_NAMES[i])
    
    if not vessels:
        print(" No vessel meshes loaded successfully.")
        return
    
    print(f"\n Successfully loaded {len(vessels)} vessel meshes")
    
    # Rescale all meshes
    tumor_mesh = rescale_if_needed(tumor_mesh)
    surface = rescale_if_needed(surface)
    vessels = [rescale_if_needed(v) for v in vessels]
    
    # Split tumor mesh into separate tumors
    tumors = extract_tumors_from_mesh(tumor_mesh)
    
    # Compute geometric + vessel metrics per tumor
    tumor_metrics = compute_tumor_metrics(tumors, surface, vessels)
    centroids = np.array([m["centroid"] for m in tumor_metrics])
    
    # Filter tumors by MWA eligibility
    eligible = [
        m for m in tumor_metrics
        if (MIN_DIAMETER_CM <= m["diameter_cm"] <= MAX_DIAMETER_CM)
        and (m["depth_cm"] <= MAX_DEPTH_CM)
    ]
    
    if eligible:
        # Sort eligible by min vessel distance (closer → higher priority)
        eligible_sorted = sorted(eligible, key=lambda m: m["min_vessel_distance_m"])
        print("\n Eligible tumors for MWA based on diameter and depth:")
        print(f"{'Priority':<10}{'Tumor':<8}{'Diameter (cm)':<15}{'Depth (cm)':<12}{'Min vessel dist (mm)':<22}")
        print("-" * 70)
        for rank, m in enumerate(eligible_sorted, start=1):
            print(f"{rank:<10}{m['tumor_index']+1:<8}{m['diameter_cm']:<15.2f}{m['depth_cm']:<12.2f}{m['min_vessel_distance_m']*1000:<22.2f}")
        selected_meta = eligible_sorted[0]
        print(f"\n🔎 Selected highest-priority tumor for heat sink analysis: Tumor {selected_meta['tumor_index']+1}")
    else:
        # No tumors meet the criteria → fall back to closest-to-vessel
        print("\n No tumors met the MWA criteria (diameter & depth).")
        print("   Selecting tumor closest to any vessel as fallback.")
        selected_meta = sorted(tumor_metrics, key=lambda m: m["min_vessel_distance_m"])[0]
    
    selected_idx = selected_meta["tumor_index"]
    centroid = selected_meta["centroid"]
    
    print(f"   Tumor {selected_idx+1} centroid: ({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})")
    print(f"   Diameter: {selected_meta['diameter_cm']:.2f} cm")
    print(f"   Depth from surface: {selected_meta['depth_cm']:.2f} cm")
    closest_vessel_name = VESSEL_NAMES[selected_meta['closest_vessel_idx']]
    print(f"   Closest vessel: {closest_vessel_name} "
          f"({selected_meta['min_vessel_distance_m']*1000:.2f} mm from centroid)")
    
    # Combine vessel points and create labels for heat-sink calculation
    print("\n Combining vessel data...")
    vessel_points = []
    vessel_labels = []
    for i, vessel in enumerate(vessels):
        pts = np.array(vessel.points)
        vessel_points.append(pts)
        vessel_labels.append(np.full(len(pts), i))
    vessel_points = np.vstack(vessel_points)
    vessel_labels = np.concatenate(vessel_labels)
    print(f"   Combined {len(vessel_points)} vessel surface points")
    
    # Generate rays from selected tumor centroid
    print("\n Generating 3D rays for path planning...")
    rays = generate_3d_rays(centroid, n_theta=20, n_phi=40)
    print(f"   Generated {len(rays)} rays")
    
    # Analyze each ray
    print("\n Analyzing heat sink effects...")
    results = []
    
    for direction in tqdm(rays, desc="Processing rays"):
        try:
            points, indices = surface.ray_trace(centroid, centroid + direction * 0.2)
            if len(points) == 0:
                continue
            
            hit_point = points[0]
            distance = np.linalg.norm(hit_point - centroid)
            
            heat_sink = compute_heat_sink_parameters(
                hit_point, vessel_points, vessel_labels, valid_vessel_names
            )
            
            if heat_sink is not None:
                heat_sink["ray_direction"] = direction
                heat_sink["path_distance"] = distance
                heat_sink["selected_tumor_index"] = int(selected_idx)
                results.append(heat_sink)
        except Exception:
            continue
    
    print(f"\n Analysis complete! Processed {len(results)} valid ray paths")
    
    if len(results) == 0:
        print(" No valid ray intersections found. Check mesh alignment.")
        return
    
    # ---------------- Summary Results ----------------
    print("\n" + "="*60)
    print("HEAT SINK ANALYSIS RESULTS")
    print("="*60)
    
    # Group results by vessel type
    vessel_results = {}
    for result in results:
        vessel_name = result["vessel_name"]
        vessel_results.setdefault(vessel_name, []).append(result)
    
    print(f"\n Summary by vessel type (for Tumor {selected_idx+1}):")
    print("-" * 80)
    print(f"{'Vessel':<15} {'Count':<8} {'Avg Distance':<12} {'Avg Heat Loss':<15} {'Max Heat Loss':<15}")
    print("-" * 80)
    
    for vessel_name, vessel_data in vessel_results.items():
        count = len(vessel_data)
        avg_dist = np.mean([r["distance_to_vessel_m"] for r in vessel_data]) * 1000  # mm
        avg_loss = np.mean([r["energy_loss_percent"] for r in vessel_data])
        max_loss = np.max([r["energy_loss_percent"] for r in vessel_data])
        print(f"{vessel_name:<15} {count:<8} {avg_dist:<12.2f} {avg_loss:<15.3f} {max_loss:<15.3f}")
    
    # Critical paths (TOP 25, not 10)
    print(f"\n Critical heat loss paths (top 25):")
    print("-" * 120)
    print(f"{'Vessel':<15} {'Distance (mm)':<15} {'Heat Loss (%)':<15} {'Power Loss (W)':<18}"
          f"{'dir_x':<10}{'dir_y':<10}{'dir_z':<10}")
    print("-" * 120)
    
    sorted_results = sorted(results, key=lambda x: x["energy_loss_percent"], reverse=True)
    
    for result in sorted_results[:25]:
        dist_mm = result["distance_to_vessel_m"] * 1000
        heat_loss_pct = result["energy_loss_percent"]
        power_loss = result["heat_loss_power_W"]
        vessel = result["vessel_name"]
        dir_vec = result["ray_direction"]
        print(f"{vessel:<15} {dist_mm:<15.2f} {heat_loss_pct:<15.3f} {power_loss:<18.3f}"
              f"{dir_vec[0]:<10.3f}{dir_vec[1]:<10.3f}{dir_vec[2]:<10.3f}")
    
    # Overall statistics
    all_heat_losses = [r["energy_loss_percent"] for r in results]
    all_distances = [r["distance_to_vessel_m"] * 1000 for r in results]
    
    print(f"\n Overall Statistics (Tumor {selected_idx+1}):")
    print(f"   Total analyzed paths: {len(results)}")
    print(f"   Average heat loss: {np.mean(all_heat_losses):.3f}%")
    print(f"   Maximum heat loss: {np.max(all_heat_losses):.3f}%")
    print(f"   Average distance to vessels: {np.mean(all_distances):.2f} mm")
    print(f"   Minimum distance to vessels: {np.min(all_distances):.2f} mm")
    
    # Direction of worst (red) and best (blue) rays
    worst_ray = sorted_results[0]
    best_ray = sorted_results[-1]
    print("\n➡ Directional summary of heat dissipation:")
    print(f"   Highest heat-loss ray (RED):")
    print(f"      Vessel: {worst_ray['vessel_name']}")
    print(f"      Distance to vessel: {worst_ray['distance_to_vessel_m']*1000:.2f} mm")
    print(f"      Heat loss: {worst_ray['energy_loss_percent']:.3f}% "
          f"({worst_ray['heat_loss_power_W']:.3f} W)")
    print(f"      Direction vector: {worst_ray['ray_direction']}")
    
    print(f"\n   Lowest heat-loss ray (BLUE):")
    print(f"      Vessel: {best_ray['vessel_name']}")
    print(f"      Distance to vessel: {best_ray['distance_to_vessel_m']*1000:.2f} mm")
    print(f"      Heat loss: {best_ray['energy_loss_percent']:.3f}% "
          f"({best_ray['heat_loss_power_W']:.3f} W)")
    print(f"      Direction vector: {best_ray['ray_direction']}")
    
    # Visualization
    try:
        create_visualization_multi(surface, vessels, tumors, centroids, selected_idx, results)
    except Exception as e:
        print(f"\n Visualization failed: {e}")
        print("   Analysis results are still valid.")
    
    print(f"\n Heat sink analysis completed successfully!")
    return results

# --------------------------------------------------------
# Main Execution
# --------------------------------------------------------

if __name__ == "__main__":
    try:
        results = analyze_heat_sink_effect_multi()
        if results:
            print(f"\n Analysis saved! You can access the results programmatically.")
            print(f"   Total paths analyzed: {len(results)}")
    except KeyboardInterrupt:
        print("\n\n Analysis interrupted by user.")
    except Exception as e:
        print(f"\n Error during analysis: {e}")
        print("Please check your dataset files and paths.")
