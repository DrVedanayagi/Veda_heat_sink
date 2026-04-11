#!/usr/bin/env python3
"""
Heat Sink Effect Analysis for Liver Ablation Treatment
======================================================

This script analyzes the heat sink effect during microwave ablation treatment
of liver tumors, considering the cooling effect of nearby blood vessels.

Author: Based on Dr. Vedanayagi Rajarajan's research
Date: November 2025
"""

import os
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --------------------------------------------------------
# Configuration and File Paths
# --------------------------------------------------------

# Dataset paths - all files should be in D:\HeatSink\Dataset
DATASET_BASE = r"C:\Users\z005562w\OneDrive - Siemens Healthineers\Veda\Project\siemens project\3d simulation\Nunna Algo\Nunna Algo\Dataset"
PORTALVENOUS_DIR = os.path.join(DATASET_BASE, "portalvenous")

# VTK file paths
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

# --------------------------------------------------------
# Physical Constants and Parameters
# --------------------------------------------------------

# Blood properties
RHO_B = 1060.0         # blood density (kg/m³)
MU_B = 3.5e-3         # blood dynamic viscosity (Pa·s)
C_B = 3700.0          # blood specific heat (J/kg·K)
K_B = 0.52            # blood thermal conductivity (W/m·K)
T_BLOOD = 37.0        # blood core temperature (°C)
T_TISSUE = 90.0       # tissue ablation temperature (°C)

# Ablation settings
POWER = 30.0          # microwave power (W)
ABLATION_TIME = 60.0  # ablation time (s)
ENERGY_INPUT = POWER * ABLATION_TIME  # total energy (J)

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
        print(f"✅ Found: {os.path.basename(filepath)}")
        return True
    else:
        print(f"❌ Missing: {filepath}")
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
        print(f"❌ Error loading {filepath}: {e}")
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
# Heat Sink Analysis Functions
# --------------------------------------------------------

def compute_heat_sink_parameters(hit_point, vessel_points, vessel_labels, vessel_names):
    """
    Compute heat sink parameters for a given point.
    
    Parameters:
    -----------
    hit_point : array
        3D coordinates of the ablation point
    vessel_points : array
        Combined vessel surface points
    vessel_labels : array
        Vessel type labels for each point
    vessel_names : list
        Names of vessel types
    
    Returns:
    --------
    dict : Heat sink analysis results
    """
    # Build KDTree for nearest neighbor search
    vessel_tree = cKDTree(vessel_points)
    dist, idx = vessel_tree.query(hit_point, k=1)
    
    if np.isinf(dist):
        return None
    
    vessel_idx = vessel_labels[idx]
    vessel_name = vessel_names[vessel_idx]
    
    # Get vessel parameters
    D = VESSEL_DIAMETERS[vessel_name]  # diameter (m)
    u = VESSEL_VELOCITIES[vessel_name]  # velocity (m/s)
    
    # Calculate dimensionless numbers
    Re = (RHO_B * u * D) / MU_B  # Reynolds number
    Pr = (C_B * MU_B) / K_B      # Prandtl number
    
    # Nusselt number (flow dependent)
    if Re >= 2300:  # Turbulent flow
        Nu = 0.023 * (Re**0.8) * (Pr**0.4)
    else:  # Laminar flow
        Nu = 4.36
    
    # Heat transfer coefficient
    h = (Nu * K_B) / D  # W/(m²·K)
    
    # Vessel segment geometry
    L_seg = 0.01  # segment length (m)
    A = np.pi * D * L_seg  # heat transfer area (m²)
    
    # Temperature difference
    dT = max(T_TISSUE - T_BLOOD, 0.1)
    
    # Heat loss calculations
    Q_loss = h * A * dT  # power loss (W)
    E_loss = Q_loss * ABLATION_TIME  # energy loss (J)
    pct_loss = 100.0 * E_loss / ENERGY_INPUT  # percentage
    
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

def generate_3d_rays(centroid, n_theta=30, n_phi=60):
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
# Main Analysis Functions
# --------------------------------------------------------

def analyze_heat_sink_effect():
    """Main function to analyze heat sink effects."""
    
    print("="*60)
    print("🔥 HEAT SINK EFFECT ANALYSIS FOR LIVER ABLATION")
    print("="*60)
    
    # Check dataset directory
    if not os.path.exists(DATASET_BASE):
        print(f"❌ Dataset directory not found: {DATASET_BASE}")
        print("Please ensure the dataset is in the correct location.")
        return
    
    print(f"\n📁 Dataset location: {DATASET_BASE}")
    print("\n🔍 Checking required files...")
    
    # Load meshes
    print("\n📥 Loading VTK meshes...")
    tumor = load_vtk_mesh(TUMOR_VTK)
    surface = load_vtk_mesh(SURFACE_VTK)
    
    if tumor is None or surface is None:
        print("❌ Failed to load required meshes. Check file paths.")
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
        print("❌ No vessel meshes loaded successfully.")
        return
    
    print(f"\n✅ Successfully loaded {len(vessels)} vessel meshes")
    
    # Rescale meshes if needed
    tumor = rescale_if_needed(tumor)
    surface = rescale_if_needed(surface)
    vessels = [rescale_if_needed(v) for v in vessels]
    
    # Calculate tumor centroid
    centroid = np.array(tumor.center)
    print(f"\n📍 Tumor centroid: ({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})")
    
    # Combine vessel points and create labels
    print("\n🔗 Combining vessel data...")
    vessel_points = []
    vessel_labels = []
    
    for i, vessel in enumerate(vessels):
        pts = np.array(vessel.points)
        vessel_points.append(pts)
        vessel_labels.append(np.full(len(pts), i))
    
    vessel_points = np.vstack(vessel_points)
    vessel_labels = np.concatenate(vessel_labels)
    
    print(f"   Combined {len(vessel_points)} vessel surface points")
    
    # Generate 3D rays for analysis
    print("\n🎯 Generating 3D rays for path planning...")
    rays = generate_3d_rays(centroid, n_theta=20, n_phi=40)  # Reduced for faster computation
    print(f"   Generated {len(rays)} rays")
    
    # Analyze each ray
    print("\n🧮 Analyzing heat sink effects...")
    results = []
    
    for i, direction in enumerate(tqdm(rays, desc="Processing rays")):
        # Cast ray to find intersection with surface
        try:
            points, indices = surface.ray_trace(centroid, centroid + direction * 0.2)
            if len(points) == 0:
                continue
            
            hit_point = points[0]
            distance = np.linalg.norm(hit_point - centroid)
            
            # Compute heat sink parameters
            heat_sink = compute_heat_sink_parameters(
                hit_point, vessel_points, vessel_labels, valid_vessel_names
            )
            
            if heat_sink is not None:
                heat_sink["ray_direction"] = direction
                heat_sink["path_distance"] = distance
                results.append(heat_sink)
                
        except Exception as e:
            # Skip failed rays
            continue
    
    print(f"\n✅ Analysis complete! Processed {len(results)} valid ray paths")
    
    if len(results) == 0:
        print("❌ No valid ray intersections found. Check mesh alignment.")
        return
    
    # Print summary results
    print("\n" + "="*60)
    print("📊 HEAT SINK ANALYSIS RESULTS")
    print("="*60)
    
    # Group results by vessel type
    vessel_results = {}
    for result in results:
        vessel_name = result["vessel_name"]
        if vessel_name not in vessel_results:
            vessel_results[vessel_name] = []
        vessel_results[vessel_name].append(result)
    
    print(f"\n📈 Summary by vessel type:")
    print("-" * 80)
    print(f"{'Vessel':<15} {'Count':<8} {'Avg Distance':<12} {'Avg Heat Loss':<15} {'Max Heat Loss':<15}")
    print("-" * 80)
    
    for vessel_name, vessel_data in vessel_results.items():
        count = len(vessel_data)
        avg_dist = np.mean([r["distance_to_vessel_m"] for r in vessel_data]) * 1000  # mm
        avg_loss = np.mean([r["energy_loss_percent"] for r in vessel_data])
        max_loss = np.max([r["energy_loss_percent"] for r in vessel_data])
        
        print(f"{vessel_name:<15} {count:<8} {avg_dist:<12.2f} {avg_loss:<15.3f} {max_loss:<15.3f}")
    
    # Find critical paths
    print(f"\n🔥 Critical heat loss paths (top 10):")
    print("-" * 100)
    print(f"{'Vessel':<15} {'Distance (mm)':<15} {'Heat Loss (%)':<15} {'Power Loss (W)':<15}")
    print("-" * 100)
    
    # Sort by heat loss percentage
    sorted_results = sorted(results, key=lambda x: x["energy_loss_percent"], reverse=True)
    
    for result in sorted_results[:10]:
        dist_mm = result["distance_to_vessel_m"] * 1000
        heat_loss_pct = result["energy_loss_percent"]
        power_loss = result["heat_loss_power_W"]
        vessel = result["vessel_name"]
        
        print(f"{vessel:<15} {dist_mm:<15.2f} {heat_loss_pct:<15.3f} {power_loss:<15.3f}")
    
    # Overall statistics
    all_heat_losses = [r["energy_loss_percent"] for r in results]
    all_distances = [r["distance_to_vessel_m"] * 1000 for r in results]
    
    print(f"\n📊 Overall Statistics:")
    print(f"   Total analyzed paths: {len(results)}")
    print(f"   Average heat loss: {np.mean(all_heat_losses):.3f}%")
    print(f"   Maximum heat loss: {np.max(all_heat_losses):.3f}%")
    print(f"   Average distance to vessels: {np.mean(all_distances):.2f} mm")
    print(f"   Minimum distance to vessels: {np.min(all_distances):.2f} mm")
    
    # Create visualization if possible
    try:
        create_visualization(tumor, surface, vessels, centroid, results)
    except Exception as e:
        print(f"\n⚠️ Visualization failed: {e}")
        print("   Analysis results are still valid.")
    
    print(f"\n✅ Heat sink analysis completed successfully!")
    return results

def create_visualization(tumor, surface, vessels, centroid, results):
    """Create 3D visualization of the analysis results."""
    
    print(f"\n🎨 Creating 3D visualization...")
    
    # Create plotter
    plotter = pv.Plotter(window_size=[1200, 800])
    
    # Add anatomical structures
    plotter.add_mesh(surface, color='lightgray', opacity=0.2, label='Body Surface')
    plotter.add_mesh(tumor, color='red', opacity=0.8, label='Tumor')
    
    # Add vessels with different colors
    vessel_colors = ['blue', 'green', 'purple', 'orange', 'cyan']
    for i, vessel in enumerate(vessels):
        color = vessel_colors[i % len(vessel_colors)]
        plotter.add_mesh(vessel, color=color, opacity=0.6, 
                        label=f'Vessel {i+1}')
    
    # Add tumor centroid
    centroid_sphere = pv.Sphere(radius=2, center=centroid)
    plotter.add_mesh(centroid_sphere, color='yellow', 
                    label='Tumor Center')
    
    # Add sample rays colored by heat loss
    if results:
        heat_losses = np.array([r["energy_loss_percent"] for r in results])
        
        if len(heat_losses) > 0 and np.max(heat_losses) > np.min(heat_losses):
            # Normalize heat losses for color mapping
            norm_losses = (heat_losses - np.min(heat_losses)) / (np.max(heat_losses) - np.min(heat_losses))
            
            # Show every 10th ray to avoid clutter
            for i in range(0, len(results), max(1, len(results)//50)):
                result = results[i]
                direction = result["ray_direction"]
                distance = result["path_distance"]
                color_val = norm_losses[i]
                
                # Create ray line
                end_point = centroid + direction * distance
                line = pv.Line(centroid, end_point)
                
                # Color from blue (low heat loss) to red (high heat loss)
                color = [color_val, 0, 1-color_val]
                plotter.add_mesh(line, color=color, line_width=2)
    
    # Add legend and labels
    plotter.add_legend(loc='upper right')
    plotter.add_text("Heat Sink Effect Analysis\n(Rays colored by heat loss intensity)", 
                    position='upper_left', font_size=12)
    
    # Set view and show
    plotter.show_axes()
    plotter.show_grid()
    
    try:
        plotter.show(auto_close=False)
    except:
        # Fallback for systems without display
        print("   Display not available - skipping interactive visualization")

# --------------------------------------------------------
# Main Execution
# --------------------------------------------------------

if __name__ == "__main__":
    try:
        # Run the heat sink analysis
        results = analyze_heat_sink_effect()
        
        if results:
            print(f"\n🎯 Analysis saved! You can access the results programmatically.")
            print(f"   Total paths analyzed: {len(results)}")
            print(f"   Results contain detailed heat sink parameters for each path.")
            
    except KeyboardInterrupt:
        print("\n\n⏹️ Analysis interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        print("Please check your dataset files and paths.")