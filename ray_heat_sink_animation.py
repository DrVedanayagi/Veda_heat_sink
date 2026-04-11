#!/usr/bin/env python3
"""
Advanced Heat Sink Animation System for Liver Ablation
=====================================================

Complete PyVista-based animation showing:
- Blood flow with moving particles
- Ablation zone growth over time
- Heat sink effects around vessels
- Temperature field evolution
- Interactive controls

Author: Enhanced for real-time visualization
Date: November 2025
"""

import os
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import time

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --------------------------------------------------------
# Configuration and File Paths
# --------------------------------------------------------

# Dataset paths
DATASET_BASE = r"C:\Users\z005562w\OneDrive - Siemens Healthineers\Veda\Project\siemens project\3d simulation\Nunna Algo\Nunna Algo\Dataset"
PORTALVENOUS_DIR = os.path.join(DATASET_BASE, "portalvenous")

# VTK file paths
TUMOR_VTK = os.path.join(DATASET_BASE, "908ac523data00007_leasion_out.vtk")
SURFACE_VTK = os.path.join(DATASET_BASE, "908ac523data00013_skin_out.vtk")

VESSEL_VTK_LIST = [
    os.path.join(PORTALVENOUS_DIR, "908ac52300001.vtk"),    # portal vein
    os.path.join(PORTALVENOUS_DIR, "908ac52300002.vtk"),    # hepatic vein
    os.path.join(PORTALVENOUS_DIR, "908ac52300003.vtk"),    # aortapyth
    os.path.join(PORTALVENOUS_DIR, "908ac52300004.vtk"),    # IVC
    os.path.join(PORTALVENOUS_DIR, "908ac52300005.vtk")     # hepatic artery
]

VESSEL_NAMES = ["portal_vein", "hepatic_vein", "aorta", "ivc", "hepatic_artery"]
VESSEL_COLORS = ['purple', 'teal', 'royalblue', 'navy', 'orange']

# --------------------------------------------------------
# Physical Constants and Animation Parameters
# --------------------------------------------------------

# Blood properties
RHO_B = 1060.0         # blood density (kg/m³)
C_B = 3700.0          # blood specific heat (J/kg·K)
K_B = 0.52            # blood thermal conductivity (W/m·K)
T_BLOOD = 37.0        # blood core temperature (°C)
T_TISSUE = 90.0       # tissue ablation temperature (°C)

# Ablation settings
POWER = 30.0          # microwave power (W)
ABLATION_TIME = 60.0  # ablation time (s)

# Animation settings
FPS = 30              # frames per second
TOTAL_FRAMES = int(ABLATION_TIME * FPS / 10)  # 10x speed up for demo
TIME_STEP = ABLATION_TIME / TOTAL_FRAMES

# Vessel parameters (diameters in m, velocities in m/s)
VESSEL_DIAMETERS = [12e-3, 8e-3, 25e-3, 20e-3, 4.5e-3]  # m
VESSEL_VELOCITIES = [0.15, 0.20, 0.40, 0.35, 0.30]      # m/s

# --------------------------------------------------------
# Utility Functions
# --------------------------------------------------------

def load_and_prepare_meshes():
    """Load all VTK meshes and prepare for animation."""
    print("🔄 Loading and preparing meshes for animation...")
    
    # Load meshes
    tumor = pv.read(TUMOR_VTK)
    surface = pv.read(SURFACE_VTK)
    vessels = [pv.read(path) for path in VESSEL_VTK_LIST if os.path.exists(path)]
    
    # Rescale if needed (mm to m)
    def rescale_mesh(mesh):
        if mesh and np.max(np.abs(mesh.points)) > 1000:
            mesh.points = mesh.points / 1000.0
        return mesh
    
    tumor = rescale_mesh(tumor)
    surface = rescale_mesh(surface)
    vessels = [rescale_mesh(v) for v in vessels]
    
    print(f"✅ Loaded: Tumor ({tumor.n_points} pts), Surface ({surface.n_points} pts), {len(vessels)} vessels")
    
    return tumor, surface, vessels

def create_blood_flow_particles(vessel, n_particles=100):
    """Create blood flow particles for animation."""
    # Sample points along vessel centerline
    if vessel.n_points < n_particles:
        n_particles = vessel.n_points // 2
    
    # Get random points on vessel surface
    sample_indices = np.random.choice(vessel.n_points, n_particles, replace=False)
    particle_positions = vessel.points[sample_indices]
    
    # Create particle spheres
    particles = pv.PolyData(particle_positions)
    
    # Add particle properties
    particle_sizes = np.random.uniform(0.001, 0.003, n_particles)  # Random sizes
    particles['size'] = particle_sizes
    particles['velocity'] = np.random.uniform(0.1, 0.5, n_particles)  # m/s
    
    return particles

def calculate_temperature_field(tumor_center, vessels, time_current, grid_resolution=20):
    """Calculate 3D temperature field around tumor at given time."""
    
    # Create simulation grid around tumor
    extent = 0.05  # 5cm around tumor
    x = np.linspace(tumor_center[0] - extent, tumor_center[0] + extent, grid_resolution)
    y = np.linspace(tumor_center[1] - extent, tumor_center[1] + extent, grid_resolution)
    z = np.linspace(tumor_center[2] - extent, tumor_center[2] + extent, grid_resolution)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    # Distance from tumor center
    dist_tumor = np.linalg.norm(grid_points - tumor_center, axis=1)
    
    # Heat diffusion from tumor (time-dependent)
    ablation_radius = 0.005 + (time_current / ABLATION_TIME) * 0.015  # 5-20mm growth
    heat_intensity = np.exp(-((dist_tumor / ablation_radius) ** 2))
    
    # Vessel cooling effects
    vessel_cooling = np.zeros_like(dist_tumor)
    if vessels:
        all_vessel_points = np.vstack([v.points for v in vessels if v is not None])
        if len(all_vessel_points) > 0:
            vessel_tree = cKDTree(all_vessel_points)
            dist_vessels, _ = vessel_tree.query(grid_points)
            
            # Exponential cooling decay from vessels
            cooling_radius = 0.010  # 10mm cooling radius
            vessel_cooling = 0.8 * np.exp(-(dist_vessels / cooling_radius) ** 2)
    
    # Temperature calculation
    T_base = T_BLOOD
    T_max_increase = (T_TISSUE - T_BLOOD) * (time_current / ABLATION_TIME)
    temperature = T_base + T_max_increase * heat_intensity * (1 - vessel_cooling)
    python .\
    # Create structured grid
    temp_grid = pv.StructuredGrid(X, Y, Z)
    temp_grid['Temperature'] = temperature
    
    return temp_grid

def create_ablation_zone(tumor_center, time_current):
    """Create ablation zone visualization."""
    
    # Zone grows over time
    base_radius = 0.005  # 5mm initial
    max_radius = 0.020   # 20mm final
    current_radius = base_radius + (time_current / ABLATION_TIME) * (max_radius - base_radius)
    
    # Create ablation zone sphere
    ablation_sphere = pv.Sphere(radius=current_radius, center=tumor_center, 
                               theta_resolution=30, phi_resolution=30)
    
    # Add temperature data
    distances = np.linalg.norm(ablation_sphere.points - tumor_center, axis=1)
    temperatures = T_BLOOD + (T_TISSUE - T_BLOOD) * np.exp(-(distances / current_radius) ** 2)
    ablation_sphere['Temperature'] = temperatures
    
    return ablation_sphere

def animate_blood_particles(particles, vessel_idx, time_step):
    """Animate blood particle movement."""
    if particles is None or particles.n_points == 0:
        return particles
    
    # Get vessel velocity
    velocity = VESSEL_VELOCITIES[vessel_idx] if vessel_idx < len(VESSEL_VELOCITIES) else 0.2
    
    # Move particles along vessel direction (simplified as random walk)
    displacement = np.random.normal(0, velocity * time_step * 0.1, (particles.n_points, 3))
    particles.points = particles.points + displacement
    
    # Add some randomness for realistic flow
    particles.points += np.random.normal(0, 0.001, particles.points.shape)
    
    return particles

# --------------------------------------------------------
# Main Animation Class
# --------------------------------------------------------

class HeatSinkAnimator:
    """Complete heat sink animation system."""
    
    def __init__(self):
        self.tumor = None
        self.surface = None
        self.vessels = []
        self.tumor_center = None
        self.blood_particles = []
        self.current_time = 0.0
        self.plotter = None
        self.is_playing = False
        
    def load_data(self):
        """Load all medical data."""
        self.tumor, self.surface, self.vessels = load_and_prepare_meshes()
        self.tumor_center = np.array(self.tumor.center)
        
        # Create blood particles for each vessel
        print("🩸 Creating blood flow particles...")
        for i, vessel in enumerate(self.vessels):
            if vessel is not None:
                particles = create_blood_flow_particles(vessel, n_particles=50)
                self.blood_particles.append(particles)
            else:
                self.blood_particles.append(None)
    
    def setup_visualization(self):
        """Setup the 3D visualization environment."""
        print("🎬 Setting up 3D animation environment...")
        
        # Create plotter with custom settings
        self.plotter = pv.Plotter(window_size=[1400, 1000])
        self.plotter.background_color = 'black'
        
        # Add static anatomy
        self.plotter.add_mesh(self.surface, color='lightgray', opacity=0.1, 
                             name='surface', label='Body Surface')
        
        # Add vessels with different colors
        for i, (vessel, color, name) in enumerate(zip(self.vessels, VESSEL_COLORS, VESSEL_NAMES)):
            if vessel is not None:
                self.plotter.add_mesh(vessel, color=color, opacity=0.6, 
                                     name=f'vessel_{i}', label=name.replace('_', ' ').title())
        
        # Add tumor
        self.plotter.add_mesh(self.tumor, color='red', opacity=0.8, 
                             name='tumor', label='Tumor')
        
        # Add tumor center marker
        center_sphere = pv.Sphere(radius=0.002, center=self.tumor_center)
        self.plotter.add_mesh(center_sphere, color='yellow', 
                             name='tumor_center', label='Tumor Center')
        
        # Setup camera and lighting
        self.plotter.camera.position = (0.2, 0.2, 0.3)
        self.plotter.camera.focal_point = self.tumor_center
        self.plotter.add_axes()
        
        # Add legend
        self.plotter.add_legend(loc='upper right', size=(0.2, 0.3))
        
        # Add title
        self.plotter.add_text("Heat Sink Effect Animation - Liver Ablation", 
                             position='upper_left', font_size=16, color='white')
    
    def update_animation(self, time_value):
        """Update animation frame."""
        self.current_time = float(time_value)
        
        try:
            # Remove previous dynamic elements
            actors_to_remove = ['ablation_zone', 'temp_field', 'time_text']
            for i in range(len(self.vessels)):
                actors_to_remove.append(f'particles_{i}')
            
            for actor_name in actors_to_remove:
                try:
                    self.plotter.remove_actor(actor_name)
                except:
                    pass
            
            # 1. Update ablation zone
            ablation_zone = create_ablation_zone(self.tumor_center, self.current_time)
            self.plotter.add_mesh(ablation_zone, scalars='Temperature', 
                                 cmap='hot', opacity=0.7, name='ablation_zone',
                                 scalar_bar_args={'title': 'Temperature (°C)'})
            
            # 2. Update temperature field (every few frames for performance)
            if int(self.current_time * 10) % 3 == 0:  # Update every 0.3 seconds
                temp_field = calculate_temperature_field(self.tumor_center, self.vessels, 
                                                       self.current_time, grid_resolution=15)
                
                # Create volume rendering of temperature
                self.plotter.add_volume(temp_field, scalars='Temperature', 
                                       cmap='plasma', opacity='sigmoid',
                                       name='temp_field')
            
            # 3. Update blood particles
            for i, particles in enumerate(self.blood_particles):
                if particles is not None:
                    # Animate particles
                    animated_particles = animate_blood_particles(particles, i, TIME_STEP)
                    
                    # Add animated particles
                    if animated_particles.n_points > 0:
                        self.plotter.add_mesh(animated_particles, 
                                            color='red', point_size=3,
                                            name=f'particles_{i}')
            
            # 4. Update time display
            time_text = f"Time: {self.current_time:.1f}s / {ABLATION_TIME:.0f}s\n"
            time_text += f"Power: {POWER:.0f}W\n"
            time_text += f"Zone Radius: {(0.005 + (self.current_time/ABLATION_TIME)*0.015)*1000:.1f}mm"
            
            self.plotter.add_text(time_text, position='lower_left', 
                                 font_size=12, color='white', name='time_text')
            
            # Force render update
            self.plotter.render()
            
        except Exception as e:
            print(f"Animation update error: {e}")
    
    def run_animation(self):
        """Run the complete animation."""
        print("🚀 Starting heat sink animation...")
        
        # Load data
        self.load_data()
        
        # Setup visualization
        self.setup_visualization()
        
        # Add time slider
        self.plotter.add_slider_widget(
            callback=self.update_animation,
            rng=[0, ABLATION_TIME],
            value=0,
            title="Ablation Time (seconds)",
            pointa=(0.1, 0.05),
            pointb=(0.9, 0.05),
            style='modern'
        )
        
        # Add play/pause button
        def toggle_play():
            self.is_playing = not self.is_playing
            if self.is_playing:
                self.auto_advance()
        
        # Add control instructions
        instructions = """
        🎮 ANIMATION CONTROLS:
        • Drag time slider to navigate
        • Mouse: Rotate, zoom, pan
        • 'r': Reset camera view
        • 'q': Quit animation
        
        📊 VISUALIZATION ELEMENTS:
        🔴 Red sphere: Growing ablation zone
        🌈 Color field: Temperature gradient
        🩸 Red dots: Blood flow particles
        💙 Blue/Teal: Major vessels
        """
        
        self.plotter.add_text(instructions, position='upper_right', 
                             font_size=10, color='lightblue')
        
        # Initial frame
        self.update_animation(0.0)
        
        # Show animation
        print("✨ Animation ready! Use the slider to control time.")
        print("   Close the window when done.")
        
        try:
            self.plotter.show(auto_close=False)
        except KeyboardInterrupt:
            print("\n🛑 Animation stopped by user")
        except Exception as e:
            print(f"🔥 Visualization error: {e}")
        finally:
            self.plotter.close()
    
    def auto_advance(self):
        """Auto-advance animation (optional feature)."""
        # This could be implemented for automatic playback
        # For now, manual control via slider is recommended
        pass

# --------------------------------------------------------
# Main Execution
# --------------------------------------------------------

def create_heat_sink_animation():
    """Create and run the complete heat sink animation."""
    
    print("=" * 60)
    print("🎬 ADVANCED HEAT SINK ANIMATION SYSTEM")
    print("=" * 60)
    print("🔥 Real-time visualization of:")
    print("   • Blood flow with moving particles")
    print("   • Ablation zone growth over time") 
    print("   • Temperature field evolution")
    print("   • Heat sink effects around vessels")
    print("   • Interactive time control")
    print("=" * 60)
    
    # Create and run animator
    animator = HeatSinkAnimator()
    animator.run_animation()
    
    print("✅ Animation complete!")

if __name__ == "__main__":
    try:
        # Check if running in proper environment
        if not os.path.exists(DATASET_BASE):
            print(f"❌ Dataset not found at: {DATASET_BASE}")
            print("Please ensure your VTK files are in the correct location.")
            exit(1)
        
        # Run the complete animation
        create_heat_sink_animation()
        
    except KeyboardInterrupt:
        print("\n🛑 Animation interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure all required packages are installed:")
        print("pip install pyvista numpy scipy matplotlib tqdm")