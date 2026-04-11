#!/usr/bin/env python3
"""
Physics-Grounded Heat-Sink–Aware Microwave Ablation Planner
===========================================================

• Empirical animal-study ablation geometry
• Ellipsoidal ablation zone (diameter, length, forward throw)
• Directional vessel heat-sink correction
• OAZ (Outside Ablation Zone) computation
• Automatic treatment parameter recommendation
• Fully interactive PyVista 3D animation

Author: Vedanayagi Rajarajan
"""

import os
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings("ignore")

# ==========================================================
# DATASET PATHS
# ==========================================================

DATASET_BASE = r"C:\Users\z005562w\OneDrive - Siemens Healthineers\Veda\Project\siemens project\3d simulation\Nunna Algo\Nunna Algo\Dataset"
PORTALVENOUS_DIR = os.path.join(DATASET_BASE, "portalvenous")

TUMOR_VTK   = os.path.join(DATASET_BASE, "908ac523data00007_leasion_out.vtk")
SURFACE_VTK = os.path.join(DATASET_BASE, "908ac523data00013_skin_out.vtk")

VESSEL_VTK_LIST = [
    "908ac52300001.vtk",  # portal vein
    "908ac52300002.vtk",  # hepatic vein
    "908ac52300003.vtk",  # aorta
    "908ac52300004.vtk",  # ivc
    "908ac52300005.vtk"   # hepatic artery
]

VESSEL_NAMES  = ["portal_vein", "hepatic_vein", "aorta", "ivc", "hepatic_artery"]
VESSEL_COLORS = ["purple", "teal", "royalblue", "navy", "gold"]

# ==========================================================
# PHYSICAL CONSTANTS
# ==========================================================

RHO_B = 1060.0
MU_B  = 3.5e-3
C_B   = 3700.0
K_B   = 0.52

T_BLOOD  = 37.0
T_TISSUE = 90.0

# ==========================================================
# EMPIRICAL ABLATION TABLE (SI UNITS)
# (Power W, Time s, Forward Throw m, Diameter m, Length m)
# ==========================================================

ABLATION_TABLE = [
    (30, 180, 0.022, 0.019, 0.023),
    (30, 300, 0.025, 0.024, 0.027),
    (30, 480, 0.049, 0.029, 0.030),
    (30, 600, 0.0547,0.031, 0.031),
    (60, 300, 0.047, 0.030, 0.033),
    (60, 480, 0.063, 0.038, 0.038),
    (90, 300, 0.052, 0.037, 0.038),
    (120,600, 0.094, 0.056, 0.050)
]

def lookup_geometry(power, time):
    for P,t,ft,d,L in ABLATION_TABLE:
        if P==power and t==time:
            return d/2, L/2, ft
    raise ValueError("No matching ablation parameters")

# ==========================================================
# VESSEL PARAMETERS
# ==========================================================

VESSEL_DIAM = [12e-3, 8e-3, 25e-3, 20e-3, 4.5e-3]
VESSEL_VEL  = [0.15,  0.20,  0.40,   0.35,   0.30]

# ==========================================================
# GEOMETRY + PHYSICS FUNCTIONS
# ==========================================================

def ellipsoid_radius(direction, axis, a, c):
    cospsi = np.dot(direction, axis)
    sin2   = 1 - cospsi**2
    return 1.0 / np.sqrt(sin2/a**2 + cospsi**2/c**2)

def heat_sink_loss(point, vessel_pts, vessel_ids, power, time):
    tree = cKDTree(vessel_pts)
    dist, idx = tree.query(point)
    vid = vessel_ids[idx]

    D = VESSEL_DIAM[vid]
    u = VESSEL_VEL[vid]

    Re = (RHO_B*u*D)/MU_B
    Pr = (C_B*MU_B)/K_B
    Nu = 0.023*(Re**0.8)*(Pr**0.4) if Re>2300 else 4.36

    h = Nu*K_B/D
    A = np.pi*D*0.01
    Q = h*A*(T_TISSUE-T_BLOOD)
    Ein = power*time

    return min(Q*time/Ein, 0.9)

def generate_rays(n=400):
    phi = np.random.uniform(0,2*np.pi,n)
    cost = np.random.uniform(-1,1,n)
    sint = np.sqrt(1-cost**2)
    return np.column_stack((sint*np.cos(phi), sint*np.sin(phi), cost))

# ==========================================================
# MAIN ANIMATION CLASS
# ==========================================================

class AblationAnimator:

    def __init__(self, power=30, time=600):
        self.power = power
        self.time  = time

    def load(self):
        self.tumor   = pv.read(TUMOR_VTK)
        self.surface = pv.read(SURFACE_VTK)
        self.vessels = [pv.read(os.path.join(PORTALVENOUS_DIR,f)) for f in VESSEL_VTK_LIST]

        for m in [self.tumor,self.surface,*self.vessels]:
            if np.max(np.abs(m.points))>1000:
                m.points/=1000

        self.center = np.array(self.tumor.center)
        self.axis   = np.array([0,0,1])

        self.vpts   = np.vstack([v.points for v in self.vessels])
        self.vids   = np.concatenate([np.full(len(v.points),i) for i,v in enumerate(self.vessels)])

    def compute_oaz(self):
        a,c,ft = lookup_geometry(self.power,self.time)
        rays = generate_rays()
        tumor_r = max(self.tumor.length,self.tumor.width,self.tumor.height)/2

        oaz=[]
        for d in rays:
            R0 = ellipsoid_radius(d,self.axis,a,c)
            loss = heat_sink_loss(self.center+d*R0,self.vpts,self.vids,self.power,self.time)
            Reff = R0*(1-loss)**(1/3)
            if Reff<tumor_r:
                oaz.append(1-(Reff/tumor_r)**3)
            else:
                oaz.append(0)

        return np.mean(oaz), np.max(oaz)

    def animate(self):
        plotter = pv.Plotter(window_size=(1500,1000))
        plotter.background_color="black"

        plotter.add_mesh(self.surface,color="slategray",opacity=0.08)
        plotter.add_mesh(self.tumor,color="crimson",opacity=0.8)

        for v,c in zip(self.vessels,VESSEL_COLORS):
            plotter.add_mesh(v,color=c,opacity=0.6)

        plotter.add_mesh(pv.Sphere(radius=0.002,center=self.center),color="yellow")

        a,c,ft = lookup_geometry(self.power,self.time)
        ell = pv.ParametricEllipsoid(a,a,c)
        ell.points += self.center + self.axis*ft
        ell["Temp"] = np.linspace(T_BLOOD,T_TISSUE,ell.n_points)

        plotter.add_mesh(ell,scalars="Temp",cmap="inferno",opacity=0.7)

        mean_oaz,max_oaz = self.compute_oaz()

        txt = f"""Power: {self.power} W
Time: {self.time}s
Mean OAZ: {mean_oaz*100:.1f} %
Max OAZ: {max_oaz*100:.1f} %"""

        plotter.add_text(txt,position="upper_left",font_size=14,color="white")
        plotter.add_axes()
        plotter.show()

# ==========================================================
# RUN
# ==========================================================

if __name__=="__main__":
    sim = AblationAnimator(power=30,time=600)
    sim.load()
    sim.animate()
