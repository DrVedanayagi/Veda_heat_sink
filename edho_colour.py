#!/usr/bin/env python3
"""
Ablation Planner v5
===================

Features:
• Empirical manufacturer table
• Directional heat sink deformation
• Live OAZ calculation
• Needle model + thermocouple markers
• Vessel recoloring (user specified)
• Play/Pause button
• Power/Time selector
• Continuous time animation
"""

import os
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# DATA PATHS
# ─────────────────────────────────────────────

DATASET_BASE = r"C:\Users\z005562w\OneDrive - Siemens Healthineers\Veda\Project\siemens project\3d simulation\Nunna Algo\Nunna Algo\Dataset"
PORTALVENOUS_DIR = os.path.join(DATASET_BASE, "portalvenous")

TUMOR_VTK = os.path.join(DATASET_BASE, "908ac523data00007_leasion_out.vtk")
SURFACE_VTK = os.path.join(DATASET_BASE, "908ac523data00013_skin_out.vtk")

VESSEL_FILES = [
    "908ac52300001.vtk",  # portal vein
    "908ac52300002.vtk",  # hepatic vein
    "908ac52300003.vtk",  # aorta
    "908ac52300004.vtk",  # ivc
    "908ac52300005.vtk"   # hepatic artery
]

VESSEL_NAMES = ["portal_vein","hepatic_vein","aorta","ivc","hepatic_artery"]

# USER REQUESTED COLORS
VESSEL_COLORS = {
    "portal_vein": "darkblue",
    "hepatic_vein": "darkblue",
    "aorta": "red",
    "ivc": "deepskyblue",
    "hepatic_artery": "orange"
}

# ─────────────────────────────────────────────
# PHYSICS CONSTANTS
# ─────────────────────────────────────────────

RHO_B=1060
MU_B=3.5e-3
C_B=3700
K_B=0.52
T_BLOOD=37
T_TISSUE=90

VESSEL_DIAM = {"portal_vein":12e-3,"hepatic_vein":8e-3,"aorta":25e-3,"ivc":20e-3,"hepatic_artery":4.5e-3}
VESSEL_VEL  = {"portal_vein":0.15,"hepatic_vein":0.20,"aorta":0.40,"ivc":0.35,"hepatic_artery":0.30}

# ─────────────────────────────────────────────
# MANUFACTURER TABLE
# (Power, Time, Forward Throw cm, Diameter cm)
# ─────────────────────────────────────────────

ABLATION_TABLE = [
    (60,480,6.33,3.8),
    (60,600,5.82,3.9),
    (90,300,5.20,3.7),
    (120,600,9.40,5.6)
]

# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────

def load_mesh(path):
    mesh = pv.read(path)
    if np.max(np.abs(mesh.points))>1000:
        mesh.points/=1000
    return mesh

def reynolds(vname):
    D=VESSEL_DIAM[vname]
    u=VESSEL_VEL[vname]
    return (RHO_B*u*D)/MU_B

def heat_loss_fraction(dist,vname,power,time):
    D=VESSEL_DIAM[vname]
    u=VESSEL_VEL[vname]
    Re=reynolds(vname)
    Pr=(C_B*MU_B)/K_B
    Nu=4.36 if Re<2300 else 0.023*(Re**0.8)*(Pr**0.4)
    h=Nu*K_B/D
    A=np.pi*D*0.01
    Q=h*A*(T_TISSUE-T_BLOOD)
    Ein=power*time
    E_loss=Q*time*np.exp(-70*dist)
    return min(E_loss/Ein,0.9)

def generate_rays(n=400):
    phi=np.random.uniform(0,2*np.pi,n)
    cost=np.random.uniform(-1,1,n)
    sint=np.sqrt(1-cost**2)
    return np.column_stack((sint*np.cos(phi),sint*np.sin(phi),cost))

# ─────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────

class AblationSystem:

    def __init__(self):
        self.power,self.time,self.fwd_cm,self.diam_cm = ABLATION_TABLE[0]
        self.is_playing=False
        self.current_time=0

    def load(self):
        self.tumor=load_mesh(TUMOR_VTK)
        self.surface=load_mesh(SURFACE_VTK)
        self.vessels={}
        for file,name in zip(VESSEL_FILES,VESSEL_NAMES):
            self.vessels[name]=load_mesh(os.path.join(PORTALVENOUS_DIR,file))

        self.centroid=np.array(self.tumor.center)
        self.vpts=np.vstack([v.points for v in self.vessels.values()])
        self.vids=np.concatenate([np.full(len(v.points),i) for i,v in enumerate(self.vessels.values())])

    # ───────────────
    # NEEDLE MODEL
    # ───────────────
    def needle_model(self,axis):
        tip=self.centroid-0.05*axis
        shaft=pv.Cylinder(center=(tip+self.centroid)/2,
                           direction=axis,
                           radius=0.001,
                           height=0.05)
        markers=[]
        for i in range(4):
            pos=tip+axis*(0.01*i)
            markers.append(pv.Sphere(radius=0.002,center=pos))
        return shaft,markers

    # ───────────────
    # DIRECTIONAL ELLIPSOID
    # ───────────────
    def build_deformed_zone(self,fraction):

        a=(self.diam_cm/100)/2*fraction
        c=(self.fwd_cm/100)/2*fraction

        rays=generate_rays()
        pts=[]
        axis=np.array([0,0,1])

        tree=cKDTree(self.vpts)

        for d in rays:
            cospsi=np.dot(d,axis)
            sin2=1-cospsi**2
            R0=1/np.sqrt(sin2/a**2+cospsi**2/c**2)
            dist,idx=tree.query(self.centroid+d*R0)
            vname=list(self.vessels.keys())[idx%len(self.vessels)]
            loss=heat_loss_fraction(dist,vname,self.power,self.time)
            Reff=R0*(1-loss)**(1/3)
            pts.append(self.centroid+d*Reff)

        cloud=pv.PolyData(np.array(pts))
        zone=cloud.delaunay_3d().extract_geometry()
        zone["Temp"]=np.linspace(T_BLOOD,T_TISSUE,zone.n_points)
        return zone

    # ───────────────
    # OAZ
    # ───────────────
    def compute_oaz(self,zone):

        tumor_pts=self.tumor.points
        inside=zone.select_enclosed_points(self.tumor)
        outside=np.sum(inside.point_data["SelectedPoints"]==0)
        return 100*outside/len(tumor_pts)

    # ───────────────
    # UI UPDATE
    # ───────────────
    def update(self,t):

        self.current_time=t
        frac=min(t/self.time,1)

        try:self.plotter.remove_actor("zone")
        except:pass
        zone=self.build_deformed_zone(frac)
        self.plotter.add_mesh(zone,scalars="Temp",cmap="plasma",
                              clim=[37,90],opacity=0.6,name="zone")

        oaz=self.compute_oaz(zone)

        energy=self.power*t

        hud=f"""t={t:.0f}s / {self.time}s
Power={self.power}W
Energy={energy:.0f}J
Zone={self.fwd_cm*frac:.1f}cm x {self.diam_cm*frac:.1f}cm
OAZ={oaz:.1f}%"""

        try:self.plotter.remove_actor("hud")
        except:pass
        self.plotter.add_text(hud,position="lower_left",
                              name="hud",font_size=12)

        self.plotter.render()

    # ───────────────
    # PLAY LOOP
    # ───────────────
    def auto_play(self):
        if not self.is_playing:return
        self.current_time+=2
        if self.current_time>self.time:
            self.is_playing=False
            return
        self.update(self.current_time)

    # ───────────────
    # RUN
    # ───────────────
    def run(self):

        self.plotter=pv.Plotter(window_size=(1500,1000))
        self.plotter.background_color="black"

        self.plotter.add_mesh(self.surface,color="lightgray",opacity=0.08)

        for name,mesh in self.vessels.items():
            self.plotter.add_mesh(mesh,color=VESSEL_COLORS[name],opacity=0.6,label=name)

        self.plotter.add_mesh(self.tumor,color="yellow",opacity=0.8)

        axis=np.array([0,0,1])
        shaft,markers=self.needle_model(axis)
        self.plotter.add_mesh(shaft,color="silver")
        for m in markers:self.plotter.add_mesh(m,color="white")

        self.plotter.add_slider_widget(
            lambda v:self.update(v),
            rng=[0,self.time],
            value=0,
            title="Ablation Time (s)",
            pointa=(0.15,0.05),
            pointb=(0.85,0.05)
        )

        def toggle():
            self.is_playing=not self.is_playing

        self.plotter.add_button_widget(toggle,position=(20,20),
                                       size=40,color_on="green",color_off="gray")

        self.plotter.add_callback(self.auto_play,interval=100)

        self.update(0)
        self.plotter.show()

# ─────────────────────────────────────────────
# RUN SYSTEM
# ─────────────────────────────────────────────

if __name__=="__main__":
    system=AblationSystem()
    system.load()
    system.run()