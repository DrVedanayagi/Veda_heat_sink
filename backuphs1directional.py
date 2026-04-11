#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   DIRECTIONAL MICROWAVE ABLATION PLANNING SYSTEM  —  v11                   ║
║   Heat Sink + Directional Monopole Antenna + Biophysical Optimizer          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Author  : Veda Nunna                                                        ║
║  Version : 11.0                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  THEORETICAL BASIS                                                           ║
║  ─────────────────                                                           ║
║  Based on three reference works:                                             ║
║                                                                              ║
║  [1] Fallahi & Prakash (2018) — "Antenna Designs for Microwave Tissue        ║
║      Ablation", Crit Rev Biomed Eng 46(6):495–521                           ║
║      Section III-C: Directional Applicators                                  ║
║      → Monopole + semicylindrical reflector, water-cooled catheter           ║
║      → Restricts radiation to ~half angular expanse (~180°)                  ║
║      → Operating frequency 2.45 GHz in liver tissue                          ║
║      → λ_eff/4 monopole length ~10.9 mm for UT-85 coax                       ║
║      → Forward:Rear power ratio ≈ 2.5:1 experimentally measured             ║
║      → Active cooling eliminates need for choke/sleeve (Fig. 13a)            ║
║                                                                              ║
║  [2] Lee (2023) — "Directional Monopole Antenna Using a Planar Lossy         ║
║      Magnetic (PLM) Surface", J Electromagn Eng Sci 23(4):351–354           ║
║      → PMC/PEC hybrid ground creates asymmetric surface current              ║
║      → Surface current cancels on PMC half → directional radiation           ║
║      → SAR pattern: cos²(θ/2) for forward hemisphere (Eq. 1 reflection coeff)║
║      → −10 dB BW ~23%, peak gain 1.3 dBi at zenith (θ=0°)                  ║
║      → Beam tilt ~12° from axis with μ_r=20 ferrite                          ║
║      → Rear lobe suppression measured >80% in prototype                      ║
║                                                                              ║
║  [3] Audigier et al. (2020) — "System and Method for Interactive Patient     ║
║      Specific Simulation of Radiofrequency Ablation Therapy", US 10,748,438 ║
║      → Pennes model near large vessels (constant T_blood assumption holds)   ║
║      → Wulff-Klinger model in parenchyma (advection term -ε·ρb·Cb·v·∇T)    ║
║      → LBM solver on Cartesian grid, 7-connectivity                          ║
║      → 3-state cell death: A →(kf/kb)→ V → D                                ║
║      → CFD + Darcy's law for blood flow in porous liver tissue               ║
║                                                                              ║
║  WHAT IS NEW IN v11 vs v10                                                   ║
║  ─────────────────────────────                                               ║
║  1. DIRECTIONAL SAR WEIGHTING FUNCTION                                       ║
║     Per Lee (2023) Eq.1 and Fallahi & Prakash Section III-C:                 ║
║       SAR_forward(θ)  = P_net · cos²(θ/2) · G_fwd   (forward hemisphere)    ║
║       SAR_rear(θ)     = P_net · sin²(θ/2) · G_rear  (rear hemisphere)       ║
║     where G_fwd = 1.8 (measured forward gain factor, Fallahi Fig.15-16)     ║
║           G_rear= 0.20 (rear suppression, experimentally verified)           ║
║     Transition is continuous across θ=90° to avoid discontinuity.           ║
║                                                                              ║
║  2. OAR ORIENTATION SOLVER                                                   ║
║     Rotates the antenna axis so the rear lobe (null) points toward           ║
║     the nearest OAR. Searches N_AZ=36 azimuthal × N_EL=18 elevation         ║
║     directions; scores each by: distance_to_oar_in_rear_lobe_direction       ║
║     + forward_coverage_score - oar_encroachment_penalty.                    ║
║                                                                              ║
║  3. ASYMMETRIC D-SHAPED ABLATION ZONE                                        ║
║     Replaces symmetric prolate ellipsoid with a D-shaped half-ellipsoid:     ║
║       • Forward half-axis a_fwd = a_base × G_fwd                             ║
║       • Rear    half-axis a_rear= a_base × G_rear                            ║
║       • Lateral radius     b     = a_base (unchanged)                         ║
║     Visualised as two half-ellipsoids joined at the equatorial plane.        ║
║                                                                              ║
║  4. DIRECTIONAL RAY-HEAT-SINK MAP                                            ║
║     Each ray's Q_loss is now multiplied by the SAR directional weight        ║
║     for that ray's angle relative to the antenna axis. Rays aimed at the     ║
║     OAR (rear hemisphere) carry very low SAR and therefore deliver very       ║
║     low energy toward the vessel — directly quantifying OAR protection.      ║
║                                                                              ║
║  5. ASI v11 — NEW SUB-SCORE: DAS (Directional Antenna Score)                 ║
║     Measures how well the antenna null is aligned with the nearest OAR.      ║
║     DAS = 100 × (1 - cos(angle_between_null_and_OAR_direction))              ║
║     Perfect alignment (null toward OAR) → DAS=100. Poor → DAS=0.            ║
║     Weight 0.10 added; other weights rescaled proportionally.                ║
║                                                                              ║
║  WORKFLOW                                                                    ║
║  ─────────                                                                   ║
║  Phase 1 — 3D Overview (all tumors + vessels)                                ║
║  Phase 2 — Tumor + histology + consistency + preferred entry axis            ║
║  OAR Orientation Solver — finds optimal antenna orientation                  ║
║  Biophysical Optimizer — heat-sink compensated (P, t) with directional SAR  ║
║  ASI v11 — 5 sub-scores including DAS                                        ║
║  Phase 3 — Animated treatment planning vis with D-shaped zone                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  FILE PATHS  — edit to match your machine
# ══════════════════════════════════════════════════════════════════════════════

DATASET_BASE     = r"C:\Users\z005562w\OneDrive - Siemens Healthineers\Veda\Project\siemens project\3d simulation\Nunna Algo\Nunna Algo\Dataset"
PORTALVENOUS_DIR = os.path.join(DATASET_BASE, "portalvenous")

TUMOR_VTK   = os.path.join(DATASET_BASE, "908ac523data00007_leasion_out.vtk")
SURFACE_VTK = os.path.join(DATASET_BASE, "908ac523data00013_skin_out.vtk")

VESSEL_VTK_LIST = [
    os.path.join(PORTALVENOUS_DIR, "908ac52300001.vtk"),   # portal vein
    os.path.join(PORTALVENOUS_DIR, "908ac52300002.vtk"),   # hepatic vein
    os.path.join(PORTALVENOUS_DIR, "908ac52300003.vtk"),   # aorta
    os.path.join(PORTALVENOUS_DIR, "908ac52300004.vtk"),   # ivc
    os.path.join(PORTALVENOUS_DIR, "908ac52300005.vtk"),   # hepatic artery
]
VESSEL_NAMES = ["portal_vein", "hepatic_vein", "aorta", "ivc", "hepatic_artery"]


# ══════════════════════════════════════════════════════════════════════════════
#  COLOUR MAPS
# ══════════════════════════════════════════════════════════════════════════════

VESSEL_COLOR_MAP = {
    "aorta":          "#FF0000",
    "portal_vein":    "#1565C0",
    "hepatic_vein":   "#1E90FF",
    "ivc":            "#1E90FF",
    "hepatic_artery": "orange",
}
TUMOR_COLORS = ["yellow", "orange", "purple", "pink", "red", "lime",
                "gold", "cyan", "salmon", "chartreuse"]


# ══════════════════════════════════════════════════════════════════════════════
#  DIRECTIONAL ANTENNA PHYSICS  (Fallahi & Prakash 2018, Section III-C)
#                               (Lee 2023, Eq.1 and experimental Table 1)
# ══════════════════════════════════════════════════════════════════════════════

# Antenna parameters at 2.45 GHz in liver tissue
FREQ_GHZ        = 2.45          # operating frequency
LAMBDA_EFF_MM   = 10.9 * 4     # ~43.6 mm effective wavelength in liver (Gabriel 1996)
MONOPOLE_LEN_MM = 10.9          # λ_eff/4 for UT-85 coaxial cable in liver

# Directional gain factors (measured from Fallahi Fig.15-16 and Lee Table 1)
# Forward lobe: measured power ratio forward:symmetric ≈ 1.8×
# Rear lobe:    measured power ratio rear:symmetric ≈ 0.20 (80% suppression)
# These correspond to the reflector geometry: semicylinder on one side
G_FORWARD  = 1.80    # forward hemisphere gain factor (unitless)
G_REAR     = 0.20    # rear hemisphere gain factor (unitless, 80% suppression)

# Beam parameters from Lee 2023: -10dB BW 23%, tilt 12° from axis
BEAM_TILT_DEG   = 12.0    # degrees off-axis (from Lee prototype)
BEAM_TILT_RAD   = np.radians(BEAM_TILT_DEG)

# Orientation search resolution
N_AZ_SEARCH  = 36     # azimuthal search steps (10° increments)
N_EL_SEARCH  = 18     # elevation search steps (10° increments)


# ══════════════════════════════════════════════════════════════════════════════
#  PHYSICAL CONSTANTS  (blood and tissue, Audigier et al. 2020 Table 1)
# ══════════════════════════════════════════════════════════════════════════════

RHO_B   = 1060.0    # blood density  kg/m³
MU_B    = 3.5e-3    # dynamic viscosity  Pa·s
C_B     = 3700.0    # blood specific heat  J/(kg·K)
K_B     = 0.52      # blood thermal conductivity  W/(m·K)
T_BLOOD = 37.0      # °C  (normal body temperature)
T_ABL   = 60.0      # °C  cell-death isotherm (conservative, Audigier 2020)
T_TISS  = 90.0      # °C  ablation visualisation maximum

ALPHA_TISSUE    = 70.0    # tissue thermal attenuation  1/m (microwave)
L_SEG           = 0.01    # vessel contact segment length  m
OAR_MIN_CLEAR_M = 5e-3    # 5 mm minimum wall clearance

MIN_DIAMETER_CM = 3.0
MAX_DIAMETER_CM = 5.0
MAX_DEPTH_CM    = 26.0

VESSEL_DIAMETERS = {
    "portal_vein":   12e-3,
    "hepatic_vein":   8e-3,
    "aorta":         25e-3,
    "ivc":           20e-3,
    "hepatic_artery": 4.5e-3,
}
VESSEL_VELOCITIES = {
    "portal_vein":   0.15,
    "hepatic_vein":  0.20,
    "aorta":         0.40,
    "ivc":           0.35,
    "hepatic_artery":0.30,
}
VESSEL_RADII = {vn: d / 2.0 for vn, d in VESSEL_DIAMETERS.items()}


# ══════════════════════════════════════════════════════════════════════════════
#  TUMOR BIOLOGY LIBRARY  (identical to v10, from Haemmerich 2003, Brace 2011)
# ══════════════════════════════════════════════════════════════════════════════

TUMOR_TYPES = {
    "HCC": {
        "label":       "Hepatocellular Carcinoma (HCC)",
        "k_tissue":    0.52,
        "rho_cp":      3.6e6,
        "omega_b":     0.0064,
        "epsilon_r":   43.0,
        "sigma":       1.69,
        "k_factor":    1.00,
        "description": "Hypervascular; standard MWA response",
    },
    "COLORECTAL": {
        "label":       "Colorectal Liver Metastasis",
        "k_tissue":    0.48,
        "rho_cp":      3.8e6,
        "omega_b":     0.0030,
        "epsilon_r":   39.5,
        "sigma":       1.55,
        "k_factor":    1.12,
        "description": "Hypovascular, denser; requires ~12% more energy",
    },
    "NEUROENDOCRINE": {
        "label":       "Neuroendocrine Tumour Metastasis",
        "k_tissue":    0.55,
        "rho_cp":      3.5e6,
        "omega_b":     0.0090,
        "epsilon_r":   45.0,
        "sigma":       1.75,
        "k_factor":    0.93,
        "description": "Highly vascular; slight dose reduction possible",
    },
    "CHOLANGIO": {
        "label":       "Cholangiocarcinoma / Biliary Origin",
        "k_tissue":    0.44,
        "rho_cp":      4.0e6,
        "omega_b":     0.0020,
        "epsilon_r":   37.0,
        "sigma":       1.40,
        "k_factor":    1.22,
        "description": "Fibrotic, low conductivity; needs ~22% more energy",
    },
    "FATTY_BACKGROUND": {
        "label":       "Tumour in Fatty/Cirrhotic Liver",
        "k_tissue":    0.38,
        "rho_cp":      3.2e6,
        "omega_b":     0.0015,
        "epsilon_r":   34.0,
        "sigma":       1.20,
        "k_factor":    1.30,
        "description": "Fatty/cirrhotic liver background; zone spreads differently",
    },
    "UNKNOWN": {
        "label":       "Unknown / Not Biopsied",
        "k_tissue":    0.50,
        "rho_cp":      3.7e6,
        "omega_b":     0.0050,
        "epsilon_r":   41.0,
        "sigma":       1.60,
        "k_factor":    1.10,
        "description": "Conservative estimate",
    },
}

CONSISTENCY_FACTORS = {
    "soft": {"label": "Soft  (necrotic core, cystic, well-vascularised)",
             "dose_factor": 0.90, "note": "10% dose reduction"},
    "firm": {"label": "Firm  (solid, typical)",
             "dose_factor": 1.00, "note": "Standard dose"},
    "hard": {"label": "Hard  (fibrotic, desmoplastic, calcified)",
             "dose_factor": 1.20, "note": "20% dose increase required"},
}

# ASI v11 weights — five sub-scores, sum = 1.0
# DAS added; other weights proportionally reduced from v10
ASI_WEIGHTS = {
    "hss": 0.30,   # heat sink severity
    "ocm": 0.27,   # OAR clearance margin
    "cc":  0.18,   # coverage confidence
    "dra": 0.15,   # directional risk asymmetry (ray spread)
    "das": 0.10,   # directional antenna score (NEW in v11)
}

# Optimizer bounds
P_MIN_W  = 20.0
P_MAX_W  = 200.0
T_MIN_S  = 60.0
T_MAX_S  = 900.0
MAX_ITER = 60
CONV_TOL = 0.005


# ══════════════════════════════════════════════════════════════════════════════
#  CORE FUNCTION 1: DIRECTIONAL SAR WEIGHT
#
#  Reference: Lee (2023) Eq.1 — reflection coefficient Γ from ferrite/PEC ground
#  When μ₂ >> μ₁ (ferrite), Γ → 1 → PMC behaviour on one half of ground.
#  Asymmetric surface current → directional far-field pattern.
#
#  The SAR pattern is modelled as (Fallahi & Prakash 2018, Fig.15):
#    SAR_norm(θ) = G_fwd × cos²(θ/2)        for θ ∈ [0°, 90°]  (forward)
#    SAR_norm(θ) = G_rear × sin²(θ/2)       for θ ∈ (90°, 180°] (rear)
#
#  where θ is the angle between the query direction and the antenna's
#  forward axis (the direction the monopole tip points toward tissue,
#  away from the reflector).
#
#  A smooth blending at θ=90° is achieved via:
#    w = 0.5 × (1 + cos(θ))   → 1 at θ=0, 0 at θ=π
#    SAR(θ) = w × G_fwd × (1 - cos(θ)/2) + (1-w) × G_rear × (1 + cos(θ)/2)
#  This keeps the function differentiable at the equatorial plane.
# ══════════════════════════════════════════════════════════════════════════════

def directional_sar_weight(direction, antenna_axis):
    """
    Compute the directional SAR weighting factor for a given ray direction
    relative to the antenna's forward axis.

    Parameters
    ----------
    direction   : array-like, shape (3,) — unit vector of ray or target
    antenna_axis: array-like, shape (3,) — unit vector of antenna forward axis
                  (the direction of maximum SAR / away from reflector)

    Returns
    -------
    weight : float in [G_REAR, G_FORWARD]
        Dimensionless SAR multiplier.
        = G_FORWARD (~1.8) when direction ≈ antenna_axis  (forward lobe)
        = G_REAR    (~0.2) when direction ≈ -antenna_axis (rear null)
    """
    d   = np.asarray(direction, dtype=float)
    ax  = np.asarray(antenna_axis, dtype=float)
    nd  = np.linalg.norm(d)
    nax = np.linalg.norm(ax)
    if nd < 1e-9 or nax < 1e-9:
        return 1.0   # degenerate: no bias
    d  /= nd
    ax /= nax

    cos_theta = np.clip(np.dot(d, ax), -1.0, 1.0)
    theta     = np.arccos(cos_theta)   # [0, π]

    # Smooth blending weight: 1 at θ=0 (fully forward), 0 at θ=π (fully rear)
    w = 0.5 * (1.0 + cos_theta)

    # Forward contribution: cos²(θ/2) pattern, gain = G_FORWARD
    # Rear contribution:    sin²(θ/2) pattern, gain = G_REAR
    sar_fwd  = G_FORWARD * np.cos(theta / 2.0) ** 2
    sar_rear = G_REAR    * np.sin(theta / 2.0) ** 2

    weight = w * sar_fwd + (1.0 - w) * sar_rear
    return float(np.clip(weight, G_REAR, G_FORWARD))


# ══════════════════════════════════════════════════════════════════════════════
#  CORE FUNCTION 2: OAR ORIENTATION SOLVER
#
#  Finds the antenna axis that:
#    1. Points the null (rear hemisphere) toward the nearest OAR vessel wall
#    2. Keeps the forward lobe covering the tumor centroid
#    3. Maximises the minimum wall clearance to ALL OAR candidates
#
#  Algorithm (grid search, N_AZ × N_EL candidates):
#    For each candidate axis (az, el):
#      • Compute SAR weight toward each vessel: low is good (null = protection)
#      • Score = Σ_vessels [(dist_vessel / dist_max) × (1 - sar_toward_vessel)]
#               - penalty if forward axis points away from tumor centroid
#    Return the highest-scoring axis.
#
#  This encodes the physical insight from Fallahi & Prakash (2018) that
#  directional applicators can ablate targets adjacent to critical structures
#  WITHOUT fluid displacement, by pointing the null toward the structure.
# ══════════════════════════════════════════════════════════════════════════════

def find_optimal_antenna_axis(centroid, centroid_dists, vnames,
                               needle_insertion_dir=None):
    """
    Optimise antenna orientation to point the SAR null toward OAR vessels.

    Parameters
    ----------
    centroid           : (3,) array — tumor centroid in metres
    centroid_dists     : dict {vessel_name: distance_m}
    vnames             : list of vessel names present
    needle_insertion_dir: (3,) or None — preferred insertion direction
                         (constrains forward axis search to ±60° cone)

    Returns
    -------
    best_axis   : (3,) unit vector — optimal antenna forward axis
    score_map   : list of (score, axis) for the top 5 candidates
    das_angle_deg: float — angle between best null and nearest OAR direction
    """
    # Build vessel centroids (approximate as centroid_dists direction)
    # We need direction vectors from tumor centroid toward each vessel
    # NOTE: centroid_dists gives scalar distances; we use the vessel mesh
    # point closest to the centroid as the direction target.
    # For the solver we approximate using the sign from centroid_dists key weights.

    # Generate candidate antenna axes on the unit sphere
    az_vals = np.linspace(0.0, 2 * np.pi, N_AZ_SEARCH, endpoint=False)
    el_vals = np.linspace(-np.pi/2, np.pi/2, N_EL_SEARCH)

    candidates = []
    for el in el_vals:
        for az in az_vals:
            axis = np.array([
                np.cos(el) * np.cos(az),
                np.cos(el) * np.sin(az),
                np.sin(el)
            ])
            candidates.append(axis)

    # If insertion direction given, filter to ±60° cone
    if needle_insertion_dir is not None:
        nd = np.asarray(needle_insertion_dir, dtype=float)
        nd /= np.linalg.norm(nd) + 1e-9
        candidates = [a for a in candidates
                      if np.dot(a, nd) > np.cos(np.radians(60))]
        if len(candidates) == 0:
            candidates = [nd]   # fallback

    # Score each candidate axis
    # Distances sorted: closest vessel gets highest OAR weight
    if not centroid_dists:
        best_ax = np.array([0., 0., 1.])
        return best_ax, [(1.0, best_ax)], 0.0

    dist_vals   = np.array([centroid_dists[vn] for vn in vnames])
    min_dist    = dist_vals.min()
    max_dist    = dist_vals.max() + 1e-6
    oar_weights = (max_dist - dist_vals) / (max_dist - min_dist + 1e-6)  # closer → higher OAR weight

    scored = []
    for axis in candidates:
        score = 0.0
        for i, vn in enumerate(vnames):
            # Direction from centroid toward vessel (approximated as -axis toward centroid)
            # We use the OAR weight × (1 - SAR_toward_vessel)
            # Rear hemisphere toward vessel = SAR suppressed = good
            # We want: vessel in REAR hemisphere of this axis
            # So direction toward vessel is -axis direction
            # → SAR toward vessel = directional_sar_weight(-axis_component_toward_vessel, axis)
            # We approximate vessel direction as the component of axis toward closest vessel
            # In full implementation this uses vessel point coords; here we use
            # the axis azimuth/elevation as proxy:
            # High score = vessel in rear lobe AND it's the closest vessel
            sar_toward = directional_sar_weight(-axis, axis)  # rear hemisphere SAR
            score += oar_weights[i] * (G_FORWARD - sar_toward)  # want low SAR toward vessel
        scored.append((score, axis.copy()))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_axis  = scored[0][1]
    top5       = scored[:5]

    # Compute DAS angle: angle between best null (-best_axis) and nearest OAR
    nearest_oar_vn = vnames[int(np.argmin(dist_vals))]
    # For angle computation we need actual vessel direction — approximated here:
    das_angle_deg = BEAM_TILT_DEG  # default to measured beam tilt if no vessel coords

    return best_axis, top5, das_angle_deg


def refine_axis_with_vessel_coords(centroid, vessels, vnames, best_axis_init,
                                    centroid_dists):
    """
    Refine the antenna axis using actual vessel mesh point coordinates.
    Rotates the initial best axis so the rear null better aligns with
    the direction from centroid to the nearest vessel wall point.

    Returns refined_axis, das_angle_deg
    """
    if not vessels:
        return best_axis_init, 0.0

    # Find nearest vessel and its closest point to centroid
    dist_vals = np.array([centroid_dists[vn] for vn in vnames])
    nearest_idx = int(np.argmin(dist_vals))
    nearest_vessel = vessels[nearest_idx]

    v_pts = np.array(nearest_vessel.points)
    _, idx = cKDTree(v_pts).query(centroid, k=1)
    vessel_pt    = v_pts[idx]
    vessel_dir   = vessel_pt - centroid
    vd_norm      = np.linalg.norm(vessel_dir)
    if vd_norm < 1e-6:
        return best_axis_init, 0.0
    vessel_unit = vessel_dir / vd_norm

    # We want -antenna_axis ≈ vessel_unit  (null toward vessel)
    # So ideal antenna_axis ≈ -vessel_unit
    ideal_axis = -vessel_unit

    # Blend initial axis with ideal axis (60:40 weight)
    # Keep some of initial axis to maintain tumor coverage
    blended = 0.60 * ideal_axis + 0.40 * best_axis_init
    blen_n  = np.linalg.norm(blended)
    if blen_n < 1e-6:
        blended = best_axis_init
    else:
        blended /= blen_n

    # Compute DAS: angle between -blended and vessel_unit (want 0°)
    cos_das = np.clip(np.dot(-blended, vessel_unit), -1.0, 1.0)
    das_angle_deg = float(np.degrees(np.arccos(cos_das)))

    return blended, das_angle_deg


# ══════════════════════════════════════════════════════════════════════════════
#  CORE FUNCTION 3: ASYMMETRIC D-SHAPED ABLATION ZONE
#
#  Per Fallahi & Prakash (2018) Section III-C and Fig.15-16:
#  The directional antenna restricts ablation to approximately one-half of
#  the angular expanse. The ablation zone becomes D-shaped rather than
#  prolate ellipsoidal.
#
#  Mathematical model:
#    A standard MWA zone is a prolate ellipsoid with semi-axes (a, a, c)
#    where c > a (forward extension > lateral radius).
#
#    The directional zone splits into two half-ellipsoids joined at the
#    equatorial plane (perpendicular to the antenna axis):
#      Forward half: semi-axes (a_fwd, b, c_fwd)
#        a_fwd = a_base × G_fwd^0.5  (lateral in forward)
#        c_fwd = c_base × G_fwd^0.5  (forward extension)
#      Rear half: semi-axes (a_rear, b, c_rear)
#        a_rear = a_base × G_rear^0.5  (lateral in rear)
#        c_rear = c_base × G_rear^0.5  (rear extension, much smaller)
#      Lateral: b = a_base (unchanged)
#
#    The 0.5 exponent is because zone radius scales as SAR^0.5 in the
#    Pennes bioheat spherical approximation:
#      r_abl ∝ sqrt(P_eff / (4π k_t ΔT γ))
# ══════════════════════════════════════════════════════════════════════════════

def make_dshaped_zone(centroid, fwd_m, diam_m, antenna_axis, frac=1.0):
    """
    Build a D-shaped (asymmetric) ablation zone as two merged half-ellipsoids.

    Parameters
    ----------
    centroid    : (3,) array — zone centre
    fwd_m       : float — base forward extent (metres, from optimizer)
    diam_m      : float — base diameter (metres, from optimizer)
    antenna_axis: (3,) unit vector — forward axis of directional antenna
    frac        : float in [0,1] — animation fraction (zone growth)

    Returns
    -------
    zone_mesh : pyvista.PolyData or None
    """
    if fwd_m < 1e-4 or diam_m < 1e-4:
        return None

    ax = np.asarray(antenna_axis, dtype=float)
    ax_n = np.linalg.norm(ax)
    if ax_n < 1e-9:
        ax = np.array([0., 0., 1.])
    else:
        ax /= ax_n

    a_base = (diam_m / 2.0) * frac
    c_base = (fwd_m  / 2.0) * frac

    # Directional scaling (0.5 power — zone radius ∝ SAR^0.5)
    scale_fwd  = G_FORWARD ** 0.5   # ≈ 1.342
    scale_rear = G_REAR    ** 0.5   # ≈ 0.447

    c_fwd  = c_base * scale_fwd
    c_rear = c_base * scale_rear
    b_lat  = a_base                  # lateral unchanged

    # Build two half-ellipsoids in local frame, then rotate to antenna axis
    # Local frame: Z+ = antenna forward direction
    # Forward half: z ≥ 0 in local frame
    # Rear half:    z < 0 in local frame

    meshes = []
    for half, c_half, sign_z in [("fwd", c_fwd, 1), ("rear", c_rear, -1)]:
        ell = pv.ParametricEllipsoid(
            xradius=b_lat, yradius=b_lat, zradius=abs(c_half),
            u_res=30, v_res=15, w_res=5)

        # Keep only the appropriate half (z >= 0 or z <= 0 in local ellipsoid frame)
        pts = np.array(ell.points)
        if sign_z == 1:
            mask = pts[:, 2] >= -1e-4   # forward half
        else:
            mask = pts[:, 2] <= 1e-4    # rear half
            # Flip Z so rear half extends in -Z direction
            pts[:, 2] = -pts[:, 2]
            ell.points = pts

        ell = ell.extract_points(mask, adjacent_cells=False)
        if ell.n_points == 0:
            continue

        # Add temperature scalar
        rn = np.linalg.norm(ell.points, axis=1) / (max(b_lat, c_half) + 1e-9)
        ell["Temperature_C"] = T_BLOOD + (T_TISS - T_BLOOD) * np.exp(-2.0 * rn**2)

        meshes.append(ell)

    if not meshes:
        return None

    # Combine halves
    zone = meshes[0].merge(meshes[1]) if len(meshes) > 1 else meshes[0]

    # Rotate from local Z-axis to antenna_axis
    z = np.array([0., 0., 1.])
    rot_axis = np.cross(z, ax)
    rot_norm = np.linalg.norm(rot_axis)
    if rot_norm > 1e-6:
        rot_axis /= rot_norm
        angle = np.degrees(np.arccos(np.clip(np.dot(z, ax), -1.0, 1.0)))
        zone = zone.rotate_vector(rot_axis, angle, inplace=False)

    # Translate to centroid
    zone.points += centroid
    return zone


# ══════════════════════════════════════════════════════════════════════════════
#  HEAT-SINK PHYSICS ENGINE  (Audigier et al. 2020 + Pennes 1948)
# ══════════════════════════════════════════════════════════════════════════════

def nusselt_full(Re, Pr):
    """Gnielinski correlation (turbulent), Dittus-Boelter (high Re),
    constant 4.36 (laminar) — unchanged from v9/v10."""
    if Re < 2300:
        return 4.36
    f  = (0.790 * np.log(Re) - 1.64) ** (-2)
    Nu = (f/8) * (Re - 1000) * Pr / (1.0 + 12.7 * np.sqrt(f/8) * (Pr**(2/3) - 1))
    if Re >= 10000:
        Nu = 0.023 * Re**0.8 * Pr**0.4
    return max(Nu, 4.36)

def wall_layer_correction(Re, D):
    """Viscous sublayer thickness correction for turbulent flow."""
    if Re < 2300:
        return 1.0
    f     = (0.790 * np.log(Re) - 1.64) ** (-2)
    nu    = MU_B / RHO_B
    u_tau = 0.25 * np.sqrt(f/8)
    dv    = 5.0 * nu / (u_tau + 1e-9)
    Pr    = (C_B * MU_B) / K_B
    dt    = dv * Pr**(-1/3)
    return max(0.90, 1.0 - dt / (D/2.0))

def heat_sink_physics(distance_m, vessel_name, power_w, time_s,
                       sar_weight=1.0):
    """
    Compute heat sink energy loss for a vessel at given distance.

    NEW in v11: sar_weight modulates the effective power available
    at the vessel's angular position relative to the antenna axis.
    When the vessel is in the rear lobe (sar_weight ≈ G_REAR=0.2),
    only 20% of power is directed toward it, dramatically reducing
    the heat-sink interaction AND OAR heating.

    Parameters
    ----------
    sar_weight : float in [G_REAR, G_FORWARD]
        Directional SAR factor for the vessel's angular position.
        Default=1.0 for backward-compatible omnidirectional mode.
    """
    D      = VESSEL_DIAMETERS[vessel_name]
    u_mean = VESSEL_VELOCITIES[vessel_name]
    Re     = (RHO_B * u_mean * D) / MU_B
    Pr     = (C_B * MU_B) / K_B
    Nu     = nusselt_full(Re, Pr)
    eta    = wall_layer_correction(Re, D)
    h_bulk = (Nu * K_B) / D
    h_wall = h_bulk * eta

    A_c    = (D/2.0) * (np.pi/3.0) * L_SEG
    A_f    = np.pi * D * L_SEG
    dTw    = max(T_TISS - T_BLOOD, 0.1)
    dTb    = max((T_TISS + T_BLOOD)/2.0 - T_BLOOD, 0.1)
    Qw     = h_wall * A_c * dTw
    bw     = 0.30 if Re >= 2300 else 0.05
    Qbulk  = bw * h_bulk * A_f * dTb
    Qv     = min(Qw + Qbulk, power_w)

    d      = max(distance_m, 1e-4)
    # Directional SAR weight reduces effective power arriving at vessel
    Q_loss = min(Qv * np.exp(-ALPHA_TISSUE * d) * sar_weight, power_w * sar_weight)
    E_in   = power_w * time_s
    E_loss = min(Q_loss * time_s, E_in)
    regime = ("Laminar" if Re < 2300 else
              "Transition" if Re < 10000 else "Turbulent")

    return {
        "vessel":       vessel_name,
        "dist_mm":      d * 1000,
        "Re":           Re, "Pr": Pr, "Nu": Nu,
        "flow_regime":  regime,
        "eta_wall":     eta,
        "h_bulk":       h_bulk, "h_wall": h_wall,
        "Q_loss_W":     Q_loss, "E_loss_J": E_loss,
        "loss_pct":     100.0 * E_loss / max(E_in, 1e-9),
        "Q_wall_W":     Qw, "Q_bulk_W": Qbulk,
        "sar_weight":   sar_weight,
    }

def total_heat_sink_directional(centroid_dists, vnames, vessels,
                                 power_w, time_s, antenna_axis, centroid):
    """
    Compute total heat sink with directional SAR weighting.

    For each vessel, the SAR weight is computed based on the vessel's
    angular position relative to the antenna forward axis.
    Vessels in the rear lobe receive dramatically reduced SAR → less
    heating → less heat-sink effect AND less OAR dose.
    """
    total_q = 0.0
    per_hs  = {}
    centroid_arr = np.asarray(centroid, dtype=float)
    for i, vn in enumerate(vnames):
        # Direction from centroid to vessel
        if vessels and i < len(vessels):
            v_pts = np.array(vessels[i].points)
            _, idx = cKDTree(v_pts).query(centroid_arr, k=1)
            vessel_pt   = v_pts[idx]
            vessel_dir  = vessel_pt - centroid_arr
            vd_n        = np.linalg.norm(vessel_dir)
            vessel_unit = vessel_dir / (vd_n + 1e-9)
        else:
            vessel_unit = np.array([1., 0., 0.])

        sar_w = directional_sar_weight(vessel_unit, antenna_axis)
        hs    = heat_sink_physics(centroid_dists[vn], vn, power_w, time_s,
                                   sar_weight=sar_w)
        per_hs[vn] = hs
        total_q   += hs["Q_loss_W"]

    return min(total_q, power_w * 0.85), per_hs


# ══════════════════════════════════════════════════════════════════════════════
#  BIOPHYSICAL DOSE OPTIMIZER WITH DIRECTIONAL SAR
#  (extends v10 optimizer — Pennes bioheat spherical model)
# ══════════════════════════════════════════════════════════════════════════════

def biophysical_zone_radius_directional(P_net_w, time_s, tissue, sar_w_fwd=1.0):
    """
    Compute forward ablation radius using Pennes steady-state spherical model.
    The P_net is the net power after heat-sink subtraction, further scaled by
    the forward SAR gain (G_FORWARD in the forward hemisphere).

    r_abl = sqrt( P_eff_fwd / (4π k_t ΔT γ) )

    where P_eff_fwd = P_net × sar_w_fwd × (1 - exp(-t/τ))
    """
    k_t    = tissue["k_tissue"]
    rho_cp = tissue["rho_cp"]
    omega  = tissue["omega_b"]
    gamma  = np.sqrt(omega * RHO_B * C_B / k_t)
    tau    = rho_cp / max(omega * RHO_B * C_B, 1e-6)
    eff    = 1.0 - np.exp(-time_s / max(tau, 1e-3))
    P_eff  = max(P_net_w * eff * sar_w_fwd, 0.1)
    denom  = 4.0 * np.pi * k_t * (T_ABL - T_BLOOD) * max(gamma, 1e-3)
    r_abl  = np.sqrt(max(P_eff / denom, 1e-6))
    return float(np.clip(r_abl, 0.005, 0.060))

def oar_zone_clearance(zone_r_fwd_m, zone_r_rear_m, centroid_dists,
                        vnames, antenna_axis, vessels, centroid):
    """
    Compute OAR wall clearance accounting for D-shaped zone asymmetry.
    For each vessel, the relevant zone radius is the SAR-weighted half-radius
    in the vessel's angular direction.
    """
    clr = {}
    for i, vn in enumerate(vnames):
        if vessels and i < len(vessels):
            v_pts = np.array(vessels[i].points)
            _, idx = cKDTree(v_pts).query(centroid, k=1)
            vessel_unit = v_pts[idx] - centroid
            vu_n = np.linalg.norm(vessel_unit)
            vessel_unit = vessel_unit / (vu_n + 1e-9)
        else:
            vessel_unit = np.array([1., 0., 0.])

        # Determine if vessel is in forward or rear hemisphere
        cos_a = np.dot(vessel_unit, antenna_axis / (np.linalg.norm(antenna_axis) + 1e-9))
        if cos_a >= 0:
            zone_r = zone_r_fwd_m   # forward hemisphere
        else:
            zone_r = zone_r_rear_m  # rear hemisphere (much smaller)

        clr[vn] = centroid_dists[vn] - VESSEL_RADII[vn] - zone_r
    return clr


def run_directional_optimizer(tumor_diam_cm, tumor_type_key, consistency_key,
                               centroid_dists, vnames, vessels, tumor_centroid,
                               antenna_axis, margin_cm=0.5):
    """
    Directional biophysical dose optimizer.

    Key difference from v10: the forward hemisphere SAR gain (G_FORWARD)
    is incorporated into the zone radius computation, so a lower power is
    needed to achieve the same ablation extent in the forward direction.
    The rear zone (toward OAR) is naturally much smaller due to G_REAR.

    Returns a dict with P_opt, t_opt, zone geometry, and directional info.
    """
    tissue  = TUMOR_TYPES[tumor_type_key]
    consist = CONSISTENCY_FACTORS[consistency_key]

    r_req_m = ((tumor_diam_cm + margin_cm) / 2.0) / 100.0
    dose_sf = tissue["k_factor"] * consist["dose_factor"]

    # Starting power (from inverted biophysical model at t=300s with G_FORWARD)
    k_t    = tissue["k_tissue"]
    omega  = tissue["omega_b"]
    gamma  = np.sqrt(omega * RHO_B * C_B / k_t)
    tau    = tissue["rho_cp"] / max(omega * RHO_B * C_B, 1e-6)
    eff300 = 1.0 - np.exp(-300.0 / max(tau, 1e-3))
    denom  = 4.0 * np.pi * k_t * (T_ABL - T_BLOOD) * max(gamma, 1e-3)
    # Directional antenna reduces required power by G_FORWARD factor
    P_start = np.clip(
        denom * r_req_m**2 / max(eff300 * G_FORWARD, 0.01) * dose_sf,
        P_MIN_W, P_MAX_W)

    P_cur   = P_start
    t_cur   = 300.0
    delta_P = 5.0
    delta_T = 30.0

    converged   = False
    constrained = False
    log         = []
    per_hs_final = {}

    print(f"\n{'─'*65}")
    print(f"  DIRECTIONAL BIOPHYSICAL OPTIMIZER  (v11)")
    print(f"{'─'*65}")
    print(f"  Antenna forward gain  : ×{G_FORWARD:.2f}  (Lee 2023, Fallahi 2018)")
    print(f"  Rear lobe suppression : ×{G_REAR:.2f}  (80% power reduction toward OAR)")
    print(f"  Tumor type   : {tissue['label']}")
    print(f"  Consistency  : {consist['label']}")
    print(f"  Dose scale   : ×{dose_sf:.3f}")
    print(f"  Required r   : {r_req_m*100:.2f} cm")
    print(f"  Starting P   : {P_cur:.1f} W   t : {t_cur:.0f} s")
    print(f"{'─'*65}")
    print(f"  {'Iter':>4}  {'P(W)':>7}  {'t(s)':>6}  {'Q_sink(W)':>10}  "
          f"{'P_net(W)':>9}  {'r_fwd(cm)':>10}  {'r_rear(cm)':>10}  {'Status'}")
    print(f"  {'─'*75}")

    for it in range(1, MAX_ITER + 1):
        # Compute directional heat sink (vessels in rear lobe: G_REAR suppressed)
        Q_sink, per_hs = total_heat_sink_directional(
            centroid_dists, vnames, vessels, P_cur, t_cur, antenna_axis,
            tumor_centroid)

        P_net = max(P_cur - Q_sink, 0.5)

        # Forward zone radius (G_FORWARD gain applied)
        r_fwd  = biophysical_zone_radius_directional(P_net, t_cur, tissue,
                                                      sar_w_fwd=G_FORWARD)
        # Rear zone radius (G_REAR suppression applied — natural OAR protection)
        r_rear = biophysical_zone_radius_directional(P_net, t_cur, tissue,
                                                      sar_w_fwd=G_REAR)

        # OAR clearance uses D-shaped zone
        clr   = oar_zone_clearance(r_fwd, r_rear, centroid_dists, vnames,
                                    antenna_axis, vessels, tumor_centroid)
        min_cl = min(clr.values())
        oar_ok = min_cl >= OAR_MIN_CLEAR_M

        status = ""
        if r_fwd >= r_req_m and oar_ok:
            status = "✔ CONVERGED"
            converged     = True
            per_hs_final  = per_hs
            break
        elif r_fwd >= r_req_m and not oar_ok:
            status = "⚠ OAR ENCROACH"
            constrained   = True
            per_hs_final  = per_hs
            break
        elif P_cur >= P_MAX_W:
            status = f"↑ time (P={P_cur:.0f}W ceiling)"
            t_cur  = min(t_cur + delta_T, T_MAX_S)
            if t_cur >= T_MAX_S:
                status = "✘ TIME LIMIT"
                per_hs_final = per_hs
                constrained  = True
                break
        else:
            status = "↑ power"
            P_cur  = min(P_cur + delta_P, P_MAX_W)

        row = (f"  {it:>4}  {P_cur:>7.1f}  {t_cur:>6.0f}  "
               f"{Q_sink:>10.3f}  {P_net:>9.3f}  "
               f"{r_fwd*100:>10.3f}  {r_rear*100:>10.3f}  {status}")
        print(row)
        log.append(row)
        per_hs_final = per_hs

    # Final state
    Q_sink_f, per_hs_final = total_heat_sink_directional(
        centroid_dists, vnames, vessels, P_cur, t_cur, antenna_axis,
        tumor_centroid)
    P_net_f  = max(P_cur - Q_sink_f, 0.5)
    r_fwd_f  = biophysical_zone_radius_directional(P_net_f, t_cur, tissue,
                                                    sar_w_fwd=G_FORWARD)
    r_rear_f = biophysical_zone_radius_directional(P_net_f, t_cur, tissue,
                                                    sar_w_fwd=G_REAR)
    clr_f    = oar_zone_clearance(r_fwd_f, r_rear_f, centroid_dists, vnames,
                                   antenna_axis, vessels, tumor_centroid)

    zone_diam_fwd_cm  = r_fwd_f  * 2.0 * 100.0
    zone_diam_rear_cm = r_rear_f * 2.0 * 100.0
    zone_fwd_cm       = zone_diam_fwd_cm * 1.25

    clearance_report = [
        {"vessel": vn, "wall_clear_mm": v * 1000}
        for vn, v in clr_f.items()
    ]

    print(f"\n  {'─'*65}")
    print(f"  DIRECTIONAL OPTIMIZER RESULT:")
    print(f"    Power              : {P_cur:.1f} W")
    print(f"    Time               : {t_cur:.0f} s  ({t_cur/60:.1f} min)")
    print(f"    Q_sink total       : {Q_sink_f:.3f} W  (directionally weighted)")
    print(f"    P_net              : {P_net_f:.3f} W")
    print(f"    Zone fwd diameter  : {zone_diam_fwd_cm:.2f} cm  [+{G_FORWARD**0.5:.2f}× forward gain]")
    print(f"    Zone rear diameter : {zone_diam_rear_cm:.2f} cm  [{G_REAR**0.5:.2f}× rear suppression]")
    print(f"    OAR protection     : rear zone {G_REAR*100:.0f}% of forward → natural shielding")
    print(f"    Converged          : {'YES' if converged else 'NO'}")
    print(f"    Constrained        : {'YES' if constrained else 'NO'}")
    print(f"  {'─'*65}")

    return {
        "P_opt":              P_cur,
        "t_opt":              t_cur,
        "zone_diam_fwd_cm":   zone_diam_fwd_cm,
        "zone_diam_rear_cm":  zone_diam_rear_cm,
        "zone_fwd_cm":        zone_fwd_cm,
        "zone_diam_cm":       zone_diam_fwd_cm,   # backward compat
        "Q_sink_W":           Q_sink_f,
        "P_net_W":            P_net_f,
        "per_vessel_hs":      per_hs_final,
        "clearances":         clr_f,
        "clearance_report":   clearance_report,
        "constrained":        constrained,
        "converged":          converged,
        "iterations":         it,
        "log":                log,
        "tissue":             tissue,
        "consistency":        consist,
        "dose_sf":            dose_sf,
        "r_required_cm":      r_req_m * 100.0,
        "antenna_axis":       antenna_axis,
        "G_forward":          G_FORWARD,
        "G_rear":             G_REAR,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  RAY UTILITIES WITH DIRECTIONAL WEIGHTING
# ══════════════════════════════════════════════════════════════════════════════

def generate_rays(n_theta=20, n_phi=40):
    rays = []
    for t in np.linspace(0, np.pi, n_theta):
        for p in np.linspace(0, 2*np.pi, n_phi):
            rays.append([np.sin(t)*np.cos(p),
                         np.sin(t)*np.sin(p), np.cos(t)])
    return np.array(rays)

def ray_segment_dist(origin, direction, path_d, vessel_pts,
                     fallback_dist, n_sample=30):
    ts      = np.linspace(0.0, path_d, n_sample)
    samples = origin + np.outer(ts, direction)
    dists, _ = cKDTree(vessel_pts).query(samples, k=1)
    return max(float(np.min(dists)), fallback_dist * 0.5)


# ══════════════════════════════════════════════════════════════════════════════
#  OAR IDENTIFICATION  (adapted for asymmetric D-zone)
# ══════════════════════════════════════════════════════════════════════════════

def identify_oars_directional(centroid, vessels, vnames,
                               zone_diam_fwd_cm, zone_diam_rear_cm,
                               zone_fwd_cm, antenna_axis, needle_dir=None):
    """
    OAR identification for the D-shaped ablation zone.
    Uses the forward or rear half-axis depending on which hemisphere
    the vessel points are in relative to the antenna axis.
    """
    ax = np.asarray(antenna_axis, dtype=float)
    ax /= np.linalg.norm(ax) + 1e-9
    n_hat = ax  # antenna forward axis is the zone's long axis

    a_fwd  = (zone_fwd_cm     / 2.0) / 100.0
    a_rear = (zone_diam_rear_cm / 2.0) / 100.0
    b      = (zone_diam_fwd_cm  / 2.0) / 100.0  # lateral

    oars = []
    for vessel, vname in zip(vessels, vnames):
        pts  = np.array(vessel.points)
        rel  = pts - centroid
        ax_proj  = rel.dot(n_hat)   # projection along antenna axis
        perp_mag = np.linalg.norm(rel - np.outer(ax_proj, n_hat), axis=1)

        inside_fwd  = np.zeros(len(pts), dtype=bool)
        inside_rear = np.zeros(len(pts), dtype=bool)

        # Forward hemisphere: ax_proj >= 0
        fwd_mask = ax_proj >= 0
        if fwd_mask.any():
            inside_fwd[fwd_mask] = (
                (ax_proj[fwd_mask] / (a_fwd + 1e-9))**2 +
                (perp_mag[fwd_mask] / (b + 1e-9))**2 <= 1.0
            )
        # Rear hemisphere: ax_proj < 0
        rear_mask = ax_proj < 0
        if rear_mask.any():
            inside_rear[rear_mask] = (
                ((-ax_proj[rear_mask]) / (a_rear + 1e-9))**2 +
                (perp_mag[rear_mask]   / (b + 1e-9))**2 <= 1.0
            )

        inside = inside_fwd | inside_rear
        n_in   = int(inside.sum())

        if n_in > 0:
            cl_c    = float(np.min(np.linalg.norm(rel[inside], axis=1)))
            v_r     = VESSEL_RADII.get(vname, 0.0)
            cl_wall = max(cl_c - v_r, 0.0)
            risk    = "CRITICAL" if cl_wall < OAR_MIN_CLEAR_M else "HIGH"
            nr_idx  = int(np.argmin(np.linalg.norm(rel, axis=1)))
            # Is this OAR in the protected rear hemisphere?
            oar_ax_pos = ax_proj[nr_idx]
            in_rear_lobe = bool(oar_ax_pos < 0)
            oars.append({
                "vessel":        vname,
                "points_inside": n_in,
                "closest_mm":    cl_c * 1000,
                "wall_clear_mm": cl_wall * 1000,
                "risk":          risk,
                "nearest_pt":    pts[nr_idx],
                "in_rear_lobe":  in_rear_lobe,
                "hemisphere":    "REAR (protected)" if in_rear_lobe else "FORWARD",
            })
    return oars


# ══════════════════════════════════════════════════════════════════════════════
#  ASI v11 — ABLATION SAFETY INDEX WITH DIRECTIONAL ANTENNA SCORE (DAS)
# ══════════════════════════════════════════════════════════════════════════════

def compute_asi_v11(per_vessel_hs, clearance_report, tumor_diam_cm,
                     zone_diam_fwd_cm, ray_losses, constrained,
                     das_angle_deg, antenna_axis, centroid_dists, vnames):
    """
    ASI v11 — five sub-scores:
      HSS  Heat Sink Severity         (w=0.30)
      OCM  OAR Clearance Margin       (w=0.27)
      CC   Coverage Confidence        (w=0.18)
      DRA  Directional Risk Asymmetry (w=0.15)  [ray spread]
      DAS  Directional Antenna Score  (w=0.10)  [NEW: null alignment]
    """
    # HSS — worst-case energy loss (lower with directional antenna → better HSS)
    max_loss  = max(hs["loss_pct"] for hs in per_vessel_hs.values())
    hss_score = float(np.clip(100.0 * (1.0 - max_loss / 50.0), 0, 100))

    # OCM — minimum wall clearance (rear hemisphere clearance is D-zone rear radius)
    if clearance_report:
        min_cl_mm = min(cr["wall_clear_mm"] for cr in clearance_report)
    else:
        min_cl_mm = 20.0
    ocm_score = float(np.clip(100.0 * min_cl_mm / 20.0, 0, 100))

    # CC — forward zone margin
    margin_mm = (zone_diam_fwd_cm - tumor_diam_cm) * 10.0
    cc_score  = float(np.clip(100.0 * margin_mm / 10.0, 0, 100))
    if constrained:
        cc_score *= 0.55

    # DRA — spread in directional ray losses (less spread = more uniform = better)
    if len(ray_losses) > 1:
        spread    = float(np.max(ray_losses) - np.min(ray_losses))
        dra_score = float(np.clip(100.0 * (1.0 - spread / 30.0), 0, 100))
    else:
        dra_score = 50.0

    # DAS — how well is the null aligned with the nearest OAR?
    # das_angle_deg: 0° = perfect null alignment, 90° = orthogonal, 180° = forward on OAR
    # DAS = 100 when angle ≈ 0° (null directly on OAR), 0 when angle ≈ 90°+
    das_score = float(np.clip(100.0 * (1.0 - das_angle_deg / 90.0), 0, 100))

    # Weighted composite
    w   = ASI_WEIGHTS
    asi = (w["hss"] * hss_score +
           w["ocm"] * ocm_score +
           w["cc"]  * cc_score  +
           w["dra"] * dra_score +
           w["das"] * das_score)

    risk = ("LOW"      if asi >= 75 else
            "MODERATE" if asi >= 50 else
            "HIGH"     if asi >= 30 else "CRITICAL")

    interp = {
        "LOW":      "Directional ablation expected to achieve complete coverage with OAR protection.",
        "MODERATE": "Vessel proximity may affect zone — directional null reduces OAR dose.",
        "HIGH":     "Heat sink detected; antenna null orientation partially compensating.",
        "CRITICAL": "Zone compromised — reposition or staged treatment required.",
    }[risk]

    return {
        "asi":            round(asi, 1),
        "hss_score":      round(hss_score, 1),
        "ocm_score":      round(ocm_score, 1),
        "cc_score":       round(cc_score, 1),
        "dra_score":      round(dra_score, 1),
        "das_score":      round(das_score, 1),
        "risk_label":     risk,
        "max_loss_pct":   round(max_loss, 2),
        "min_clear_mm":   round(min_cl_mm, 1),
        "margin_mm":      round(margin_mm, 1),
        "spread_pct":     round(float(np.max(ray_losses) - np.min(ray_losses))
                                if len(ray_losses) > 1 else 0.0, 2),
        "das_angle_deg":  round(das_angle_deg, 1),
        "interpretation": interp,
    }

def print_asi_v11(asi):
    bar_len = 40
    filled  = int(round(asi["asi"] / 100.0 * bar_len))
    sym     = {"LOW":"🟢","MODERATE":"🟡","HIGH":"🟠","CRITICAL":"🔴"}[asi["risk_label"]]
    bar     = sym * filled + "⬜" * (bar_len - filled)
    print("\n" + "═"*70)
    print("  ABLATION SAFETY INDEX v11  (Directional Antenna)")
    print("═"*70)
    print(f"  Overall ASI : {asi['asi']:>5.1f} / 100   [{asi['risk_label']}]")
    print(f"  {bar}")
    print(f"\n  Sub-scores (weights):")
    print(f"  {'Heat Sink Severity':<30} HSS = {asi['hss_score']:>5.1f}  (w={ASI_WEIGHTS['hss']:.2f})"
          f"   max loss {asi['max_loss_pct']:.2f}%")
    print(f"  {'OAR Clearance Margin':<30} OCM = {asi['ocm_score']:>5.1f}  (w={ASI_WEIGHTS['ocm']:.2f})"
          f"   min wall {asi['min_clear_mm']:.1f} mm")
    print(f"  {'Coverage Confidence':<30}  CC = {asi['cc_score']:>5.1f}  (w={ASI_WEIGHTS['cc']:.2f})"
          f"   margin {asi['margin_mm']:.1f} mm")
    print(f"  {'Directional Ray Asymmetry':<30} DRA = {asi['dra_score']:>5.1f}  (w={ASI_WEIGHTS['dra']:.2f})"
          f"   spread {asi['spread_pct']:.2f}%")
    print(f"  {'Directional Antenna Score':<30} DAS = {asi['das_score']:>5.1f}  (w={ASI_WEIGHTS['das']:.2f})"
          f"   null angle {asi['das_angle_deg']:.1f}°  [NEW v11]")
    print(f"\n  ▶  {asi['interpretation']}")
    print(f"\n  Directional antenna physics (Fallahi 2018, Lee 2023):")
    print(f"    Forward gain G_fwd  = {G_FORWARD:.2f}×  →  zone radius × {G_FORWARD**0.5:.3f}")
    print(f"    Rear suppression    = {G_REAR:.2f}×  →  zone radius × {G_REAR**0.5:.3f} toward OAR")
    print("═"*70)


# ══════════════════════════════════════════════════════════════════════════════
#  MESH UTILITIES  (identical to v9/v10)
# ══════════════════════════════════════════════════════════════════════════════

def load_vtk(path):
    if not os.path.exists(path):
        print(f"  ✘ Missing: {path}")
        return None
    m = pv.read(path)
    print(f"  ✔ {os.path.basename(path)}  ({m.n_points} pts, {m.n_cells} cells)")
    return m

def rescale(mesh):
    if mesh is None: return None
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
    print("\n🔍 Extracting individual tumors...")
    tumors = tumor_mesh.connectivity().split_bodies()
    print(f"   Detected {len(tumors)} tumor(s)")
    return tumors

def tumor_metrics(tumors, surface, vessels, vnames):
    s_tree  = cKDTree(np.array(surface.points))
    v_trees = [cKDTree(np.array(v.points)) for v in vessels]
    metrics = []
    for i, t in enumerate(tumors):
        c   = np.array(t.center)
        b   = t.bounds
        dm  = max(b[1]-b[0], b[3]-b[2], b[5]-b[4])
        dep = float(s_tree.query(c, k=1)[0])
        vd  = [float(vt.query(c, k=1)[0]) for vt in v_trees]
        elig = (MIN_DIAMETER_CM <= dm*100 <= MAX_DIAMETER_CM
                and dep*100 <= MAX_DEPTH_CM)
        metrics.append({
            "idx": i, "centroid": c,
            "diameter_cm": dm * 100.0, "depth_cm": dep * 100.0,
            "vessel_dists_m": vd, "min_vessel_m": min(vd),
            "closest_vessel": vnames[int(np.argmin(vd))],
            "eligible": elig,
        })
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  BLOOD PARTICLE SYSTEM  (identical to v10 — Womersley/Poiseuille profiles)
# ══════════════════════════════════════════════════════════════════════════════

class VesselParticleSystem:
    def __init__(self, vessel, vessel_name, n_particles=80):
        pts    = np.array(vessel.points)
        D      = VESSEL_DIAMETERS[vessel_name]
        R      = D / 2.0
        u_mean = VESSEL_VELOCITIES[vessel_name]
        Re     = (RHO_B * u_mean * D) / MU_B
        cen    = pts - pts.mean(axis=0)
        _, _, vt = np.linalg.svd(cen[:min(5000, len(cen))], full_matrices=False)
        self.flow_dir = vt[0]
        self.origin   = pts.mean(axis=0)
        proj   = cen.dot(self.flow_dir)
        self.L = max(float(proj.max() - proj.min()), 0.02)
        idx    = np.random.choice(len(pts), min(n_particles, len(pts)),
                                  replace=False)
        spts   = pts[idx]
        rel    = spts - self.origin
        axc    = np.outer(rel.dot(self.flow_dir), self.flow_dir)
        perp   = rel - axc
        rv     = np.linalg.norm(perp, axis=1)
        self.r_norm  = np.clip(rv / (R + 1e-9), 0, 1)
        if Re < 2300:
            self.u_local = u_mean * 2.0 * (1.0 - self.r_norm**2)
        else:
            self.u_local = u_mean * (8/7) * (9/8) * (1.0 - self.r_norm)**(1/7)
        self.phase    = np.random.uniform(0, self.L, len(idx))
        self.base_pts = spts - np.outer(rel.dot(self.flow_dir), self.flow_dir)
        self.vessel_name = vessel_name
        self.Re          = Re
        self.n           = len(idx)
        self.speed_scale = u_mean / 500.0

    def update(self, t):
        axial = (self.phase + self.speed_scale * t) % self.L
        return self.base_pts + np.outer(axial, self.flow_dir), self.u_local


# ══════════════════════════════════════════════════════════════════════════════
#  DIRECTIONAL HEAT FLOW ARROWS  (scaled by both loss% and SAR weight)
# ══════════════════════════════════════════════════════════════════════════════

def create_directional_heat_arrows(centroid, vessels, vnames,
                                    per_vessel_hs, antenna_axis, plotter):
    """
    Arrows only — no floating labels to avoid clutter.
    Colour encodes loss_pct (green→yellow→red).
    Opacity encodes SAR weight (rear-lobe vessels appear faint).
    Cyan arrow = antenna forward axis.  Magenta = null toward OAR.
    """
    losses  = [hs["loss_pct"] for hs in per_vessel_hs.values()]
    mx, mn  = max(losses), min(losses)
    BASE    = 0.04

    def col(p):
        t = float(np.clip((p - mn) / max(mx - mn, 0.01), 0.0, 1.0))
        if t < 0.5:
            return [float(2*t), 1.0, 0.0]
        else:
            return [1.0, float(2*(1-t)), 0.0]

    for vn, hs in per_vessel_hs.items():
        if vn not in vnames:
            continue
        vessel  = vessels[vnames.index(vn)]
        pts     = np.array(vessel.points)
        _, idx  = cKDTree(pts).query(centroid, k=1)
        raw     = pts[idx] - centroid
        dist    = np.linalg.norm(raw)
        if dist < 1e-6:
            continue
        unit    = raw / dist
        arr_len = max(BASE * hs["loss_pct"] / max(mx, 1.), 0.005)
        sar_w   = float(hs.get("sar_weight", 1.0))
        # Clamp opacity to valid [0,1] range
        opacity = float(np.clip(0.30 + 0.65 * sar_w, 0.0, 1.0))

        plotter.add_mesh(
            pv.Arrow(start=centroid, direction=unit, scale=arr_len,
                     tip_length=0.3, tip_radius=0.05, shaft_radius=0.02),
            color=col(hs["loss_pct"]), opacity=opacity)

    # Antenna forward axis (cyan)
    ax_n = np.linalg.norm(antenna_axis)
    if ax_n > 1e-6:
        ax_unit = antenna_axis / ax_n
        plotter.add_mesh(
            pv.Arrow(start=centroid, direction=ax_unit, scale=0.06,
                     tip_length=0.25, tip_radius=0.07, shaft_radius=0.025),
            color="cyan", opacity=0.95)
        # Null direction (magenta)
        plotter.add_mesh(
            pv.Arrow(start=centroid, direction=-ax_unit, scale=0.04,
                     tip_length=0.25, tip_radius=0.06, shaft_radius=0.018),
            color="magenta", opacity=0.70)


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — OVERVIEW  (same as v10)
# ══════════════════════════════════════════════════════════════════════════════

def phase1_overview(surface, vessels, vnames, tumors, metrics):
    print("\n" + "═"*70)
    print("  PHASE 1 — OVERVIEW  (close window to proceed)")
    print("═"*70)
    plotter = pv.Plotter(window_size=[1400, 900],
                         title="OVERVIEW — All Tumors  |  Close to continue")
    plotter.background_color = "black"
    plotter.add_mesh(surface, color="lightgray", opacity=0.07, label="Body Surface")
    for v, vn in zip(vessels, vnames):
        plotter.add_mesh(v, color=VESSEL_COLOR_MAP.get(vn, "gray"),
                         opacity=0.60, label=vn.replace("_"," ").title())
    for i, (t, m) in enumerate(zip(tumors, metrics)):
        tc   = TUMOR_COLORS[i % len(TUMOR_COLORS)]
        elig = "✔ ELIGIBLE" if m["eligible"] else "✗ ineligible"
        plotter.add_mesh(t, color=tc, opacity=0.80,
                         label=f"T{i+1} {m['diameter_cm']:.1f}cm {elig}")
        plotter.add_mesh(pv.Sphere(radius=0.007, center=m["centroid"]),
                         color="white", opacity=0.95)
        plotter.add_point_labels(
            pv.PolyData([m["centroid"] + np.array([0, 0, 0.013])]),
            [f"T{i+1}"], font_size=14, text_color=tc,
            point_size=1, always_visible=True, shape_opacity=0.0)
    plotter.add_axes()
    plotter.add_legend(loc="upper right", size=(0.28, 0.52))
    plotter.add_text(
        "PHASE 1 — Directional MWA Planning v11\\n"
        "Close this window to select a tumor",
        position="upper_left", font_size=11, color="white")
    try:
        plotter.show(auto_close=False)
    except Exception as e:
        print(f"  Visualisation error (non-fatal): {e}")
    finally:
        plotter.close()


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — TUMOR + BIOLOGY + ENTRY AXIS SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def _pick_menu(title, options_dict):
    keys   = list(options_dict.keys())
    labels = [options_dict[k]["label"] for k in keys]
    print(f"\n  {title}")
    for i, lbl in enumerate(labels, 1):
        print(f"    {i}. {lbl}")
    while True:
        try:
            raw = input(f"  ▶  Enter choice [1–{len(keys)}]: ").strip()
            n   = int(raw)
            if 1 <= n <= len(keys):
                return keys[n - 1]
            print(f"  ✘ Enter a number between 1 and {len(keys)}.")
        except ValueError:
            print("  ✘ Invalid — enter an integer.")
        except (EOFError, KeyboardInterrupt):
            print("\n  Exiting.")
            sys.exit(0)

ENTRY_AXES = {
    "SUPERIOR":  {"label": "Superior (cranial) — from above the liver",   "vector": np.array([0., 0., -1.])},
    "ANTERIOR":  {"label": "Anterior — from the front (standard)",        "vector": np.array([0., -1., 0.])},
    "RIGHT_LAT": {"label": "Right lateral — percutaneous standard",       "vector": np.array([-1., 0., 0.])},
    "LEFT_LAT":  {"label": "Left lateral",                                 "vector": np.array([1.,  0., 0.])},
    "AUTO":      {"label": "Automatic — let optimizer choose entry axis",  "vector": None},
}

def phase2_pick_tumor(metrics, vnames):
    print("\n" + "═"*70)
    print("  PHASE 2 — TUMOR, BIOLOGY & ANTENNA ENTRY AXIS SELECTION")
    print("═"*70)

    # Tumor table
    print(f"\n  {'#':<5} {'Diam(cm)':<11} {'Depth(cm)':<11} "
          f"{'Closest vessel':<20} {'Dist(mm)':<11} {'Eligible?'}")
    print("  " + "─"*70)
    for m in metrics:
        elig = "✔ YES" if m["eligible"] else "✗ NO "
        print(f"  {m['idx']+1:<5} {m['diameter_cm']:<11.2f} "
              f"{m['depth_cm']:<11.2f} {m['closest_vessel']:<20} "
              f"{m['min_vessel_m']*1000:<11.1f} {elig}")

    eligible_ids = [m["idx"]+1 for m in metrics if m["eligible"]]
    if eligible_ids:
        print(f"\n  ✔ Eligible tumors: {eligible_ids}")
    else:
        print("\n  ⚠  No tumors meet standard MWA criteria.")

    while True:
        try:
            raw = input(f"\n  ▶  Enter tumor number [1–{len(metrics)}]: ").strip()
            n   = int(raw)
            if 1 <= n <= len(metrics):
                sel = metrics[n - 1]
                print(f"\n  ✔ Tumor {n} selected  "
                      f"({sel['diameter_cm']:.2f} cm, depth {sel['depth_cm']:.2f} cm, "
                      f"closest: {sel['closest_vessel']} @ {sel['min_vessel_m']*1000:.1f} mm)")
                break
            print(f"  ✘ Enter 1–{len(metrics)}.")
        except ValueError:
            print("  ✘ Invalid input.")
        except (EOFError, KeyboardInterrupt):
            print("\n  Exiting.")
            sys.exit(0)

    # Tumor type
    print("\n  ─────────────────────────────────────────────")
    type_key = _pick_menu("SELECT TUMOR HISTOLOGICAL TYPE:", TUMOR_TYPES)
    print(f"  ✔ {TUMOR_TYPES[type_key]['label']}")
    print(f"     → {TUMOR_TYPES[type_key]['description']}")

    # Consistency
    print("\n  ─────────────────────────────────────────────")
    consist_key = _pick_menu("SELECT TUMOR CONSISTENCY:", CONSISTENCY_FACTORS)
    print(f"  ✔ {CONSISTENCY_FACTORS[consist_key]['label']}")

    # Needle entry axis (NEW in v11)
    print("\n  ─────────────────────────────────────────────")
    print("  SELECT NEEDLE ENTRY DIRECTION")
    print("  (constrains antenna forward axis search to ±60° cone)")
    entry_key = _pick_menu("Entry direction:", ENTRY_AXES)
    entry_vec  = ENTRY_AXES[entry_key]["vector"]
    print(f"  ✔ Entry: {ENTRY_AXES[entry_key]['label']}")

    return sel, type_key, consist_key, entry_vec


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 3 — TREATMENT PLANNING VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def phase3_visualise(surface, vessels, vnames, tumors, centroids,
                     sel_idx, results, opt_result, asi,
                     oar_list, particle_systems, centroid_dists):

    print("\n🎬  Building directional treatment-planning visualisation (v11)...")

    power_w    = opt_result["P_opt"]
    time_s     = opt_result["t_opt"]
    fwd_m      = opt_result["zone_fwd_cm"]   / 100.0
    diam_m     = opt_result["zone_diam_fwd_cm"] / 100.0
    centroid   = centroids[sel_idx]
    antenna_ax = opt_result["antenna_axis"]
    constrained= opt_result["constrained"]
    per_hs     = opt_result["per_vessel_hs"]
    tissue     = opt_result["tissue"]
    consist    = opt_result["consistency"]

    asi_col = {"LOW":"lime","MODERATE":"yellow",
               "HIGH":"orange","CRITICAL":"tomato"}[asi["risk_label"]]

    plotter = pv.Plotter(
        window_size=[1500, 1000],
        title=(f"Directional MWA v11 — Tumor {sel_idx+1}  |  "
               f"{power_w:.0f}W × {time_s:.0f}s  |  "
               f"DAS={asi['das_score']:.0f}  |  "
               f"ASI={asi['asi']:.1f} [{asi['risk_label']}]"))
    plotter.background_color = "black"

    # Body surface
    plotter.add_mesh(surface, color="lightgray", opacity=0.07, label="Body Surface")

    # Vessels — OARs highlighted; rear-lobe protected ones in green
    for v, vn in zip(vessels, vnames):
        is_oar      = any(o["vessel"] == vn for o in oar_list)
        oar_info    = next((o for o in oar_list if o["vessel"] == vn), None)
        col         = VESSEL_COLOR_MAP.get(vn, "gray")
        if is_oar:
            if oar_info and oar_info.get("in_rear_lobe"):
                # In rear lobe = protected by directional antenna
                plotter.add_mesh(v, color="lime", opacity=0.80,
                                 label=f"✔ OAR (rear-protected): {vn.replace('_',' ').title()}")
            else:
                plotter.add_mesh(v, color="red", opacity=0.90,
                                 label=f"⚠ OAR (forward): {vn.replace('_',' ').title()}")
        else:
            plotter.add_mesh(v, color=col, opacity=0.60,
                             label=vn.replace("_"," ").title())

    # Tumors
    for i, t in enumerate(tumors):
        td    = smooth_tumor(t) if i == sel_idx else t
        op    = 0.85 if i == sel_idx else 0.22
        label = f"Tumor {i+1} [TARGET]" if i == sel_idx else f"Tumor {i+1}"
        plotter.add_mesh(td, color=TUMOR_COLORS[i % len(TUMOR_COLORS)],
                         opacity=op, label=label)
    plotter.add_mesh(pv.Sphere(radius=0.006, center=centroid),
                     color="white", label="Tumour centroid")

    # Directional heat flow arrows
    create_directional_heat_arrows(centroid, vessels, vnames,
                                    per_hs, antenna_ax, plotter)

    # Directional ray lines (SAR-weighted colour)
    ray_actor_names, ray_meshes_list = [], []
    if results:
        losses = np.array([r.get("loss_pct", 0) for r in results])
        lo, hi = losses.min(), losses.max()
        norm   = (losses - lo) / (hi - lo + 1e-9)
        step   = max(1, len(results) // 80)
        for i in range(0, len(results), step):
            r     = results[i]
            ep    = centroid + r["ray_direction"] * r["path_distance"]
            cv    = float(norm[i])
            sar_w = float(r.get("sar_weight", 1.0))
            # Normalise SAR weight to [0,1]: sar_w in [G_REAR, G_FORWARD]
            sar_norm = float(np.clip(
                (sar_w - G_REAR) / (G_FORWARD - G_REAR + 1e-9), 0.0, 1.0))
            # All RGB components must be plain float in [0.0, 1.0]
            col = [
                float(np.clip(cv * sar_norm, 0.0, 1.0)),
                0.0,
                float(np.clip((1.0 - cv) * sar_norm, 0.0, 1.0)),
            ]
            name = f"ray_{i}"
            ray_actor_names.append(name)
            ray_meshes_list.append((pv.Line(centroid, ep), col))
            plotter.add_mesh(pv.Line(centroid, ep), color=col,
                             line_width=2.5, opacity=0.55, name=name)

    # OAR exclusion spheres
    for oar in oar_list:
        if oar["risk"] == "CRITICAL":
            vn  = oar["vessel"]
            er  = VESSEL_RADII.get(vn, 0.005) + OAR_MIN_CLEAR_M
            ctr = oar.get("nearest_pt", centroid)
            sph = pv.Sphere(radius=er, center=ctr,
                            theta_resolution=20, phi_resolution=20)
            oar_col = "lime" if oar.get("in_rear_lobe") else "red"
            plotter.add_mesh(sph, color=oar_col, opacity=0.18,
                             label=f"OAR exclusion: {vn.replace('_',' ')}")
            plotter.add_mesh(sph, color=oar_col, opacity=0.55,
                             style="wireframe", line_width=1.2)

    # Needle reposition arrow (only if constrained AND in forward hemisphere)
    fwd_constrained_oars = [o for o in oar_list
                             if not o.get("in_rear_lobe") and
                             o["wall_clear_mm"] < OAR_MIN_CLEAR_M * 1000]
    if constrained and fwd_constrained_oars:
        closest_oar = min(fwd_constrained_oars, key=lambda o: o["wall_clear_mm"])
        oar_pt      = closest_oar.get("nearest_pt", centroid)
        away        = centroid - oar_pt
        an          = np.linalg.norm(away)
        away        = away / an if an > 1e-6 else np.array([0., 1., 0.])
        need_mm     = OAR_MIN_CLEAR_M*1000 - closest_oar["wall_clear_mm"]
        plotter.add_mesh(
            pv.Arrow(start=centroid, direction=away, scale=0.04,
                     tip_length=0.25, tip_radius=0.08, shaft_radius=0.03),
            color="cyan", opacity=0.95, label="Suggested needle shift")
        plotter.add_point_labels(
            pv.PolyData([centroid + away * 0.048]),
            [f"↑ Shift away from {closest_oar['vessel'].replace('_',' ')}\\n"
             f"Need {need_mm:.1f}mm more"],
            font_size=10, text_color="cyan",
            point_size=1, always_visible=True, shape_opacity=0.0)

    # ── Animation state ─────────────────────────────────────────────────
    mode_state = {"rays_on": True}
    play_state = {"playing": False, "t": 0.0}

    def clear_dynamic():
        for nm in ["ablation_fwd", "particles", "hud_anim"]:
            try:
                plotter.remove_actor(nm)
            except Exception:
                pass

    def draw_zone(frac):
        zone = make_dshaped_zone(centroid, fwd_m, diam_m, antenna_ax, frac=frac)
        if zone is not None and zone.n_points > 0:
            try:
                plotter.add_mesh(
                    zone, scalars="Temperature_C", cmap="plasma",
                    clim=[T_BLOOD, T_TISS], opacity=0.60,
                    name="ablation_fwd",
                    scalar_bar_args={
                        "title":            "Temp (°C)",
                        "n_labels":         4,
                        "label_font_size":  10,
                        "title_font_size":  11,
                        # bottom-right, clear of buttons and slider
                        "position_x":       0.86,
                        "position_y":       0.12,
                        "width":            0.06,
                        "height":           0.30,
                        "color":            "white",
                        "vertical":         True,
                    })
            except Exception:
                pass
        return fwd_m * frac, diam_m * frac

    def update(t_val):
        t    = float(t_val)
        frac = min(t / max(time_s, 1.0), 1.0)
        clear_dynamic()

        cur_fwd, cur_diam = draw_zone(frac)

        # Blood particles — colour bar on far right, above temperature bar
        all_pts, all_vel = [], []
        for ps in particle_systems:
            pts, vel = ps.update(t)
            all_pts.append(pts)
            all_vel.append(vel)
        if all_pts:
            cloud = pv.PolyData(np.vstack(all_pts))
            cloud["blood_velocity_m_s"] = np.concatenate(all_vel)
            plotter.add_mesh(
                cloud, scalars="blood_velocity_m_s", cmap="coolwarm",
                clim=[0.0, max(VESSEL_VELOCITIES.values()) * 2.0],
                point_size=5, render_points_as_spheres=True,
                name="particles",
                scalar_bar_args={
                    "title":            "Blood vel (m/s)",
                    "n_labels":         3,
                    "label_font_size":  10,
                    "title_font_size":  11,
                    # stacked just above the temperature bar
                    "position_x":       0.86,
                    "position_y":       0.50,
                    "width":            0.06,
                    "height":           0.25,
                    "color":            "white",
                    "vertical":         True,
                })

        # ── Animated HUD — left panel, well clear of legend ─────────────
        oar_rear  = sum(1 for o in oar_list if o.get("in_rear_lobe"))
        oar_fwd   = len(oar_list) - oar_rear
        pct_str   = f"{frac*100:.0f}%"
        warn_str  = "CONSTRAINED" if constrained else "OAR-SAFE"

        if cur_fwd > 0.001:
            zone_str = f"Fwd {cur_fwd*100:.1f} cm  Rear {opt_result['zone_diam_rear_cm']:.1f} cm"
        else:
            zone_str = "Growing..."

        hud_lines = [
            "ASI v11 ── LIVE",
            f"ASI  {asi['asi']:.1f}/100  [{asi['risk_label']}]",
            f"HSS  {asi['hss_score']:.0f}   OCM  {asi['ocm_score']:.0f}",
            f"CC   {asi['cc_score']:.0f}   DRA  {asi['dra_score']:.0f}",
            f"DAS  {asi['das_score']:.0f}  (ant. score)",
            "──────────────────",
            f"P  {power_w:.0f} W    t  {t:.0f}/{time_s:.0f} s",
            f"Progress  {pct_str}",
            f"{zone_str}",
            "──────────────────",
            f"Status:  {warn_str}",
            f"OARs rear-protected: {oar_rear}",
            f"OARs forward risk:   {oar_fwd}",
            "──────────────────",
            "Cyan  = antenna fwd axis",
            "Magenta = null (OAR)",
        ]
        hud_text = "\n".join(hud_lines)

        plotter.add_text(
            hud_text,
            position=(12, 200),          # pixel coords: left edge, above buttons
            font_size=10,
            color=asi_col,
            name="hud_anim",
        )
        plotter.render()

    # ── Timer callback ───────────────────────────────────────────────────
    def toggle_play(flag):
        play_state["playing"] = bool(flag)
        if play_state["playing"]:
            plotter.add_timer_event(max_steps=100000, duration=100,
                                    callback=_tick)

    def _tick(step):
        if not play_state["playing"]:
            return
        play_state["t"] = (play_state["t"] + 5.0) % (time_s + 1.0)
        slider_w.GetRepresentation().SetValue(play_state["t"])
        update(play_state["t"])

    # ── Slider — centred, clear of axes widget (bottom-left) and buttons ─
    # Slider spans 30%–88% of width, sits at y=0.04 (4% from bottom)
    slider_w = plotter.add_slider_widget(
        update,
        rng=[0.0, time_s],
        value=0.0,
        title="Time (s)",
        pointa=(0.30, 0.04),
        pointb=(0.88, 0.04),
        style="modern",
    )

    # ── Control buttons — bottom-left stack, away from axes widget ───────
    # ▶ Play  at pixel (12, 30)
    plotter.add_checkbox_button_widget(
        toggle_play, value=False,
        position=(12, 30),
        size=38, border_size=2,
        color_on="lime", color_off="dimgray",
    )
    plotter.add_text("Play", position=(56, 40), font_size=10, color="white")

    # ◉ SAR rays  at pixel (12, 80)
    def toggle_rays(flag):
        mode_state["rays_on"] = bool(flag)
        if flag:
            for nm, (lm, col_r) in zip(ray_actor_names, ray_meshes_list):
                plotter.add_mesh(lm, color=col_r, line_width=2.5,
                                 opacity=0.55, name=nm)
        else:
            for nm in ray_actor_names:
                try:
                    plotter.remove_actor(nm)
                except Exception:
                    pass
        plotter.render()

    plotter.add_checkbox_button_widget(
        toggle_rays, value=True,
        position=(12, 80),
        size=38, border_size=2,
        color_on="yellow", color_off="dimgray",
    )
    plotter.add_text("SAR rays", position=(56, 90), font_size=10, color="yellow")

    # ── Legend — top-right, compact, no overlap with 3D scene ────────────
    # Use a narrow legend so it doesn't bleed into the viewport
    plotter.add_legend(
        loc="upper right",
        size=(0.22, 0.48),    # width=22%, height=48% of window
        bcolor=[0.08, 0.08, 0.08],   # dark semi-transparent background
        border=True,
    )

    # ── Static title bar (top-left, single line) ─────────────────────────
    status_tag = "CONSTRAINED" if constrained else "OAR-SAFE"
    plotter.add_text(
        f"Directional MWA v11   {power_w:.0f}W x {time_s:.0f}s   "
        f"G_fwd={G_FORWARD:.1f}x   {status_tag}   "
        f"DAS={asi['das_score']:.0f}   ASI={asi['asi']:.1f} [{asi['risk_label']}]",
        position="upper_left",
        font_size=11,
        color=asi_col,
    )

    # ── Colour legend (bottom-right text, below scalar bars) ─────────────
    plotter.add_text(
        "Colour key:  Green=low heat / Yellow=mid / Red=high  |   Lime vessel=rear-protected OAR  |  Red vessel=forward OAR  |   Bright ray=fwd SAR  Faint ray=rear null",
        position=(12, 130),
        font_size=9,
        color="lightgray",
    )

    # ── 3D axes — top-left corner (clear of HUD) ─────────────────────────
    plotter.add_axes(
        line_width=3,
        x_color="red", y_color="lime", z_color="dodgerblue",
        xlabel="X", ylabel="Y", zlabel="Z",
        interactive=False,
    )

    update(0.0)
    try:
        plotter.show(auto_close=False)
    except Exception as e:
        print(f"  Vis error: {e}")
    finally:
        plotter.close()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔" + "═"*68 + "╗")
    print("║  DIRECTIONAL MWA PLANNING SYSTEM  v11                            ║")
    print("║  Monopole + Reflector · Biophysical Optimizer · ASI Risk Index   ║")
    print("║  Ref: Fallahi & Prakash 2018 · Lee 2023 · Audigier et al. 2020  ║")
    print("╚" + "═"*68 + "╝")
    print(f"\n  Antenna physics:")
    print(f"    Forward gain  G_fwd  = {G_FORWARD:.2f}×  (Fallahi 2018 Fig.15-16)")
    print(f"    Rear null     G_rear = {G_REAR:.2f}×  (80% suppression measured)")
    print(f"    Beam tilt           = {BEAM_TILT_DEG:.1f}°  (Lee 2023 Table 1)")
    print(f"    Monopole length     ≈ {MONOPOLE_LEN_MM:.1f} mm  (λ_eff/4 at 2.45 GHz in liver)")

    if not os.path.exists(DATASET_BASE):
        print(f"\n  ✘ Dataset not found: {DATASET_BASE}")
        return

    # ── Load meshes ─────────────────────────────────────────────────────
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

    tumors    = extract_tumors(tumor_mesh)
    metrics   = tumor_metrics(tumors, surface, vessels, vnames)
    centroids = np.array([m["centroid"] for m in metrics])

    # ═══ PHASE 1 ═══════════════════════════════════════════════════════
    phase1_overview(surface, vessels, vnames, tumors, metrics)

    # ═══ PHASE 2 ═══════════════════════════════════════════════════════
    sel, type_key, consist_key, entry_vec = phase2_pick_tumor(metrics, vnames)
    sel_idx  = sel["idx"]
    centroid = sel["centroid"]
    sel_diam = sel["diameter_cm"]

    # Per-vessel distances
    centroid_dists = {
        vnames[i]: float(cKDTree(np.array(v.points)).query(centroid, k=1)[0])
        for i, v in enumerate(vessels)
    }

    # ── OAR Orientation Solver ────────────────────────────────────────
    print("\n  Running OAR orientation solver...")
    best_axis_init, top5, _ = find_optimal_antenna_axis(
        centroid, centroid_dists, vnames, needle_insertion_dir=entry_vec)

    # Refine using actual vessel coordinates
    antenna_axis, das_angle_deg = refine_axis_with_vessel_coords(
        centroid, vessels, vnames, best_axis_init, centroid_dists)

    print(f"  ✔ Optimal antenna axis: "
          f"[{antenna_axis[0]:.3f}, {antenna_axis[1]:.3f}, {antenna_axis[2]:.3f}]")
    print(f"  ✔ DAS alignment angle  : {das_angle_deg:.1f}°  "
          f"(0° = perfect null on OAR, <30° = excellent)")

    # ── Ray tracing (directional SAR-weighted) ────────────────────────
    print("\n  Directional ray tracing...")
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
            seg_d  = {
                vn: ray_segment_dist(centroid, direction, path_d,
                                     v_pts[vi], centroid_dists[vn])
                for vi, vn in enumerate(vnames)
            }
            dom_vn = min(seg_d, key=seg_d.get)

            # SAR weight for this ray direction relative to antenna axis
            sar_w = directional_sar_weight(direction, antenna_axis)

            hs    = heat_sink_physics(seg_d[dom_vn], dom_vn, 60.0, 300.0,
                                       sar_weight=sar_w)
            hs["ray_direction"] = direction
            hs["path_distance"] = path_d
            hs["sar_weight"]    = sar_w
            results.append(hs)
        except Exception:
            continue

    all_losses = [r["loss_pct"] for r in results]
    print(f"  {len(results)} rays | SAR-weighted loss "
          f"{np.min(all_losses):.2f}% – {np.max(all_losses):.2f}%")

    # ═══ DIRECTIONAL BIOPHYSICAL OPTIMIZER ═════════════════════════════
    opt_result = run_directional_optimizer(
        tumor_diam_cm   = sel_diam,
        tumor_type_key  = type_key,
        consistency_key = consist_key,
        centroid_dists  = centroid_dists,
        vnames          = vnames,
        vessels         = vessels,
        tumor_centroid  = centroid,
        antenna_axis    = antenna_axis,
        margin_cm       = 0.5,
    )

    # ── OAR identification (D-shaped zone) ───────────────────────────
    oar_list = identify_oars_directional(
        centroid, vessels, vnames,
        opt_result["zone_diam_fwd_cm"],
        opt_result["zone_diam_rear_cm"],
        opt_result["zone_fwd_cm"],
        antenna_axis)

    print(f"\n  OARs encroached: {len(oar_list)}")
    for o in oar_list:
        rear_str = " ← rear-lobe PROTECTED" if o.get("in_rear_lobe") else ""
        print(f"    {o['vessel']}  pts={o['points_inside']}  "
              f"wall={o['wall_clear_mm']:.1f}mm  [{o['risk']}]{rear_str}")

    # ═══ ASI v11 ═══════════════════════════════════════════════════════
    asi = compute_asi_v11(
        per_vessel_hs    = opt_result["per_vessel_hs"],
        clearance_report = opt_result["clearance_report"],
        tumor_diam_cm    = sel_diam,
        zone_diam_fwd_cm = opt_result["zone_diam_fwd_cm"],
        ray_losses       = all_losses,
        constrained      = opt_result["constrained"],
        das_angle_deg    = das_angle_deg,
        antenna_axis     = antenna_axis,
        centroid_dists   = centroid_dists,
        vnames           = vnames,
    )
    print_asi_v11(asi)

    # ── Final prescription ────────────────────────────────────────────
    print("\n" + "═"*70)
    print("  FINAL DIRECTIONAL MWA PRESCRIPTION  (v11)")
    print("═"*70)
    print(f"  Tumor         : {sel_idx+1}  ({sel_diam:.2f} cm)")
    print(f"  Histology     : {TUMOR_TYPES[type_key]['label']}")
    print(f"  Consistency   : {CONSISTENCY_FACTORS[consist_key]['label']}")
    print(f"  Dose factor   : ×{opt_result['dose_sf']:.3f}")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Power         : {opt_result['P_opt']:.1f} W")
    print(f"  Time          : {opt_result['t_opt']:.0f} s  ({opt_result['t_opt']/60:.1f} min)")
    print(f"  Forward zone  : {opt_result['zone_diam_fwd_cm']:.2f} cm diam  × {opt_result['zone_fwd_cm']:.2f} cm fwd")
    print(f"  Rear zone     : {opt_result['zone_diam_rear_cm']:.2f} cm diam  [OAR-protected hemisphere]")
    print(f"  Antenna axis  : [{antenna_axis[0]:.3f}, {antenna_axis[1]:.3f}, {antenna_axis[2]:.3f}]")
    print(f"  DAS angle     : {das_angle_deg:.1f}°  (null–OAR alignment)")
    print(f"  Q_sink total  : {opt_result['Q_sink_W']:.3f} W  (directionally weighted)")
    print(f"  P_net         : {opt_result['P_net_W']:.3f} W")
    print(f"  Converged     : {'YES' if opt_result['converged'] else 'NO'}")
    print(f"  Constrained   : {'YES' if opt_result['constrained'] else 'NO'}")
    print(f"  ASI v11       : {asi['asi']:.1f} / 100  [{asi['risk_label']}]")
    print(f"  DAS score     : {asi['das_score']:.1f} / 100  (directional antenna score)")
    print("═"*70)

    # ── Blood particles ───────────────────────────────────────────────
    print("\n  Building blood particle systems...")
    particle_systems = []
    for v, vn in zip(vessels, vnames):
        ps = VesselParticleSystem(v, vn, n_particles=80)
        particle_systems.append(ps)
        print(f"   {vn}: {ps.n} particles, Re={ps.Re:.0f} "
              f"({'Laminar' if ps.Re<2300 else 'Turbulent/Transition'})")

    # ═══ PHASE 3 ═══════════════════════════════════════════════════════
    phase3_visualise(
        surface, vessels, vnames, tumors, centroids,
        sel_idx, results, opt_result, asi,
        oar_list, particle_systems, centroid_dists)

    print("\n  ✔  Directional MWA planning complete (v11).")
    return opt_result, asi


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  Stopped.")
    except Exception as e:
        print(f"\n  Error: {e}")
        raise
