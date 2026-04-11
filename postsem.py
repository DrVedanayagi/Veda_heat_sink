# -------------------------------------------------------
# 3D ABLATION PATH + HEAT SINK (Open3D + PyVista loader)
# -------------------------------------------------------

import os
import math
import numpy as np
import open3d as o3d
import trimesh
import pyvista as pv  # ⭐ use PyVista as robust VTK reader

# -------------------------------------------------------
# 1. PyVista VTK → TRIMESH (robust for your dataset)
# -------------------------------------------------------
def read_vtk_to_trimesh(path: str) -> trimesh.Trimesh:
    """Read a VTK surface mesh using PyVista, convert to trimesh."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    mesh = pv.read(path)

    if mesh.n_points == 0:
        raise ValueError(f"[ERROR] No points in: {path}")

    # PyVista face array format: [3, i0, i1, i2, 3, j0, j1, j2, ...]
    faces_raw = mesh.faces
    if faces_raw.size == 0:
        raise ValueError(f"[ERROR] No faces in: {path}")

    # In your dataset, all surfaces are triangles → faces_raw.size == 4 * n_faces
    if faces_raw.size % 4 != 0:
        raise ValueError(
            f"[ERROR] Faces array not multiple of 4 in: {path}. "
            f"Got length={faces_raw.size}. This suggests non-triangular cells."
        )

    faces = faces_raw.reshape(-1, 4)[:, 1:4]

    tm = trimesh.Trimesh(vertices=np.asarray(mesh.points),
                         faces=faces,
                         process=False)

    if tm.is_empty or tm.faces.shape[0] == 0:
        raise ValueError(f"[ERROR] Trimesh has no triangles for: {path}")

    return tm

# -------------------------------------------------------
# 2. TRIMESH → OPEN3D
# -------------------------------------------------------
def tm_to_o3d(tm: trimesh.Trimesh, color=(0.7, 0.7, 0.7)) -> o3d.geometry.TriangleMesh:
    """Convert trimesh.Trimesh to Open3D TriangleMesh."""
    if tm.faces.shape[0] == 0:
        raise ValueError("Trimesh has no faces; cannot convert to Open3D.")

    m = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(tm.vertices),
        triangles=o3d.utility.Vector3iVector(tm.faces)
    )
    m.compute_vertex_normals()
    m.paint_uniform_color(color)
    return m

# -------------------------------------------------------
# 3. CONFIG – your portalvenous dataset
# -------------------------------------------------------
BASE = r"C:\ved_project\3d simulation\portalvenous\portalvenous"

PATHS = {
    # lesion / tumor surface (you used *_out.vtk for surface lesion earlier)
    "tumor":   os.path.join(BASE, "908ac52300007_out.vtk"),

    # liver / body surface
    "surface": os.path.join(BASE, "908ac52300013.vtk"),

    # vessels
    "portal":  os.path.join(BASE, "908ac52300001.vtk"),
    "hepatic": os.path.join(BASE, "908ac52300002.vtk"),
    "aorta":   os.path.join(BASE, "908ac52300003.vtk"),
    "ivc":     os.path.join(BASE, "908ac52300004.vtk"),
    "ha":      os.path.join(BASE, "908ac52300005.vtk"),
}

# -------------------------------------------------------
# 4. BUILD RAYCASTING SCENE
# -------------------------------------------------------
def build_scene(mesh_list):
    """Build an Open3D RaycastingScene from a list of legacy TriangleMeshes."""
    scene = o3d.t.geometry.RaycastingScene()
    for m in mesh_list:
        if len(m.triangles) == 0:
            raise ValueError("One mesh has no triangles when building scene.")
        tmesh = o3d.t.geometry.TriangleMesh.from_legacy(m)
        scene.add_triangles(tmesh)
    return scene

# -------------------------------------------------------
# 5. RAY SAMPLING ON A SPHERE
# -------------------------------------------------------
def sphere_dirs(n_theta=40, n_phi=80):
    """Generate approximately uniform directions on a sphere."""
    dirs = []
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    for t in theta:
        for p in phi:
            dirs.append([
                math.sin(t) * math.cos(p),
                math.sin(t) * math.sin(p),
                math.cos(t),
            ])
    return np.array(dirs, dtype=np.float32)

# -------------------------------------------------------
# 6. SIMPLE HEAT-SINK ATTENUATION MODEL
# -------------------------------------------------------
def heat_sink(dist, k=4.0):
    """
    Simple exponential cooling model.
    Smaller vessel distance → stronger cooling (smaller factor).
    """
    return math.exp(-k / max(dist, 1e-3))

# -------------------------------------------------------
# 7. RAYCASTING + HEAT
# -------------------------------------------------------
def raycast_with_heat(origin, surface_scene, vessel_scene, directions):
    """
    For each direction:
      - Cast ray from origin.
      - Find surface intersection.
      - Check if any vessel is hit before surface.
      - Mark ray as 'safe' or 'blocked' and assign heat attenuation.
    """
    safe_rays = []
    blocked_rays = []
    heat_factors = []

    origin = origin.astype(np.float32)

    for d in directions:
        d = d / np.linalg.norm(d)

        ray = o3d.core.Tensor(
            [[origin[0], origin[1], origin[2], d[0], d[1], d[2]]],
            dtype=o3d.core.Dtype.Float32
        )

        # Surface hit
        surf_res = surface_scene.cast_rays(ray)
        t_surf = float(surf_res["t_hit"].numpy()[0])

        if not math.isfinite(t_surf) or t_surf <= 0:
            continue

        surf_pt = origin + d * t_surf

        # Vessel hit
        ves_res = vessel_scene.cast_rays(ray)
        t_v = float(ves_res["t_hit"].numpy()[0])

        if math.isfinite(t_v) and 0 < t_v < t_surf:
            # Blocked by vessel
            blocked_rays.append((origin, origin + d * t_v))
            heat_factors.append(heat_sink(t_v))
        else:
            # Safe ray up to surface
            safe_rays.append((origin, surf_pt))
            heat_factors.append(1.0)

    return safe_rays, blocked_rays, heat_factors

# -------------------------------------------------------
# 8. MAIN PIPELINE
# -------------------------------------------------------
def main():
    print("Loading meshes via PyVista...")

    tumor_tm   = read_vtk_to_trimesh(PATHS["tumor"])
    surface_tm = read_vtk_to_trimesh(PATHS["surface"])

    vessel_tms = [
        read_vtk_to_trimesh(PATHS["portal"]),
        read_vtk_to_trimesh(PATHS["hepatic"]),
        read_vtk_to_trimesh(PATHS["aorta"]),
        read_vtk_to_trimesh(PATHS["ivc"]),
        read_vtk_to_trimesh(PATHS["ha"]),
    ]

    print(f"Tumor vertices: {tumor_tm.vertices.shape[0]}, faces: {tumor_tm.faces.shape[0]}")
    print(f"Surface vertices: {surface_tm.vertices.shape[0]}, faces: {surface_tm.faces.shape[0]}")

    # Convert to Open3D
    tumor_o3   = tm_to_o3d(tumor_tm, (1, 0, 0))
    surface_o3 = tm_to_o3d(surface_tm, (0.8, 0.8, 0.8))

    vessel_o3 = [
        tm_to_o3d(vessel_tms[0], (0, 0, 1)),
        tm_to_o3d(vessel_tms[1], (1, 0, 1)),
        tm_to_o3d(vessel_tms[2], (1, 0, 0)),
        tm_to_o3d(vessel_tms[3], (1, 1, 0)),
        tm_to_o3d(vessel_tms[4], (1, 0.5, 0)),
    ]

    # Tumor centroid (consistent with mesh coordinates)
    centroid = tumor_tm.vertices.mean(axis=0)
    print("Tumor centroid:", centroid)

    # Build raycasting scenes
    surface_scene = build_scene([surface_o3])
    vessel_scene  = build_scene(vessel_o3)

    # Directions for ray casting
    dirs = sphere_dirs()

    # Compute safe / blocked rays with heat sink
    safe, blocked, heat = raycast_with_heat(centroid, surface_scene, vessel_scene, dirs)
    print(f"Safe rays: {len(safe)}")
    print(f"Blocked rays: {len(blocked)}")

    # ---------------------------------------------------
    # Visualization in Open3D
    # ---------------------------------------------------
    geoms = [surface_o3] + vessel_o3 + [tumor_o3]

    # Safe rays → green
    for s, e in safe[::20]:
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector([s, e]),
            lines=o3d.utility.Vector2iVector([[0, 1]])
        )
        ls.paint_uniform_color((0, 1, 0))
        geoms.append(ls)

    # Blocked rays → red
    for s, e in blocked[::20]:
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector([s, e]),
            lines=o3d.utility.Vector2iVector([[0, 1]])
        )
        ls.paint_uniform_color((1, 0, 0))
        geoms.append(ls)

    # Centroid marker
    cs = o3d.geometry.TriangleMesh.create_sphere(radius=3.0)
    cs.translate(centroid)
    cs.paint_uniform_color((1, 1, 0))
    geoms.append(cs)

    print("Launching Open3D viewer...")
    o3d.visualization.draw_geometries(geoms)

# -------------------------------------------------------
# 9. ENTRY POINT
# -------------------------------------------------------
if __name__ == "__main__":
    main()
