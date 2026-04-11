import os
import math
import numpy as np
import open3d as o3d
import trimesh

from vtkmodules.vtkIOLegacy import vtkPolyDataReader
from vtkmodules.vtkCommonCore import vtkIdList
from vtkmodules.util import numpy_support
# ---------- helper to read VTK polygonal mesh into trimesh ----------
def read_vtk_to_trimesh(vtk_path):
    reader = vtkPolyDataReader()
    reader.SetFileName(vtk_path)
    reader.Update()
    poly = reader.GetOutput()
    vtk_points = numpy_support.vtk_to_numpy(poly.GetPoints().GetData())
    # collect triangles from polys connectivity
    cells = poly.GetPolys()
    id_list = vtk.vtkIdList()  # but we avoid directly using it — simpler: use connectivity array
    # Use numpy conversion for cells:
    arr = numpy_support.vtk_to_numpy(cells.GetData())
    # VTK polygons array format: [n0, id0,id1,..., n1, id0,id1,...]
    # We can parse into triangles assuming everything is triangles (n==3)
    faces = []
    i = 0
    while i < len(arr):
        n = int(arr[i])
        if n == 3:
            faces.append([int(arr[i+1]), int(arr[i+2]), int(arr[i+3])])
        else:
            # if polygon has >3 vertices, triangulate simply by fan triangulation
            verts = [int(x) for x in arr[i+1:i+1+n]]
            for k in range(1, n-1):
                faces.append([verts[0], verts[k], verts[k+1]])
        i += 1 + n
    faces = np.array(faces, dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=vtk_points, faces=faces, process=False)
    return mesh