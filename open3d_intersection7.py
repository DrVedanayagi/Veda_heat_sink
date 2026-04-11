import sys
print(sys.executable)

import open3d as o3d
import trimesh
import numpy as np
from vtkmodules.vtkIOLegacy import vtkPolyDataReader
from vtkmodules.vtkCommonCore import vtkIdList
from vtkmodules.util import numpy_support
import os
import math
import volume
from misc import utils
def getDatasetpath():
    return 'C:/ved_project/Nunna Algo/Dataset'
def getMultilabelpath():
    return 'multilabel.mhd'
def getSphearPath():
    return 'C:/ved_project/Nunna Algo/Dataset'
def getSphearfilepath():
    return 'optimized_spheres.txt'    
def  getSkinfilepath():
    return '908ac523data00013_skin_out.vtk'

def getCellIds(polydata):
        cells = polydata.GetPolys()
        ids = []
        faceIds  = vtkIdList()       
        cells.InitTraversal()
        while cells.GetNextCell(faceIds):
            for i in range(0, faceIds.GetNumberOfIds()):
                pId = faceIds.GetId(i)
                ids.append(pId)
        ids = np.array(ids)
        return ids        

def getColor():
    return [50,50,250,100]

def getLeasionfilepath():
    return 'portalvenous\\908ac52300007_out.vtk'

def  getSkeltonfilepath():
    return '908ac523data00014_skelton_out.vtk'
def getHepaticveinpath():
    return 'portalvenous\\908ac52300002_out.vtk'
def getPortalvenousveinpath():
    return 'portalvenous\\908ac52300001_out.vtk'
def getAortapath():
    return 'portalvenous\\908ac52300003_out.vtk'
def getIvcpath():
    return 'portalvenous\\908ac52300004_out.vtk'
def getHApath():
    return 'portalvenous\\908ac52300005_out.vtk'
def getLeftKidneypath():
    return 'portalvenous\\908ac52300008_out.vtk'
def getRightKidneypath():
    return 'portalvenous\\908ac52300009_out.vtk'
def getHeartpath():
    return 'portalvenous\\908ac52300010_out.vtk'
def createMesh(vtk_path):
    reader = vtkPolyDataReader()    
    reader.SetFileName(vtk_path)
    reader.Update()
    polyData = reader.GetOutput()
    mesh={}
    mesh['POINTS'] = numpy_support.vtk_to_numpy(polyData.GetPoints().GetData()) 
    numpy_cells_leasion = getCellIds(polyData) 
    polygons = numpy_cells_leasion.reshape(-1,3)
    mesh['POLYGONS']= polygons # the unspecified value is inferred by numpy
    Mesh = trimesh.Trimesh(vertices= mesh['POINTS'], faces= mesh['POLYGONS'],use_embree=True)
    Mesh.visual.face_colors = getColor()
    #Mesh.unmerge_vertices()
    return Mesh
def createSphere(center,radius):
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=radius)
    sphere.apply_translation([center[0],center[1],center[2]])
    sphere.unmerge_vertices()
    return sphere
def ele_sqrt(x):
    return np.sqrt(x[0]**2+x[1]**2+x[2]**2)
#---------------
vtk_hvpath = os.path.join(getDatasetpath(),getHepaticveinpath())
geom_hv = createMesh(vtk_hvpath)


mesh_hv =  o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(geom_hv.vertices.copy()),
                                                triangles=o3d.utility.Vector3iVector(geom_hv.faces.copy()))
mesh_hv.compute_vertex_normals()
mesh_hv.paint_uniform_color([1., 0.043137, 0.83921568627450])

#-----------------
vtk_pvpath = os.path.join(getDatasetpath(),getPortalvenousveinpath())
geom_pv = createMesh(vtk_pvpath)
mesh_pv= o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(geom_pv.vertices.copy()),
                                                triangles=o3d.utility.Vector3iVector(geom_pv.faces.copy()))
mesh_pv.compute_vertex_normals()
mesh_pv.paint_uniform_color([0.5, .5, 1.])
#-------------------------
vtk_aorta = os.path.join(getDatasetpath(),getAortapath())
geom_aorta = createMesh(vtk_aorta)
mesh_aorta= o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(geom_aorta.vertices.copy()),
                                                triangles=o3d.utility.Vector3iVector(geom_aorta.faces.copy()))
mesh_aorta.compute_vertex_normals()
mesh_aorta.paint_uniform_color([0.69019607843137254901960784313725, 1., 0.22745098039215686274509803921569])

#-----------------
vtk_ivcpath = os.path.join(getDatasetpath(),getIvcpath())
geom_ivc = createMesh(vtk_ivcpath)
mesh_ivc= o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(geom_ivc.vertices.copy()),
                                                triangles=o3d.utility.Vector3iVector(geom_ivc.faces.copy()))
mesh_ivc.compute_vertex_normals()
mesh_ivc.paint_uniform_color([1., 1., 0.])

#-----------------
vtk_hapath = os.path.join(getDatasetpath(),getHApath())
geom_ha = createMesh(vtk_hapath)
mesh_ha= o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(geom_ha.vertices.copy()),
                                                triangles=o3d.utility.Vector3iVector(geom_ha.faces.copy()))
mesh_ha.compute_vertex_normals()
mesh_ha.paint_uniform_color([1., 0.17254901960784313725490196078431, 0.28235294117647058823529411764706])
#-----------------
vtk_leftkidney = os.path.join(getDatasetpath(),getLeftKidneypath())
geom_left = createMesh(vtk_leftkidney)
mesh_left = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(geom_left.vertices.copy()),
                                                triangles=o3d.utility.Vector3iVector(geom_left.faces.copy()))
mesh_left.compute_vertex_normals()
mesh_left.paint_uniform_color([0.33333, 0., 0.4980392])
#-----------------
vtk_rightkidney = os.path.join(getDatasetpath(),getRightKidneypath())
geom_right = createMesh(vtk_rightkidney)
mesh_right = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(geom_right.vertices.copy()),
                                                triangles=o3d.utility.Vector3iVector(geom_right.faces.copy()))
mesh_right.compute_vertex_normals()
mesh_right.paint_uniform_color([0.66666, 0., 0.33333])
#-----------------
vtk_heartpath = os.path.join(getDatasetpath(), getHeartpath())
geom_heart = createMesh(vtk_heartpath)
mesh_heart = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(geom_heart.vertices.copy()),
                                                triangles=o3d.utility.Vector3iVector(geom_heart.faces.copy()))
mesh_heart.compute_vertex_normals()
mesh_heart.paint_uniform_color([0.66666, 0., 0.33333])

#-----------------
vtk_skinpath=os.path.join(getDatasetpath(),getSkinfilepath())                
geom_skin = createMesh(vtk_skinpath)   

mesh_skin= o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(geom_skin.vertices.copy()),
                                                triangles=o3d.utility.Vector3iVector(geom_skin.faces.copy()))
mesh_skin.compute_vertex_normals()
mesh_skin.paint_uniform_color([0.5, .5, 1.])

#-----------------
vtk_leasionpath = os.path.join(getDatasetpath(),getLeasionfilepath())
geom_leasion = createMesh(vtk_leasionpath)      
#split the leasion
# Mesh_leasion_objs = geom_leasion.split()
# Mesh_leasion_objs[3].show()
# Mesh_leasion_objs[3].unmerge_vertices()
mesh_leasion= o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(geom_leasion.vertices.copy()),
                                                    triangles=o3d.utility.Vector3iVector(geom_leasion.faces.copy()))
mesh_leasion.compute_vertex_normals()
color = list(np.random.choice(range(255), size=3))
color[0] /= 255
color[1] /= 255
color[2] /= 255
mesh_leasion.paint_uniform_color(color)
#-----------------
vtk_skeltonpath=os.path.join(getDatasetpath(),getSkeltonfilepath())                
geom_skelton = createMesh(vtk_skeltonpath)   
mesh_skelton= o3d.cpu.pybind.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(geom_skelton.vertices.copy()),
                                                    triangles=o3d.utility.Vector3iVector(geom_skelton.faces.copy()))
mesh_skelton.compute_vertex_normals()
mesh_skelton.paint_uniform_color([0.2, 0.706, 0]) 
#-----------------

o3d.visualization.draw_geometries([mesh_hv,mesh_pv,mesh_skelton,mesh_aorta,mesh_skin,mesh_ivc,mesh_ha,mesh_left,mesh_right,mesh_heart,mesh_leasion])

#-----------
scene = o3d.t.geometry.RaycastingScene()
#-----------
skelton_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_skelton)
skelton_id = scene.add_triangles(skelton_mesh)

hv_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_hv)
hv_id = scene.add_triangles(hv_mesh)

pv_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_pv)
pv_id = scene.add_triangles(pv_mesh)

aorta_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_aorta)
aorta_id = scene.add_triangles(aorta_mesh)

ivc_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_ivc)
ivc_id = scene.add_triangles(ivc_mesh)

ha_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_ha)
ha_id = scene.add_triangles(ha_mesh)

left_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_left)
left_id = scene.add_triangles(left_mesh)

right_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_right)
right_id = scene.add_triangles(right_mesh)

heart_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_heart)
heart_id = scene.add_triangles(heart_mesh)
#----------
pcd = mesh_skin.sample_points_poisson_disk(5000)
pcd.estimate_normals()
pcd.orient_normals_consistent_tangent_plane(1)
pcd.paint_uniform_color([0., 1., 0.])
o3d.visualization.draw_geometries([pcd])
#---------

multilable_file = os.path.join(getDatasetpath(),getMultilabelpath())
vol = volume.Volume(multilable_file)
v = np.zeros((4,4))
v[0,0] = 1/vol.dim_size[0]
v[1,1] = 1/vol.dim_size[1]
v[2,2] = 1/vol.dim_size[2]
v[3,3] = 1
A = np.dot(vol.model_mat,v)
las_ras=np.zeros((4,4))#shall be matrix
las_ras[0,0]=-1
las_ras[1,1]=-1
las_ras[2,2]=1
las_ras[3,3]= 1
#-----------------
points_w=[]
spheres_list=[]
file_path = os.path.join(getSphearPath(),getSphearfilepath())
with open(file_path, 'r') as file:
    for line in file:
        x, y, z, radius = map(float, line.strip().split(','))
        pos = [x,y,z,1.0]
        voxel = np.transpose(np.array(pos))
        wspace = np.dot(A,voxel)
        tmp = np.dot(np.array(las_ras),wspace)
        points_w.append(tmp)
        s=createSphere(tmp,radius)
        spheres_list.append(s)
#
hit_pcd_list=[]
nonhit_pcd_list=[]
pivot_pcd_list=[]
hit_lineset_list=[]
nonhit_lineset_list=[]
nonhit_len_list=[]
#rays
ray_correspondences=[]
ray_lineset_points=[]

for p in points_w:
    pivot = [p[0],p[1],p[2]]
    #--------------
    ray_direction = [0,0,1]
    for i in range(len(pcd.points)):
        ray_direction = [(pcd.points[i][0] - pivot[0]),
                (pcd.points[i][1] - pivot[1]),
                (pcd.points[i][2] - pivot[2])]    
        if(i==0):
            rays = o3d.core.Tensor([[pivot[0],pivot[1],pivot[2],ray_direction[0],ray_direction[1],ray_direction[2]]],dtype=o3d.core.Dtype.Float32)        
            ray_lineset_points.append([pivot[0],pivot[1],pivot[2]])
        else:
            r=o3d.core.Tensor([[pivot[0],pivot[1],pivot[2],ray_direction[0],ray_direction[1],ray_direction[2]]],dtype=o3d.core.Dtype.Float32)
            rays = rays.append(r,axis=0)
            ray_lineset_points.append([pcd.points[i][0],pcd.points[i][1],pcd.points[i][2]])
            ray_correspondences.append((0, i))

    #-----------
    ray_line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(ray_lineset_points),
        lines=o3d.utility.Vector2iVector(ray_correspondences),
    )

    color = [1., 0., 1.]
    colors = [color for i in range(len(ray_correspondences))]
    ray_line_set.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([ray_line_set])
    ans = scene.cast_rays(rays)
    hit = ans['t_hit'].isfinite()
    not_hit = ans['t_hit'].isinf()
    points = rays[hit][:,:3] + rays[hit][:,3:]*ans['t_hit'][hit].reshape((-1,1))
    hit_points = points.numpy()

    nohit_points =rays[not_hit][:,:3]+ rays[not_hit][:,3:]
    print(f"nohit_tensor={nohit_points}")

    nohit_pcd = o3d.t.geometry.PointCloud(nohit_points)
    nohit_pcd.paint_uniform_color([0., 1., 0.])
    nonhit_pcd_list.append(nohit_pcd.to_legacy())
    
    #------ hit point cloud
    pcd_points = o3d.t.geometry.PointCloud(points)
    pcd_points.paint_uniform_color([1., 0., 0.])
    hit_pcd_list.append(pcd_points.to_legacy())

    #------- for display
    pivot_pcd=o3d.t.geometry.PointCloud(o3d.core.Tensor([[pivot[0],pivot[1],pivot[2]]],dtype=o3d.core.Dtype.Float32))
    pivot_pcd.paint_uniform_color([0., 1., 0.])
    pivot_pcd_list.append(pivot_pcd.to_legacy())

    
    #-------
    nohit_correspondences=[]
    nohit_lineset_points=[]
    lineset_points=[]
    correspondences=[]
    nohit_lineset_points.append([pivot[0],pivot[1],pivot[2]])
    lineset_points.append([pivot[0],pivot[1],pivot[2]])

    #hit points
    for i in range(len(hit_points)):
        lineset_points.append([hit_points[i][0],hit_points[i][1],hit_points[i][2]])
        correspondences.append((0, i))
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(lineset_points),
        lines=o3d.utility.Vector2iVector(correspondences),
    )
    # color = list(np.random.choice(range(255), size=3))
    # color[0] /= 255
    # color[1] /= 255
    # color[2] /= 255
    color = [0., 0., 1.]
    colors = [color for i in range(len(correspondences))]
    line_set.colors = o3d.utility.Vector3dVector(colors)
   
    hit_lineset_list.append(line_set)

    #non hits
    nohit_points = nohit_points.numpy()
    for i in range(len(nohit_points)):
        nohit_lineset_points.append([nohit_points[i][0],nohit_points[i][1],nohit_points[i][2]])
        nohit_correspondences.append((0, i))
    nohit_line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(nohit_lineset_points),
        lines=o3d.utility.Vector2iVector(nohit_correspondences),
    )
    # color = list(np.random.choice(range(255), size=3))
    # color[0] /= 255
    # color[1] /= 255
    # color[2] /= 255
    color = [0., 0., 1.]
    nohit_colors = [color for i in range(len(nohit_correspondences))]
    nohit_line_set.colors = o3d.utility.Vector3dVector(nohit_colors)
    
    #--------
    nonhit_lineset_list.append(nohit_line_set)
    #o3d.visualization.draw_geometries([nohit_pcd.to_legacy()])
    #o3d.visualization.draw_geometries([nohit_pcd.to_legacy(),pcd_points.to_legacy(),pivot_pcd.to_legacy()])
    
    
    o3d.visualization.draw_geometries([line_set,nohit_line_set])
    
    color = [0., 1., 0.]
    nohit_colors = [color for i in range(len(nohit_correspondences))]
    nohit_line_set.colors = o3d.utility.Vector3dVector(nohit_colors)
    #o3d.visualization.draw_geometries([nohit_line_set])

    nonhit_numpy = rays[not_hit][:,:3].numpy()
    nonhit_dir = rays[not_hit][:,3:].numpy()
    nonhit_len=[]
    for idx in range(len(nonhit_numpy)):
        v= math.sqrt(nonhit_dir[idx][0]**2+nonhit_dir[idx][1]**2+nonhit_dir[idx][2]**2)
        print(f"v={v}")
        arr = np.append(nonhit_numpy[idx]+nonhit_dir[idx],[v])
        #arr = np.append([idx],[v])
        print(f"arr={arr}")
        nonhit_len.append(arr)
    
    nonhit_len_list.append(nonhit_len)

organs=[mesh_hv,mesh_pv,mesh_aorta,mesh_ivc,mesh_ha,mesh_left,mesh_right,mesh_heart,mesh_skelton,mesh_leasion]
o3d.visualization.draw_geometries(organs+nonhit_pcd_list+hit_pcd_list+pivot_pcd_list+hit_lineset_list+nonhit_lineset_list)

shortest_dist_list=[]
for nohit_len in nonhit_len_list:
    min = 99999.0
    shortest_dist = []
    for iter in range(len(nonhit_len)):
        v = nonhit_len[iter][3]
        if(v< min):
            min=v
            arr = np.array(nonhit_len[iter])
            shortest_dist=arr
    shortest_dist_list.append(shortest_dist)
    print(f"min={min}")  
    print(f"shortest_dist={shortest_dist}") 

  
shortest_line_set_list=[]
i = 0
for shortest_dist in shortest_dist_list:
    pivot = points_w[i]
    shortest_lineset_points = [
        [pivot[0],pivot[1],pivot[2]],
        [shortest_dist[0],shortest_dist[1],shortest_dist[2]]    
    ]
    shortest_lines = [
        [0, 1]   
    ]
    shortest_colors = [[1, 0, 0] for i in range(len(shortest_lines))]
    shortest_line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(shortest_lineset_points),
        lines=o3d.utility.Vector2iVector(shortest_lines),
    )
    shortest_line_set.colors = o3d.utility.Vector3dVector(shortest_colors)
    i+=1
    shortest_line_set_list.append(shortest_line_set)

#++++++++++++++++++++++++++++++++++++++
# mesh_leasion_0 = Mesh_leasion_objs[3]
# center,radius,error = trimesh.nsphere.fit_nsphere(mesh_leasion_0.vertices)
# sphere = createSphere(center,radius)
#spheres_list.append(sphere)
#sphere_o3d=[]
#for sphere in spheres_list:
# mesh_sphere= o3d.cpu.pybind.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(sphere.vertices.copy()),
#                                                 triangles=o3d.utility.Vector3iVector(sphere.faces.copy()))

# color = list(np.random.choice(range(255), size=3))
# color[0] /= 255
# color[1] /= 255
# color[2] /= 255
# mesh_sphere.paint_uniform_color(color)
 #   sphere_o3d.append(mesh_sphere)
    
organs_skleton=[mesh_skelton,mesh_leasion]
o3d.visualization.draw_geometries(organs_skleton+hit_pcd_list+shortest_line_set_list+nonhit_pcd_list)