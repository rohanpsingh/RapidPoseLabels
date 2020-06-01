import numpy as np
import argparse
import open3d as o3d
import transforms3d.quaternions as tfq
import transforms3d.affines as tfa
import sys
from skimage.util.shape import view_as_blocks

np.set_printoptions(threshold=sys.maxsize, linewidth=700)
np.set_printoptions(precision=4, suppress=True)

def visualize_keypoints(X, Y, T, model_off_path, scene_ply_path):
    vis_mesh_list = []
    vis_mesh_list.append(o3d.geometry.TriangleMesh.create_coordinate_frame(0.2))

    if scene_ply_path is not None:
        sce_mesh = o3d.io.read_point_cloud(scene_ply_path)
        sce_mesh.transform(T)
        #sce_mesh = sce_mesh.voxel_down_sample(voxel_size=0.002)
        vis_mesh_list.append(sce_mesh)

    if model_off_path is not None:
        obj_mesh = o3d.io.read_triangle_mesh(model_off_path)
        obj_mesh.paint_uniform_color([0.6, 0.6, 0.1])
        obj_cloud = obj_mesh.sample_points_uniformly(5000)
        vis_mesh_list.append(obj_cloud)

    for pt in X:
        keypt_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.006)
        pt = np.dot(T[:3,:3], pt) + T[:3,3]
        keypt_mesh.translate(pt)
        keypt_mesh.paint_uniform_color([0.7, 0.1, 0.1]) #red
        #keypt_mesh.paint_uniform_color([0.1, 0.1, 0.7]) #blue
        vis_mesh_list.append(keypt_mesh)

    for pt in Y:
        keypt_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        keypt_mesh.translate(pt)
        #keypt_mesh.paint_uniform_color([0.7, 0.1, 0.1]) #red
        keypt_mesh.paint_uniform_color([0.1, 0.1, 0.7]) #blue
        vis_mesh_list.append(keypt_mesh)

    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(10.0, 0.0)
        return False

    #o3d.visualization.draw_geometries(vis_mesh_list)
    o3d.visualization.draw_geometries_with_animation_callback(vis_mesh_list,rotate_view)

def procrustes(X, Y, scaling=True, reflection='best'):
    n,m = X.shape
    ny,my = Y.shape
    muX = X.mean(0)
    muY = Y.mean(0)
    X0 = X - muX
    Y0 = Y - muY
    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)
    X0 /= normX
    Y0 /= normY
    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':
        have_reflection = np.linalg.det(T) < 0
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)
    traceTA = s.sum()
    if scaling:
        b = traceTA * normX / normY
        d = 1 - traceTA**2
        Z = normX*traceTA*np.dot(Y0, T) + muX
    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)
    tform = {'rotation':T, 'scale':b, 'translation':c}
    return d, Z, tform

def get_visibility(select):
    vec = []
    row, col = 0, 0
    while col<(select.shape[1]):
        m = select[row:row+3, col:col+3]
        if (m.shape[0]==0) or not (m == np.eye(3)).all():
            vec.append(0)
        else:
            vec.append(1)
            row+=3
        col+=3
    return vec

def get_object_definition(pp_file, vec):
    with open(pp_file, 'r') as file:
        lines = [[float(i.rsplit('=')[1].rsplit('"')[1]) for i in line.split()[1:4]] for line in file.readlines()[8:-1]]
    out = [kpt for v, kpt in zip(vec, lines) if v]
    return np.asarray(out)
        
def get_object_manual(kpts, vec):
    out = [kpt for v, kpt in zip(vec, kpts.transpose()) if v]
    return np.asarray(out)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--off", required=True)
    ap.add_argument("--ply", required=True)
    ap.add_argument("--visualize", required=True, default=True)
    opt = ap.parse_args()

    visualize    = (opt.visualize.lower()=='true')
    
    input_array  = np.load(opt.input)
    ref_keypts = input_array['ref']
    select_mat = input_array['sm']
    opt_output = input_array['res']

    num_scenes = ref_keypts.shape[0]
    num_keypts = ref_keypts.shape[2]
    print("Number of scenes: ", num_scenes)
    print("Number of keypts: ", num_keypts)

    sce_id = 0
    vis_vec = get_visibility(select_mat[:3*num_keypts,:3*num_keypts])
    obj_man = get_object_manual(ref_keypts[sce_id], vis_vec)
    obj_def = get_object_definition(opt.points, vis_vec)
    d, Z, tform = procrustes(obj_def, obj_man, False)

    T = tfa.compose(tform['translation'], np.linalg.inv(tform['rotation']), np.ones(3))
    
    obj_all = get_object_definition(opt.points, np.ones(num_keypts))
    obj_est = opt_output[(num_scenes-1)*7:].reshape((num_keypts, 3))
    err = []
    for (x, y) in zip(obj_all, obj_est):
        y = (T[:3,:3].dot(y) + T[:3,3])
        err.append(((x-y)**2).sum()**0.5)

    print("Mean error: ", sum(err)/len(err))
    print("---")
    if visualize:
        visualize_keypoints(obj_est, obj_all, T, opt.off, opt.ply)
