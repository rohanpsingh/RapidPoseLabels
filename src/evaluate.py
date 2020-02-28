import numpy as np
import argparse
import open3d as o3d
import transforms3d.quaternions as tfq
import transforms3d.affines as tfa

def visualize_keypoints(X, Y, T):
    vis_mesh_list = []
    vis_mesh_list.append(o3d.geometry.TriangleMesh.create_coordinate_frame(0.2))

    for pt in X:
        keypt_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.006)
        pt = np.dot(T[:3,:3], pt) + T[:3,3]
        keypt_mesh.translate(pt)
        keypt_mesh.paint_uniform_color([0.1, 0.1, 0.7])
        vis_mesh_list.append(keypt_mesh)

    for pt in Y:
        keypt_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        keypt_mesh.translate(pt)
        keypt_mesh.paint_uniform_color([0.7, 0.1, 0.1])
        vis_mesh_list.append(keypt_mesh)

    o3d.visualization.draw_geometries(vis_mesh_list)

def procrustes(X, Y):
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

    traceTA = s.sum()
    b = 1
    d = 1 + ssY/ssX - 2 * traceTA * normY / normX
    Z = normY*np.dot(Y0, T) + muX
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)
    tform = {'rotation':T, 'scale':b, 'translation':c}
    return d, Z, tform

def get_object_definition(pp_file):
    with open(pp_file, 'r') as file:
        lines = [[float(i.rsplit('=')[1].rsplit('"')[1]) for i in line.split()[1:4]] for line in file.readlines()[8:-1]]
    return np.asarray(lines)
        
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True)
    ap.add_argument("--input", required=True)
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

    obj_def = get_object_definition(opt.points)
    obj_est = opt_output[(num_scenes-1)*7:].reshape((num_keypts, 3))
    d, Z, tform = procrustes(obj_def, obj_est)

    T = tfa.compose(tform['translation'], np.linalg.inv(tform['rotation']), np.ones(3))
    
    err = []
    for (x, y) in zip(obj_def, obj_est):
        y = (T[:3,:3].dot(y) + T[:3,3])
        err.append(((x-y)**2).sum()**0.5)
 
    print("Mean error: ", sum(err)/len(err))
    print("---")
    if visualize:
        visualize_keypoints(obj_est, obj_def, T)
