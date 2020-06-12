import sys
import numpy as np
import transforms3d.quaternions as tfq
import transforms3d.affines as tfa

def get_object_definition(pp_file, vis_vec):
    """
    Parse the sparse model file.
    Return numpy array of shape (Nx3).
    """
    points = []
    names = []
    with open(pp_file, 'r') as ppfile:
        lines = [line.strip().split()[1:] for line in ppfile.readlines() if line.strip().find('point')==1]
    for line in lines: 
        dict_ = {name:float(val.strip('"')) for item in line for name,val in [item.strip('/>').split('=')]}
        points.append([dict_['x'], dict_['y'], dict_['z']])
        names.append(int(dict_['name']))
    #keypoints are unique, so they must be sorted according to ID
    out = np.asarray([points[i] for i,v in zip(np.argsort(names), vis_vec) if v])
    return out
        
def get_object_manual(kpts, vis_vec):
    out = [kpt for v, kpt in zip(vis_vec, kpts.transpose()) if v]
    return np.asarray(out)

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

def predict(model_file, labeled_points, selection_vector):
    num_scenes = labeled_points.shape[0]
    num_keypts = labeled_points.shape[2]
    selection_arr = np.asarray(selection_vector).reshape(num_scenes, num_keypts)
    poses_vec = []
    #get object pose and relative scene transformations using Procrustes analysis
    for point_set, visibility in zip(labeled_points, selection_arr):
        obj_man = get_object_manual(point_set, visibility)
        obj_def = get_object_definition(model_file, visibility)
        _, _, tform = procrustes(obj_def, obj_man, False)
        obj_pose = tfa.compose(tform['translation'], tform['rotation'], np.ones(3))
        poses_vec.append(obj_pose)
    #since object has not moved between scenes...
    output_vec = [poses_vec[0].dot(np.linalg.inv(tf)) for tf in poses_vec]
    return True, output_vec
