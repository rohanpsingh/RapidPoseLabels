import sys
import numpy as np
import transforms3d.quaternions as tfq
import transforms3d.affines as tfa

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

def predict(model_points, labeled_points, selection_vector):
    selection_arr = np.asarray(selection_vector).reshape(labeled_points.shape[:2])
    poses_vec = []
    #get object pose and relative scene transformations using Procrustes analysis
    for point_set, visibility in zip(labeled_points, selection_arr):
        manual_set = np.asarray([kpt for flag, kpt in zip(visibility, point_set) if flag])
        model_set  = np.asarray([kpt for flag, kpt in zip(visibility, model_points) if flag])
        _, _, tform = procrustes(model_set, manual_set, False)
        obj_pose = tfa.compose(tform['translation'], np.linalg.inv(tform['rotation']), np.ones(3))
        poses_vec.append(np.linalg.inv(obj_pose))
    return True, poses_vec
