import numpy as np
import scipy.optimize
import transforms3d.quaternions as tfq
import transforms3d.affines as tfa

def predict(ref_kpts, scene_t, scene_q, scene_P, select_mat):

    #generate reference keypoint vector
    ref_kpts_vec = ref_kpts.transpose(0,2,1).flatten()

    len_ts = scene_t[1:].size
    len_qs = scene_q[1:].size
    len_Ps = scene_P.size
    
    #constraint function
    def cons_func(x):
        qs = x[len_ts : len_ts+len_qs].reshape(scene_q[1:,:].shape)
        return np.asarray([(np.linalg.norm(tfq.qnorm(q)**2-1)) for q in qs])
        
    #rotate point using quaternion function
    def rotate(q, pt):
        qmult1 = tfq.qmult(q,(np.append(0, pt).transpose()))
        qmult2 = tfq.qmult(qmult1,tfq.qconjugate(q))
        return (np.asarray(qmult2))[1:]

    #error function
    def error_func(x):
        ts = np.concatenate((np.array([[0,0,0]]), x[ : len_ts].reshape(scene_t[1:, :].shape)))
        qs = np.concatenate((np.array([[1,0,0,0]]), x[len_ts : len_ts+len_qs].reshape(scene_q[1:, :].shape)))
        Ps = x[len_ts+len_qs : ].reshape(scene_P.shape)
        list_pts = []
        for (q, t, pts) in zip(qs, ts, Ps[np.newaxis,:].repeat(qs.shape[0], axis=0)):
            l = [(rotate(q, pt) + t) for pt in pts]
            list_pts = np.append(list_pts, l)
        err = np.linalg.norm(select_mat.dot(list_pts) - select_mat.dot(ref_kpts_vec))**2
        return err

    #define constraint
    cons = ({'type': 'eq',
             'fun' : cons_func})

    #generate initial vector
    ini_vals = np.concatenate((scene_t[1:, :].flatten(), scene_q[1:, :].flatten(), scene_P.flatten()))

    #perform optimization
    res = scipy.optimize.minimize(error_func, ini_vals, constraints=cons, method='SLSQP', tol=1e-9, options={'disp':True,'ftol':1e-9, 'maxiter':1000})

    return res
