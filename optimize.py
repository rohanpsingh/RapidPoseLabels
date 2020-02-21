import numpy as np
import scipy.optimize
import transforms3d.quaternions as tfq
import transforms3d.affines as tfa
import sys
import open3d as o3d

def predict(ref_kpts, scene_t, scene_q, scene_P, select_mat):
    local_keypt_vec  = []

    #generate local keypoint vector
    for keypts in ref_kpts:
        for keypt in keypts.transpose():
            local_keypt_vec.append(keypt)
    local_keypt_vec = np.asarray(local_keypt_vec).flatten()

    len_ts = scene_t[1:].size
    len_qs = scene_q[1:].size
    len_Ps = scene_P.size
    
    def cons_func(x):
        qs = x[len_ts : len_ts+len_qs].reshape(scene_q[1:,:].shape)
        return np.asarray([(np.linalg.norm(tfq.qnorm(q)**2-1)) for q in qs])
        
    def rotate(q, pt):
        qmult1 = tfq.qmult(q,(np.append(0, pt).transpose()))
        qmult2 = tfq.qmult(qmult1,tfq.qconjugate(q))
        return (np.asarray(qmult2))[1:]

    def error_func(x):
        ts = x[ : len_ts].reshape(scene_t[1:, :].shape)
        qs = x[len_ts : len_ts+len_qs].reshape(scene_q[1:, :].shape)
        Ps = x[len_ts+len_qs : ].reshape(scene_P.shape)
        list_pts = Ps.flatten()
        for (q, t, pts) in zip(qs, ts, Ps[np.newaxis,:].repeat(qs.shape[0], axis=0)):
            for pt in pts:
                list_pts = np.append(list_pts, (rotate(q, pt) + t))
        err = np.linalg.norm(select_mat.dot(list_pts) - select_mat.dot(local_keypt_vec))**2
        return err

    cons = ({'type': 'eq',
             'fun' : cons_func})

    init_vals = np.concatenate((scene_t[1:, :].flatten(), scene_q[1:, :].flatten(), scene_P.flatten()))
    #res = scipy.optimize.minimize(error_func, init_vals, method='BFGS', options={'disp':True})
    res = scipy.optimize.minimize(error_func, init_vals, constraints=cons, method='SLSQP', tol=1e-9, options={'disp':True,'ftol':1e-8, 'maxiter':1000})
    #res = scipy.optimize.minimize(error_func, init_vals, constraints=cons, method='trust-constr', tol=1e-8, options={'disp':True, 'maxiter':400})

    if res.success:
        print("--------\n--------\n--------")
        output_vec = res.x
        np.savez("saved_tmp_output", res=output_vec)
        out_ts = output_vec[:len_ts].reshape(scene_t[1:, :].shape)
        out_qs = output_vec[len_ts:len_ts+len_qs].reshape(scene_q[1:, :].shape)
        out_Ps = output_vec[len_ts+len_qs:].reshape(scene_P.shape)
        object_model = out_Ps.transpose()
        print("SUCCESS")
        np.set_printoptions(precision=5, suppress=True)
        print("Output translations:\n", out_ts)
        print("Output quaternions:\n", out_qs)
        print("Object Model:\n", object_model, object_model.shape)
        print("--------\n--------\n--------")

    return res
