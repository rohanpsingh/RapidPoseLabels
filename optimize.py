import numpy as np
import scipy.optimize
import transforms3d as tf

def predict(ref_kpts, init_vec, select_mat):
    local_keypt_vec  = []
        
    #generate local keypoint vector
    for keypts in ref_kpts:
        for keypt in keypts.transpose():
            local_keypt_vec.append(keypt)
    local_keypt_vec = np.asarray(local_keypt_vec).flatten()
    local_keypt_vec = select_mat.dot(local_keypt_vec)

    scene_t = init_vec[0]
    scene_q = init_vec[1]
    scene_P = init_vec[2]

    len_ts = scene_t.size
    len_qs = scene_q.size
    len_Ps = scene_P.size
    

    def cons_func(x):
        qs = x[(len_ts-3):len_ts + len_qs - 7]
        err = 0
        for i in range(int(qs.size/4)):
            q  =  x[(len_ts-3):len_ts + len_qs - 7][i*4:(i+1)*4]
            qc = tf.quaternions.qconjugate(q)
            qm = qc.dot(q)
            err += (qm - 1)
        return err
        
    def rotate(q, pt):
        qmult1 = tf.quaternions.qmult(q,(np.append(0, pt).transpose()))
        qmult2 = tf.quaternions.qmult(qmult1,(np.asarray([q[0]] + list(-1*q[1:]))))
        return (np.asarray(qmult2))[1:]

    def error_func(x):
        ts = x[:(len_ts-3)].reshape(scene_t[1:, :].shape)
        qs = x[(len_ts-3):len_ts + len_qs - 7].reshape(scene_q[1:, :].shape)
        Ps = x[len_ts + len_qs - 7:].reshape(scene_P.shape)
        list_pts = np.array([])
        for pt in Ps:
            list_pts = np.append(list_pts, pt)
        for (q, t, pts) in zip(qs, ts, Ps[np.newaxis,:].repeat(qs.shape[0], axis=0)):
            for pt in pts:
                pt_tx = rotate(q, pt) + t
                list_pts = np.append(list_pts, pt_tx)
        err = np.linalg.norm(select_mat.dot(list_pts) - local_keypt_vec, ord=2)**2
        return err

    cons = ({'type': 'eq',
             'fun' : cons_func})

    init_vec = np.concatenate((scene_t[1:, :].flatten().copy(), scene_q[1:, :].flatten().copy(), scene_P.flatten().copy()))
    #res = scipy.optimize.minimize(error_func, init_vec, method='BFGS', options={'disp':True})
    res = scipy.optimize.minimize(error_func, init_vec, constraints=cons, method='SLSQP', options={'disp':True})

    print("--------\n--------\n--------")
    if res.success:
        output_vec = res.x
        out_ts = output_vec[:(len_ts-3)].reshape(scene_t[1:, :].shape)
        out_qs = output_vec[(len_ts-3):(len_ts + len_qs -7)].reshape(scene_q[1:, :].shape)
        out_Ps = output_vec[(len_ts + len_qs-7):].reshape(scene_P.shape)
        object_model = out_Ps.transpose()
        print("SUCCESS")
        #np.set_printoptions(precision=3, suppress=True)
        print("Output translations:\n", out_ts)
        print("Output quaternions:\n", out_qs)
        print("Object Model:\n", object_model)
        return True
    else:
        return False


def test_with_dummy():
    obj_model = np.array([[0, 0, 0],
                          [0, 1, 0],
                          [1, 0, 0],
                          [1, 1, 0],
                          [0, 0, 1],
                          [0, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]])
    list_of_tfs = []
    t_ = 3
    input_ts = np.array([[0,  0,    0],
                         [1*t_, 1*t_, 0],
                         [1*t_, 0, 1*t_],
                         [1*t_,-1*t_, 0],
                         [1*t_, 0,-1*t_],
                         [-1*t_, 1*t_, 0],
                         [1*t_, -1*t_, 1*t_],
                         [-1*t_,-1*t_, 0]])

    ns = 8
    nk = 8
    input_ts  = input_ts[:ns, :]
    obj_model = obj_model[:ns, :]

    for input_t in input_ts:
        list_of_tfs.append(tf.affines.compose(input_t, tf.quaternions.quat2mat(np.array([1, 0, 0, 0])), np.ones(3)))

    ref_kpts = np.zeros((ns, 3, nk))
    for (scene_id, t) in enumerate(list_of_tfs):
        for (keypt_id, pt) in enumerate(obj_model):
            pt = np.append(pt, 1)
            ref_kpts[scene_id, :, keypt_id] = t.dot(pt)[:3]

    scene_t = np.array([[0, 0, 0]]).repeat(ref_kpts.shape[0], axis=0)
    scene_q = np.array([[1, 0, 0, 0]]).repeat(ref_kpts.shape[0], axis=0)
    scene_P = np.array([[0, 0, 0]]).repeat(ref_kpts.shape[2], axis=0)

    selection_matrix = np.eye((ns*nk*3))
    predict(ref_kpts, (scene_t, scene_q, scene_P), selection_matrix)
    print("Input points:\n", ref_kpts)

if __name__ == '__main__':
    test_with_dummy()
