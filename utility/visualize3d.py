import numpy as np
import argparse
import open3d as o3d
import transforms3d.affines as tfa
import os, sys
from skimage.util.shape import view_as_blocks
import evaluate3d

def process(input_array, pp, off):
    ref_kpts   = input_array['ref']
    select_mat = input_array['sm']
    opt_output = input_array['res']
    num_scenes = ref_kpts.shape[0]
    num_keypts = ref_kpts.shape[2]
    print("Number of scenes: ", num_scenes)
    print("Number of keypts: ", num_keypts)

    sce_id  = 0
    vis_vec = evaluate3d.get_visibility(select_mat[:3*num_keypts,:3*num_keypts])
    obj_man = evaluate3d.get_object_manual(ref_kpts[sce_id], vis_vec)
    obj_def = evaluate3d.get_object_definition(pp, vis_vec)
    d, Z, tform = evaluate3d.procrustes(obj_def, obj_man, False)

    T = tfa.compose(tform['translation'], np.linalg.inv(tform['rotation']), np.ones(3))
    
    obj_all = evaluate3d.get_object_definition(pp, np.ones(num_keypts))
    obj_est = opt_output[(num_scenes-1)*7:].reshape((num_keypts, 3))
    err = []
    for (x, y) in zip(obj_all, obj_est):
        y = (T[:3,:3].dot(y) + T[:3,3])
        err.append(((x-y)**2).sum()**0.5)

    print("Mean error: ", sum(err)/len(err))
    if visualize:
        evaluate3d.visualize_keypoints(obj_est, obj_all, T, off, None)
    print("---")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--off", required=True)
    ap.add_argument("--visualize", required=False, default='false')
    opt = ap.parse_args()

    inp_pp_dir   = opt.points
    inp_off_dir  = opt.off
    inp_npz_dir  = opt.input
    visualize    = False
    visualize    = (opt.visualize.lower()=='true')

    for npz in os.listdir(inp_npz_dir):
        input_array  = np.load(os.path.join(inp_npz_dir, npz))
        obj_name      = os.path.splitext(npz)[0]
        pp_file_path  = os.path.join(inp_pp_dir, obj_name + '_picked_points.pp')
        off_file_path = os.path.join(inp_off_dir, obj_name + '.off')
        print("Object:----->", obj_name)
        process(input_array, pp_file_path, off_file_path)

    sys.exit(-1)
