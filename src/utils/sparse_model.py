import os
import numpy as np

class SparseModel:
    def writer(self, object_model, file_path="sparse_model.txt"):
        """
        Function to save the generated sparse model in the following format
        <point x="000000" y="000000" z="000000" name="0"/> in a .txt file.
        Also writes some meta data.
        Input arguments:
        object_model - (Nx3) numpy array holding 3D positions of all keypoints
                       where N is the number of keypoints on the model.
        filename     - name of the output file inside the output directory.
                       (sparse_model.txt by default)
        """
        out_str = ["<SparseObjectPoints>"]
        for idx, point in enumerate(object_model):
            kpt_str = str("\t<point x=\"{}\" y=\"{}\" z=\"{}\"".format(*list(point)))
            kpt_str = kpt_str + str(" name=\"{}\"/>".format(idx))
            out_str.append(kpt_str)
        out_str.append("</SparseObjectPoints>\n")
        with open(os.path.join(file_path), 'a') as out_file:
            out_file.write("\n".join(out_str))
        return

    def reader(self, file_path):
        """
        Function to parse the sparse model file.
        (This function can also parse *.pp PickedPoints file generated
        through MeshLab using the PickPoints too.)
        Return numpy array of shape (Nx3).
        """
        if not os.path.exists(file_path):
            raise Exception('File path %s does not exists' % file_path)
        points = []
        names = []
        with open(file_path, 'r') as in_file:
            lines = [line.strip().split()[1:] for line in in_file.readlines() if line.strip().find('point')==1]
        for line in lines:
            dict_ = {name:float(val.strip('"')) for item in line for name,val in [item.strip('/>').split('=')]}
            points.append([dict_['x'], dict_['y'], dict_['z']])
            names.append(int(dict_['name']))
        #keypoints are unique, so they must be sorted according to ID
        out = np.asarray([points[i] for i in np.argsort(names)])
        return out

    def grasp_writer(self, mat, file_path="grasp_point.txt"):
        """
        Function to write the generated grasp point (as a 4x4 transformation)
        to disk in a .txt file.
        Input arguments:
        mat          - (4x4) numpy array holding the affine transformation
                       to reach the grasp point with respect to the origin
        filename     - name of the output file inside the output directory.
                       (sparse_model.txt by default)
        """
        out_str = ["<GraspPoint>"]
        for idx, row in enumerate(mat):
            kpt_str = str("\t{}".format(list(row)))
            out_str.append(kpt_str)
        out_str.append("</GraspPoint>\n")
        with open(os.path.join(file_path), 'a') as out_file:
            out_file.write("\n".join(out_str))
        return
