import os
import open3d as o3d
import argparse
import tkinter as tk


def visualize(path_to_dataset):
    # Get list of directories
    list_of_scene_dirs = [d for d in os.listdir(path_to_dataset) if os.path.isdir(os.path.join(path_to_dataset, d))]
    list_of_scene_dirs.sort()
    list_of_scene_meshes = [os.path.join(path_to_dataset, d, d+'.ply') for d in list_of_scene_dirs]
    print("Number of scenes: ", len(list_of_scene_dirs))
    print("List of scenes: ", list_of_scene_dirs)
    print("List of scenes meshes: ", list_of_scene_meshes)

    # Create grid
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    # Visualize
    visualizers = []
    scene_meshes = []
    for idx, scene_mesh_path in enumerate(list_of_scene_meshes):
        scene_mesh = o3d.io.read_point_cloud(scene_mesh_path)
        scene_meshes.append(scene_mesh)
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=repr(idx), width=960, height=540, left=0, top=0)
        vis.add_geometry(scene_mesh)
        visualizers.append(vis)

    while True:
        for vis, scene_mesh in zip(visualizers, scene_meshes):
            vis.update_geometry(scene_mesh)
            vis.poll_events()
            vis.update_renderer()
    return

if __name__ == '__main__':

    # Get command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help='path to dataset dir')
    opt = ap.parse_args()

    visualize(opt.dataset)
