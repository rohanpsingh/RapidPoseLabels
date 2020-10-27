




# Automated Data Annotation for 6-DoF Object Pose Estimation

This is a tool for rapid generation of labeled training dataset primarily for the purpose of training keypoint detector networks for full pose estimation of a rigid, non-articulated 3D object in RGB images. The code is based on our paper: *Rapid Pose Label Generation through Sparse Representation of Unknown Objects*, [Rohan P. Singh](https://github.com/rohanpsingh), [Mehdi Benallegue](https://github.com/mehdi-benallegue), Yusuke Yoshiyasu, Fumio Kanehiro. [under-review]


We provide a GUI to fetch minimal user input. Using the given software, we have been able to generate large, accurately--labeled, training datasets consisting of multiple objects in different scenes (environments with varying background conditions, illuminations, clutter etc.) using just a handheld RGB-D sensor in only a few hours, including the time involved in capturing the raw dataset. And ultimately, used the training dataset for training a bounding-box detector ([YOLOv3](https://github.com/AlexeyAB/darknet)) and a keypoint detector network ([ObjectKeypointTrainer](https://github.com/rohanpsingh/ObjectKeypointTrainer)).

The code in this repository forms Part-1 of the full software:
![pose-estimation-github](https://user-images.githubusercontent.com/16384313/84745705-ec04bf00-afef-11ea-9966-c88f24c9a3ba.png)

Links to other parts:
- Part-2: [ObjectKeypointTrainer](https://github.com/rohanpsingh/ObjectKeypointTrainer)
- Part-3: Not-yet-available

## Dependencies

All or several parts of the given Python 3.7.4 code are dependent on the following:
- OpenCV
- [open3d](http://www.open3d.org/docs/release/getting_started.html)
- [transforms3d](https://matthew-brett.github.io/transforms3d)

We recommend satisfying above dependencies to be able to use all scripts, though it should be possible to bypass some requirements depending to the use case. We recommend working in a [conda](https://docs.conda.io/en/latest/) environment.
### Other dependencies
For pre-processing of the raw dataset (extracting frames if you have a ROS bagfile and for dense 3D reconstruction) we rely on the following applications:
1. [bag-to-png](https://gist.github.com/rohanpsingh/9ac99c46aef8ccb618cdad18cd20e068)
2. [png-to-klg](https://github.com/HTLife/png_to_klg)
3. [ElasticFusion](https://github.com/mp3guy/ElasticFusion)

## Usage
### 1. Preparing the dataset(s)
We assume that using [bag-to-png](https://gist.github.com/rohanpsingh/9ac99c46aef8ccb618cdad18cd20e068), [png-to-klg](https://github.com/HTLife/png_to_klg) and [ElasticFusion](https://github.com/mp3guy/ElasticFusion), the user is able to generate a dataset directory tree which looks like follows:
```
dataset_dir/
├── wrench_tool_data/
│   ├── 00/
│	│	├── 00.klg
│	│	├── 00.ply
│	│	├── associations.txt
│	│	├── camera.poses
│	│	├── depth.txt
│	│	├── rgb.txt
│	│	├── rgb/
│	│	└── depth/
│   ├── 01/
│   ├── 02/
│   ├── 03/
│   ├── 04/
│   └── camera.txt
├── object_1_data/...
└── object_2_data/...
```
where ```camera.poses``` and ```00.ply``` are the camera trajectory and the dense scene generated by ElasticFusion respectively. Ideally, the user has collected raw dataset for different scenes/environments in directories ```00, 01, 02,...``` . ```camera.txt``` contains the camera intrinsics as ```fx fy cx cy```.

### 2. How to use
This should bring up the main GUI:
```
$ python main.py --dataset <path-to-dataset-dir> --keypoints <number-of-keypoints-to-be>
```
<p align="center">
<img src="https://user-images.githubusercontent.com/16384313/84734452-d5ed0380-afdb-11ea-88e8-cddb0b01c312.png" alt="GUI" width="80%">
<p>

#### a. If object model is NOT available
In the case where the user has no model of any kind for their object, the first step is to generate a sparse model using the GUI and then build a rough dense model using the ```join.py``` script. 

To create the sparse model, first choose about 6-10 points on the object. Since you need to click on these points in RGB images later, make sure the points are uniquely identifiable. Also make sure the points are well distributed around the object, so a few points are visible if you look at the object from any view. Remember the order and location of the chosen points and launch the GUI, setting the ```--keypoints``` argument equal to the number of chosen keypoints.

1. Click on "Create a new model".
2. Click on "Load New Image" and manually label all keypoints decided on the object which are visible.
3. Click on "Skip KeyPt" if keypoint is not visible (**Keypoint labeling is order sensitive**).
4. To shuffle, click on "Load New Image" again.
5. Click on "Next Scene" when you have clicked on as many points as you can see.
6. Repeat Steps 2-5 for each scene.
7. Click on "Compute".

If manual label was done maintaining the constraints described in the paper, the optimization step should succeed and produce a ```sparse_model.txt``` and ```saved_meta_data.npz``` in the output directory. The ```saved_meta_data.npz``` archive holds data of relative scene transformations and the manually clicked points with their IDs (important for generating the actual labels using ```generate.py``` and evaluation with respect to ground-truth, if available).

The user can now generate a dense model for their object like so:
```
$ python join.py --dataset <path-to-dataset-dir> --sparse <path-to-sparse-model.txt> --meta <path-to-saved-meta-data>
```
The script will end while throwing the Open3D Interactive Visualization window. Perform [manual cropping](http://www.open3d.org/docs/release/tutorial/visualization/interactive_visualization.html#crop-geometry) if required and save as ```dense.ply```.

#### b. If object model is available
If you already have a model file for your object (generated through CAD, scanners etc.), follow these steps to obtain ```sparse_model.txt```:

1. Open the model file in Meshlab.
2. Click on PickPoints (Edit > PickPoints).
3. Pick 6-10 keypoints on the object arbitrarily.
4. "Save" as sparse_model.txt.

Once ```sparse_model.txt``` has been generated for a particular object, it is easy to generate labels for any scene. This requires the user to uniquely localize at least 3 points defined in the sparse model in the scene.

1. Launch the GUI.
2. Click on "Use existing model".
3. Choose a previously generated ```sparse_model.txt``` (A Meshlab *.pp file can also be used in this step).
4. Click on "Load New Image".
5. Label at least 3 keypoints (that exist in the sparse model file).
6. Click on "Skip KeyPt" if keypoint is not visible (**Keypoint labeling is order sensitive**).
7. Click on "Compute".

"Compute" tries to solve an orthogonal Procrustes problem on the given manual clicks and the input sparse model file. This will generate the  ```saved_meta_data.npz``` again for the scenes for which labeling was done.

### 3. Generate the labels
Once (1) sparse model file, (2) dense model file and (3) scene transformations (in saved_meta_data.npz) are available, run the following command:
```
$ python generate.py --sparse <path-to-sparse-model> --dense <path-to-dense-model> --meta <path-to-saved-meta-data-npz> --dataset <path-to-dataset-dir> --output <path-to-output-directory> --visualize
```
That is it. This should be enough to generate keypoint labels for stacked-hourglass-training as described in [ObjectKeypointTrainer](https://github.com/rohanpsingh/ObjectKeypointTrainer), mask labels for training a pixel-wise segmentation and bounding-box labels for training a generic object detector. Other types of labels are possible too, please create Issue or Pull request :)

### (Optional) To define a Grasp Point
The GUI can be used to define a 4x4 affine transformation from the origin of the sparse model to a grasp point, as desirable.
1. Launch the GUI (no need for ```--keypoint``` argument this time)
2. Select "Define grasp point"
3. Browse to the sparse model file.
4. Load an image from the ```00``` scene, and click on 2 points.

The first clicked point defines the position of grasp point and the second point decides the orientation.

TODO: Currently, this works only if user defines grasp point in the "first scene". This is because the sparse object model is defined with respect to "first viewpoint" in "first scene". Ideally, the user should be able to load any scene's PLY and define the grasp point in an interactive 3D environment.

TODO: Grasp point orientation is sometimes affected by noise in scene point cloud.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
