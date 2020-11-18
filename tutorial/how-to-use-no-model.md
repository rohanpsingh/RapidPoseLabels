## Tutorial
### How To Use (when you don't have an object model)

### Step 1. Choose the keypoints
For the tutorial we are going to use [this sample dataset](https://o365tsukuba-my.sharepoint.com/:u:/g/personal/s1920861_u_tsukuba_ac_jp/EY0pnDY7LBBKi_9PFfOH5hoBRcxrUbPJMB0LCUL674UI-Q?e=GTKGla). Download and move it into rapidposelabels/data/.  

It consists of the Oolong Tea object in 5 different RGB-D scenes. The scene meshes and camera trajectory have already been generated using ElasticFusion and available with the dataset.
Choosing the location and number of keypoints on the object is up to the user. But, for the sake of this exercise, let's pick the following 10 points:

<p align="left">
  <img src="./docs/img/oolong_points.png" alt="oolong" width="30%">
  <img src="./docs/img/oolong.gif" alt="oolong" width="20%">
<p>

### Step 2. Launch the GUI
Assuming you decided to go with the 10 keypoints above, fire up the GUI with this command:  

```
$ python main.py --output tutorial_no_model --keypoints 10
```

### Step 3. Load Dataset
Once the main window pops up, go to 'Load Dataset' under the 'File' menu.  
This will show all the loaded scenes under the chosen directory in the 'Scenes' dock window. Next, click on 'Next scene' and 'Load New Image' to load a random frame from the first scene.  

<p align="center">
  <img src="./docs/img/load_dataset.gif" alt="gui" width="50%">
<p>

### Step 4. Start the annotation
Start to label the chosen points sequentially in the GUI. Use the 'Load New Image' or the slider to display new images from the scene. Mark as many points you can in this scene and click 'Next scene' when you're done. Note that  

- If you don't see a keypoint and want to move to the next one, click 'Skip keypoint'. This create a [-1, -1] keypoint in the dock window.  

- Clicking on 'Next scene' will fill up the remaining points with [-1, -1], i.e. remaining points will be skipped.  

- (If a point *vanishes* in the GUI when you load a new image, this means the depth value at that location could not be obtained. Please click 'Scene reset' and mark all points in this scene again when this happens.)

<p align="center">
  <img src="./docs/img/annotate_oolong_scene.gif" alt="gui" width="50%">
<p>
