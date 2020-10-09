import os
import numpy as np
import cv2

class DatasetWriter:
    def __init__(self, output_dir):
        """
        Constructor for Writer class.
        Input arguments:
        output_dir - path to output directory
        """
        self.output_dir = output_dir
        #create sub-directories if they dont exist
        for dir_name in ["bboxes", "center", "scale", "label", "frames", "masks"]:
            if not os.path.isdir(os.path.join(self.output_dir, dir_name)):
                os.makedirs(os.path.join(self.output_dir, dir_name))

    def write_to_disk(self, sample, index):
        """
        Function to write the generated sample (keypoint, center, scale, mask and the RGB image)
        in a format as expected by the ObjectKeypointTrainer training module
        (https://github.com/rohanpsingh/ObjectKeypointTrainer#preparing-the-dataset).
        Bounding-boxes are saved in the format as expected by darknet-yolov3
        (https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects).
        Input arguments:
        sample - labeled sample (RGB image, (keypoint, center pos, scale, mask))
        index  - counter for naming images
        """
        rgb_image = sample[0]
        kpt_label = sample[1][0]
        cen_label = sample[1][1]
        sca_label = sample[1][2]
        mask_label = sample[1][3]

        width = rgb_image.shape[1]
        height = rgb_image.shape[0]
        #write bounding box for yolo
        bboxfile = open(os.path.join(self.output_dir, 'bboxes', 'frame_' + repr(index).zfill(5) + '.txt'), 'w')
        bboxfile.write('0\t' + repr(cen_label[0]/width) + '\t' + repr(cen_label[1]/height) + '\t' +
                       repr(sca_label*200/width) + '\t' + repr(sca_label*200/height) + '\n')
        bboxfile.close()
        #write center to center/center_0####.txt
        centerfile = os.path.join(self.output_dir, 'center', 'center_' + repr(index).zfill(5) + '.txt')
        np.savetxt(centerfile, cen_label)
        #write scale to scale/scales_0####.txt
        scalesfile = os.path.join(self.output_dir, 'scale', 'scales_' + repr(index).zfill(5) + '.txt')
        np.savetxt(scalesfile, np.asarray([sca_label]))
        #write keypoints to label/label_0####.txt
        labelfile = os.path.join(self.output_dir, 'label', 'label_' + repr(index).zfill(5) + '.txt')
        np.savetxt(labelfile, kpt_label)
        #write RGB image to frames/frame_0####.txt
        cv2.imwrite(os.path.join(self.output_dir, 'frames', 'frame_' + repr(index).zfill(5) + '.jpg'), rgb_image)
        #write mask label to masks/mask_0####.txt
        cv2.imwrite(os.path.join(self.output_dir, 'masks', 'mask_' + repr(index).zfill(5) + '.jpg'), mask_label)
        return

