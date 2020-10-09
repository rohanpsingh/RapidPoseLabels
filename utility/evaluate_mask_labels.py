import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt


def computeIntersectionOverUnion(img1, img2, visualize=False):

    images = [img1, img2]
    for img in images:
        assert(img.dtype == np.uint8)
        assert(img.ndim == 2)
    union = np.logical_or(*images)
    intersection = np.logical_and(*images)
    IoU = np.sum(intersection) / float(np.sum(union))

    if visualize:
        fig = plt.figure()
        subplot = fig.add_subplot(2,3,1)
        plt.imshow(images[0])
        subplot.set_title('image 1')

        subplot = fig.add_subplot(2,3,2)
        plt.imshow(images[1])
        subplot.set_title('image 2')

        subplot = fig.add_subplot(2,3,4)
        plt.imshow(intersection)
        subplot.set_title('intersection')

        subplot = fig.add_subplot(2,3,5)
        plt.imshow(union)
        subplot.set_title('union')

        subplot = fig.add_subplot(2,3,6)
        plt.imshow(intersection.astype(int) + union.astype(int))
        subplot.set_title('IoU')

        plt.suptitle('computed IoU = %.5f' % (IoU), fontsize=24)
        plt.show()

    return IoU


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print('usage: ', sys.argv[0], ' <dir1> <dir2>')
        sys.exit(1)

    images1 = [cv2.imread(os.path.join(sys.argv[1], fn), cv2.IMREAD_GRAYSCALE) for fn in os.listdir(sys.argv[1])]
    images2 = [cv2.imread(os.path.join(sys.argv[2], fn), cv2.IMREAD_GRAYSCALE) for fn in os.listdir(sys.argv[2])]

    list_iou = []
    for img1, img2 in zip(images1, images2):
        iou = computeIntersectionOverUnion(img1, img2, False)
        list_iou.append(iou)
    print("mean IoU = ", sum(list_iou)/len(list_iou))
