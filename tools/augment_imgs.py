"""Augments images and saves them.
"""

from imgaug import augmenters as iaa
import os
import numpy as np
import cv2


# From which directory to read images that are to be augmented.
IN_DIR = r"sundries\data\Ben\cropped_all"

# Where to save augmented images
OUT_DIR = IN_DIR + "_augmented"


# Augmentations applied to images: affine transformations, dropout..
seq = iaa.Sequential([
    # iaa.Sometimes(0.25,
    #               iaa.CoarseDropout((0.05, 0.2), size_percent=(0.02, 0.1))
    # ),
    iaa.OneOf([
        iaa.Affine(
            rotate=(-15, 15),
            shear=(-8, 8)),
        iaa.PiecewiseAffine(scale=(0.01, 0.05))
        ])
    ], random_order=False)


for batch_idx in range(10):
    # 'images' should be either a 4D numpy array of shape (N, height, width, channels)
    # or a list of 3D numpy arrays, each having shape (height, width, channels).
    # Grayscale images must have shape (height, width, 1) each.
    # All images must have numpy's dtype uint8. Values are expected to be in
    # range 0-255.

    for dir_ in os.listdir(IN_DIR):
        print("Batch no: " + str(batch_idx) + ", dir: " + str(dir_))
        if os.path.isdir(os.path.join(IN_DIR, dir_)):
            files = os.listdir(os.path.join(IN_DIR, dir_))
            j = 0

            subdir = os.path.join(IN_DIR, dir_)
            # images = [cv2.imread(os.path.join(subdir, fn), 0) for fn in files]  # If images are right size
            images = [cv2.resize(cv2.imread(os.path.join(subdir, fn), 0), (128, 128)) for fn in files]
            images_aug = seq.augment_images(images)

            # Create directory if it does not exist
            out_subdir = OUT_DIR + r"\\" + str(dir_) + r"\\"
            if not os.path.exists(out_subdir):
                os.makedirs(out_subdir)

            for img_ in images_aug:

                # Binarize image, so that there is only black and white pixels
                img_bw = cv2.threshold(img_, 127, 255, cv2.THRESH_BINARY)[1]
                cv2.imwrite(out_subdir + str(batch_idx) + "_" + str(j).zfill(5) + ".png", img_bw)

                # cv2.imshow("test", img_bw)
                # cv2.waitKey()

                j += 1
