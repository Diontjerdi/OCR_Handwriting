import numpy as np
import os
import csv
import cv2
import ast

from tools.imgTools import process_char, segment_chars, crop_border


def read_data(filename, process=False):
    """
    Read in data from a list with paths to images and corresponding labels
    :param filename: (str) path to file from which to read the data.
    :param process: (bool) Whether to process the characters.
    :return: (NumPy array) List of images (NumPy arrays) and labels.
    """
    print("...Reading data...")
    images = []

    # Read .csv file and add image paths and labels to list.
    with open(filename, 'rt') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)

        list_of_images, labels = [], []
        for row in reader:
            list_of_images.append(row[0])
            labels.append(int(row[1]))

    # Read in and optionally process images, add to list
    if process:
        for img in list_of_images:
            img = cv2.imread(img, 0)
            img = process_char(img, skeleton=False, deslant=True)

            # cv2.imshow("", img)
            # cv2.waitKey()

            images.append(img)

    else:   # If characters are already preprocessed
        for img in list_of_images:
            img = cv2.imread(img, 0)
            if not img.shape == (32, 32):
                img = cv2.resize(img, (32, 32))
            images.append(img)

    images = np.asarray(images, dtype=np.float32)
    images = images / 255

    labels = np.asarray(labels)

    return images, labels


def read_data_test(in_dir):
    """
    Same as read_data(), but modified for testing performance on handwritten letter.
    :param in_dir (str) path to directory that contains the character images.
    """
    print("...Reading data...")
    images = []
    files = os.listdir(in_dir)

    for file in files:
        if file.endswith(".png"):
            img = cv2.imread(os.path.join(in_dir, file), 0)
            img = process_char(img, skeleton=False, deslant=True)

            # cv2.imshow("test", img)
            # cv2.waitKey()

            images.append(img)

    images = np.asarray(images, dtype=np.float32)
    images = images / 255

    labels = [43, 14, 21, 21, 24, 39, 18, 24, 23, 54, 51, 43, 44, 49, 59, 50, 41, 48, 60, 37, 47, 36, 38, 46, 52, 56,
              36, 53, 55, 61, 45, 56, 39, 42, 40, 48, 60, 57, 50, 58, 25, 10, 12, 20, 22, 34, 11, 24, 33, 32, 18, 29,
              17, 15, 18, 31, 14, 13, 24, 35, 14, 23, 21, 18, 26, 30, 24, 27, 19, 30, 16, 28, 12, 32, 22, 15, 19, 24,
              27, 13, 11, 10, 23, 20, 16, 21, 34, 25, 17, 28, 31, 14, 33, 29, 26, 30, 18, 35, 9, 8, 7, 6, 5, 4, 3, 2,
              1, 0, 3, 5, 7, 9, 0, 2, 1, 4, 8, 6]

    eval_lens = [5, 4, 6, 2, 2, 5, 6, 5, 2, 3, 4, 2, 3, 4, 4, 5, 6, 4, 3, 5, 4, 6, 4, 4, 10, 10]  # Word lengths

    return images, np.array(labels), eval_lens


# Read in data from a list with paths and multilabels
def read_data_multi(filename, process=False):
    """
    Same as read_data(), but modified for testing performance on images of words.
    """
    print("...Reading data...")
    images, labels = [], []

    word_lens = []

    with open(filename, 'rt') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)

        for row in reader:
            lbl_list = ast.literal_eval(row[1])  # Convert string representation of list of labels to list

            img = cv2.imread(os.path.join(row[0]), 0)

            cuts = segment_chars(img)

            word_lens.append(len(lbl_list))

            for cut in cuts:
                if process:
                    cut_im = process_char(cut, skeleton=False, deslant=True)
                    images.append(cut_im)
                else:
                    # TODO: Check what is effective
                    cut_im = crop_border(cut)
                    cut_im = cv2.resize(cut_im, (32, 32))
                    images.append(cut_im)

                # # Show cuts
                # cv2.imshow("cut",cv2.resize(crop_border(cut_im), (32,32)))
                # cv2.waitKey()

            labels.extend(lbl_list)

    images = np.asarray(images, dtype=np.float32)
    images = images / 255

    labels = np.asarray(labels)

    # return images, labels
    return images, labels, word_lens
