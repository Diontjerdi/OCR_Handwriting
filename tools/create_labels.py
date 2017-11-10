""" Walks past all files in a directory (assuming that files of a class are in a directory with the classname)
and writes the path and label of each file to a text file.
"""
import os
import csv
import numpy as np


def data_to_text(in_dir):
    # with open('data\\labels.txt', 'w', newline='') as csvfile:
    with open(os.path.join(in_dir, 'labels.txt'), 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='\"', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(["Image", "Label"])
        for root, dirs, files in os.walk(in_dir):
            for file in files:
                if file.endswith('.png'):

                    # regular encoding
                    label = os.path.basename(os.path.normpath(root))
                    # csvwriter.writerow([os.path.join(root, file), label])  # Absolute path
                    csvwriter.writerow([os.path.join(in_dir + "\\" + str(label) + '\\', file), label])  # Relative path


def data_to_text_split(in_dir):
    for subdir in ["train", "test"]:
        with open(os.path.join(in_dir, subdir + '.txt'), 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quotechar='\"', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(["Image", "Label"])
            for root, dirs, files in os.walk(os.path.join(in_dir, subdir)):
                for file in files:
                    if file.endswith('.png'):

                        # regular encoding
                        label = os.path.basename(os.path.normpath(root))
                        # csvwriter.writerow([os.path.join(root, file), label])  # Absolute path
                        csvwriter.writerow([os.path.join(in_dir + "\\" + subdir +
                                                         "\\" + str(label) + '\\', file), label])  # Relative path


# If data is in a single directory
# data_to_text(r"data\Cheat3_crop_skel")

# If data is split into train and test directories
# data_to_text_split(r"data")
data_to_text_split(r"sundries\data\HASY_Ben_Char74K\mix_deslant_split0.1")

