# python code to select image with high average intensity

import os
import cv2
from shutil import copyfile
import numpy as np

def select_image():
    root_dir = 'Datasets/GTSRB/Train'
    out_dir = 'Datasets/GTSRB/Selected_Train2'

    # # Walk through all the classes in the directories
    # for subdir, dirs, files in os.walk(root_dir):
    #     for dir in dirs:
    #         dirpath = os.path.join(root_dir, dir)
    #         outpath = os.path.join(out_dir, dir)
    #
    #         # create the directory if not exist
    #         if not os.path.exists(outpath):
    #             os.makedirs(outpath)
    #
    #         # array to store average intensity
    #         inten = []
    #         fname_array = []
    #         out_array = []
    #
    #         for fname in os.listdir(dirpath):
    #             img = cv2.imread(os.path.join(dirpath, fname))
    #             inten.append(np.mean(img))
    #             fname_array.append(os.path.join(dirpath, fname))
    #             out_array.append(os.path.join(outpath, fname))
    #
    #         # calculate the median
    #         inten_percentile = np.percentile(inten, 75)
    #
    #         # copy the file to the new directory
    #         for id, fname in enumerate(fname_array):
    #             if inten[id] > inten_percentile:
    #                 copyfile(fname_array[id], out_array[id])

    all = []
    # Walk through all the classes in the directories
    for subdir, dirs, files in os.walk(root_dir):
        for dir in dirs:
            dirpath = os.path.join(root_dir, dir)

            for fname in os.listdir(dirpath):
                img = cv2.imread(os.path.join(dirpath, fname))
                all.append(np.mean(img))

    inten_percentile = np.percentile(all, 60)

    # Walk through all the classes in the directories
    for subdir, dirs, files in os.walk(root_dir):
        for dir in dirs:
            dirpath = os.path.join(root_dir, dir)
            outpath = os.path.join(out_dir, dir)

            # create the directory if not exist
            if not os.path.exists(outpath):
                os.makedirs(outpath)

            # array to store average intensity
            inten = []
            fname_array = []
            out_array = []

            for fname in os.listdir(dirpath):
                img = cv2.imread(os.path.join(dirpath, fname))
                inten.append(np.mean(img))
                fname_array.append(os.path.join(dirpath, fname))
                out_array.append(os.path.join(outpath, fname))

            # copy the file to the new directory
            for id, fname in enumerate(fname_array):
                if inten[id] > inten_percentile:
                    copyfile(fname_array[id], out_array[id])

if __name__ == '__main__':
    # img = cv2.imread('Datasets/GTSRB/Train/25/00025_00018_00019.png')
    # img2 = cv2.imread('Datasets/GTSRB/Train/25/00025_00019_00000.png')
    # print(np.mean(img))
    # print(np.mean(img2))
    select_image()
