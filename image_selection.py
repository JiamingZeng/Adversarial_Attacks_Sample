# python code to select image with high average intensity

import os
import cv2
from shutil import copyfile
import numpy as np
import time
import torch
import csv
from matplotlib import pyplot as plt
from kmeans_pytorch import kmeans


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
    # select_image()

    # for subdir, dirs, files in os.walk(root_dir):
    #     for dir in dirs:
    #         dirpath = os.path.join(root_dir, dir)
    # Try to work on the first directory and seperate in different directories
    for i in range(43):

        scores = []
        files = []
        filename = 'Datasets/GTSRB/inclass_comparison/inclass_comparison_{}.txt'.format(i)
        with open(filename, 'r') as file:
            csvreader = csv.reader(file)
            header = next(csvreader)
            for row in csvreader:
                score = []
                after_split = str(row[0]).split(' ')
                files.append(after_split[0])
                for a in after_split:
                    if a != after_split[0] and a != '':
                        score.append(float(a))
                scores.append(score)
        scores_tensor = torch.FloatTensor(scores)

        flag = True
        while flag:
            flag = False
            num_clusters = 5
            cluster_ids_x, cluster_centers = kmeans(
                X=scores_tensor, num_clusters=num_clusters, distance='euclidean'
            )
            cluster_ids_x = cluster_ids_x.numpy()
            for i in range(num_clusters):
                s = list(cluster_ids_x).count(i)
                if s < 4:
                    flag = True

        out_dir = 'Datasets/GTSRB/Selected_Train3/'.format(i)
        # create the directory if not exist
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        for i in range(len(cluster_ids_x)):
            out_sub_dir = str(cluster_ids_x[i])
            out_dir_spec = os.path.join(out_dir, out_sub_dir)

            # create the directory if not exist
            if not os.path.exists(out_dir_spec):
                os.makedirs(out_dir_spec)

            input = files[i]
            output_file_name = os.path.join(out_dir_spec, os.path.basename(input))
            copyfile(input, output_file_name)
