# python code to load and find the properties of certain image (traffic signs)

# import necessary libraries
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
import csv
import os, random
import cv2
from DISTS_pytorch import DISTS
from swd import swd

# Function to normalize the Image
def normalize(img):
    # define custom transform function
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # transform the image to get the original mean and std
    img_tr = transform(img)

    # calculate mean and std
    mean, std = img_tr.mean([1,2]), img_tr.std([1,2])

    # Define custom normalization function
    transform_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # get normalized image
    return transform_norm(img)

# function to resize and convert images to tensor
def img_resize_to_np(img_path):
    # convert to rgb
    img = Image.open(img_path).convert('RGB')

    # resized Images
    img = img.resize((64, 64))

    # normalize
    img_normalized = normalize(img)

    # convert normalized image to numpy
    img_np = np.array(img_normalized)
    img_np = img_np.reshape(3, 64, 64)
    # return img_np.reshape(3, 256, 256)

    # Create a larger sample_case
    return np.expand_dims(img_np, axis = 0)

# threshold calculation Function
def calculate_threshold():
    root_dir = 'Datasets/GTSRB/Selected_Train'
    random.seed(10)

    test_rows = []
    with open('Datasets/GTSRB/Test.csv', 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            test_rows.append(row)

    # prepare the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DISTS().to(device)

    results = []

    # save the train file
    train_file = []

    # Walk through all the classes in the directories
    with open("results_full_100.txt", "w+") as f:
        for subdir, dirs, files in os.walk(root_dir):
            for dir in dirs:
                # Generate x
                x = np.empty([0, 3, 256, 256])

                dirpath = os.path.join(root_dir, dir)
                # Randomly select 3 images as the sample iamges for comparing
                filenames = random.sample(os.listdir(dirpath), 3)

                for fname in filenames:
                    x = np.append(x, img_resize_to_np(os.path.join(dirpath, fname)), axis = 0)
                    print(os.path.join(dirpath, fname))
                    train_file.append(os.path.join(dirpath, fname))

                # conver tht stop_sign training np array to tensor for further calculation
                x = torch.tensor(x, dtype=torch.float)
                ref = x.to(device)

                # filter the test signs
                test_signs = filter(lambda c:int(c[6]) != int(dir), test_rows)
                test_signs_ls = list(test_signs)
                test_signs_list = [row[7] for row in test_signs_ls]

                res = []
                count = 0
                for i in test_signs_list:
                    count += 1
                    if count > 100:
                        break
                    fname = 'Datasets/GTSRB/{}'.format(i)
                    y = img_resize_to_np(fname)
                    y = torch.tensor(y, dtype=torch.float)
                    dist = y.to(device)
                    score = torch.mean(model(ref, dist))
                    tup = (i, score.item())
                    res.append([dir, i, score.item()])
                    print([dir, i, score.item()])
                    f.write(str(dir) + " " + str(i) + " " + str(score.item()) + "\n")
                results.append(min(res, key=lambda x:x[2]))

    results.sort(key=lambda x:int(x[0]))
    with open("results.txt", "w+") as f:
        for result in results:
            f.write(str(result[0]) + " " + str(result[1]) + " " + str(result[2]) + "\n")

    with open("model_pictures.txt", "w+") as f:
        for fname in train_file:
            f.write(str(fname) + "\n")

def verify_result():
    results = []

    with open('results.txt') as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
        for line in lines:
            results.append(line.split(' '))

    # convert to tensor for further comparing
    for line in results:
        line[2] = torch.tensor(float(line[2]), dtype=torch.float)

    images = dict()
    with open('model_pictures.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            dir = line[30:32]
            if dir[1] == '/':
                dir = dir[0]
            if int(dir) not in images:
                images[int(dir)] = [line]
            else:
                images[int(dir)].append(line)

    test_rows = []
    with open('Datasets/GTSRB/Test.csv', 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            test_rows.append(row)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DISTS().to(device)

    accs = []
    suitable = []
    classified = []
    for i in range(100, 200):
        acc = []
        suit = []
        fname = 'Datasets/GTSRB/Test/{:05d}.png'.format(i)
        correct_class = test_rows[i][6]

        for key in sorted(images):
            x = np.empty([0, 3, 256, 256])

            # for loop through the each class of sample images
            for img_path in images[key]:
                x = np.append(x, img_resize_to_np(img_path), axis = 0)
            x = torch.tensor(x, dtype=torch.float)
            ref = x.to(device)
            y = img_resize_to_np(fname)
            y = torch.tensor(y, dtype=torch.float)
            dist = y.to(device)
            score = torch.mean(model(ref, dist))

            acc.append((i, key, score.item()))
            print(acc[key])
            if score < results[key][2]:
                print(results[key], score)
                suit.append(key)

        min_score = list(min(acc, key = lambda x:x[2]))
        min_score.append(correct_class)
        classified.append(min_score)
        print("This picture is classified as ", classified[i - 100])
        accs.append(acc)
        suitable.append(suit)

    with open('accuracy_result_out.txt', 'w+') as file:
        accuracy = 0
        included = 0
        for id, classify in enumerate(classified):
            file.write("The estimated class for image " + str(classify[0]) + " is " + str(classify[1]) + \
            ". The correct images class is " + str(classify[3]) + "\n")
            if int(classify[1]) == int(classify[3]):
                accuracy += 1
            file.write("The class below threshold includes " + str(suitable[id]) + "\n")
            if int(classify[3]) in suitable[id]:
                included += 1
        file.write("The accuracy for this sample is " + str(accuracy) + "/100.\n")
        file.write("Pictures included for this sample is " + str(included) + "/100.\n")
        for acc in accs:
            file.write(str(acc) + "\n")

if __name__ == '__main__':
    # verify_result()
    # test_rows = []
    # with open('Datasets/GTSRB/Test.csv', 'r') as file:
    #     csvreader = csv.reader(file)
    #     header = next(csvreader)
    #     for row in csvreader:
    #         test_rows.append(row)
    #
    # with open('accuracy_result_only.txt') as f:
    #     lines = f.readlines()
    #     lines = [line.rstrip() for line in lines]
    #     line = lines[0]
    #     count = 100
    #     count2 = 0
    #     for line in lines:
    #         line = line.replace("[", "")
    #         line = line.replace("]", "")
    #         line = line.replace("(", "")
    #         line = line.replace(")", "")
    #         results = line.split(',')
    #         correct_class = test_rows[int(results[0])][6]
    #         i = 0
    #         acc = []
    #         for r in results:
    #             if i % 3 == 2:
    #                 acc.append(r)
    #             i += 1
    #         np_acc = np.array(acc).astype(float)
    #         print("Mean for the " + str(count) + " image is " + str(np.mean(r, dtype=np.float64)) + " Correct image class is " +
    #         correct_class + " Image score is " + str(acc[int(correct_class)]) + "\n")
    #         if float(np.mean(r, dtype=np.float64)) > float(acc[int(correct_class)]):
    #             count2 += 1
    #         min = np.amin(np_acc)
    #         res = int(np.where(np_acc == min)[0])
    #         print("the lowest class is " + str(res) + " score is " + str(min))
    #         count += 1
    #     print(str(count2))

    # prepare the device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = DISTS().to(device)
    #
    # root_dir = 'Datasets/GTSRB/Selected_Train/0'
    # filepath2 = 'Datasets/GTSRB/Test/00195.png'
    # x = np.empty([0, 3, 64, 64])
    # i = 0
    # for subdir, dirs, files in os.walk(root_dir):
    #     for file in files:
    #         if i > 150:
    #             break
    #         print(os.path.join(root_dir, file), str(i))
    #         i += 1
    #         x = np.append(x, img_resize_to_np(os.path.join(root_dir, file)), axis = 0)
    # x = torch.tensor(x, dtype=torch.float)
    # ref = x.to(device)
    # y = img_resize_to_np(filepath2)
    # y = torch.tensor(y, dtype=torch.float)
    # dist = y.to(device)
    # arr = model(ref, dist)
    # score = torch.mean(arr)
    # print(score, torch.max(arr), torch.min(arr))

    filepath = 'Datasets/GTSRB/Test/00195.png'
    sample = cv2.imread(filepath)
    first = int(sample.shape[0] * 0.2)
    second = int(sample.shape[1] * 0.2)
    chg_img = sample[first:sample.shape[0] - first, second:sample.shape[1] - second]
    cv2.imshow('image', chg_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #
    # # Read the data from the csv file
    # rows = []
    # with open('Datasets/GTSRB/Train.csv', 'r') as file:
    #     csvreader = csv.reader(file)
    #     header = next(csvreader)
    #     for row in csvreader:
    #         rows.append(row)
    #
    # # Filter the rows to stop signs
    # stop_signs = filter(lambda c:int(c[6]) == 14, rows)
    #
    # # read test files
    # test_rows = []
    # with open('Datasets/GTSRB/Test.csv', 'r') as file:
    #     csvreader = csv.reader(file)
    #     header = next(csvreader)
    #     for row in csvreader:
    #         test_rows.append(row)
    #
    # # filter the test stop sign
    # test_rows = [i + [int(i[7][5:10])] for i in test_rows]
    # # print(test_rows[:10])
    # stop_test_signs = filter(lambda c:int(c[6]) == 14, test_rows)
    # stop_test_id = []
    # for i in stop_test_signs:
    #     stop_test_id.append(i[8])
    # # print(len(stop_test_id))
    #
    # # Get all the stop_signs from the training dataset
    # x = np.empty([0, 3, 256, 256])
    # count = 0
    # for stop_sign in stop_signs:
    #     # limit the images to 3
    #     count += 1
    #     if count > 3:
    #         break
    #     img_path = 'Datasets/GTSRB/{}'.format(stop_sign[7])
    #     x = np.append(x, img_resize_to_np(img_path), axis = 0)
    # x = np.append(x, img_resize_to_np('Datasets/GTSRB/Train/14/00014_00001_00019.png'), axis = 0)
    # x = np.append(x, img_resize_to_np('Datasets/GTSRB/Train/14/00014_00002_00019.png'), axis = 0)
    # x = np.append(x, img_resize_to_np('Datasets/GTSRB/Train/14/00014_00003_00019.png'), axis = 0)
    #
    # # conver tht stop_sign training np array to tensor for further calculation
    # x = torch.tensor(x, dtype=torch.float)
    #
    # # prepare the device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = DISTS().to(device)
    # ref = x.to(device)
    #
    # results = []
    #
    # # loop through all the images we need to calculate compare to the selected stop_signs
    # for i in stop_test_id:
    #     # print('Datasets/GTSRB/Test/00{:03d}.png'.format(i))
    #     y = img_resize_to_np('Datasets/GTSRB/Test/{:05d}.png'.format(i))
    #     y = torch.tensor(y, dtype=torch.float)
    #     dist = y.to(device)
    #     score = torch.mean(model(ref, dist))
    #     results.append((i, score.item()))
    #     print((i, score.item()))
    #
    # # loop through the first 300 for comparing results
    # for i in range(300):
    #     # print('Datasets/GTSRB/Test/00{:03d}.png'.format(i))
    #     if i in stop_test_id:
    #         continue
    #     y = img_resize_to_np('Datasets/GTSRB/Test/{:05d}.png'.format(i))
    #     y = torch.tensor(y, dtype=torch.float)
    #     dist = y.to(device)
    #     score = torch.mean(model(ref, dist))
    #     results.append((i, score.item()))
    #     print((i, score.item()))
    #
    # # Sort and write the results to file
    # results.sort(key = lambda x:x[1])
    # with open("results.txt", "w") as f:
    #     for result in results:
    #         if result[0] in stop_test_id:
    #             f.write(str(result[0]) + " " + str(result[1]) + " stop" + "\n")
    #         else:
    #             f.write(str(result[0]) + " " + str(result[1]) + "\n")
    #
    # stop_count = 0
    # count = 0
    # for result in results:
    #     count += 1
    #     if count > 300:
    #         break
    #     if result[0] in stop_test_id:
    #         stop_count += 1
    #
    # with open("results.txt", "a") as f:
    #     f.write("Stop signs include: {}/{}, Total include: {}/{}".format(stop_count, len(stop_test_id), 300, len(results)))
