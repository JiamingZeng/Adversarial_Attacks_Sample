# python code to load and find the properties of certain image (traffic signs)

# import necessary libraries
from PIL import Image, ImageOps, ImageChops
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
import csv
import os, random
import cv2
from DISTS_pytorch import DISTS
from swd import swd

#prepare the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DISTS().to(device)

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
def img_resize_to_np(img):
    # convert to rgb
#    img = Image.open(img_path).convert('RGB')

    # resized Images
    img = img.resize((32, 32))

    # normalize
    img_normalized = normalize(img)

    # convert normalized image to numpy
    img_np = np.array(img_normalized)
    img_np = img_np.reshape(3, 32, 32)
    # return img_np.reshape(3, 256, 256)

    # Create a larger sample_case
    return np.expand_dims(img_np, axis = 0)

# threshold calculation Function
def calculate_threshold():
    root_dir = 'Datasets/GTSRB/Selected_Train3'
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

    num = 0
    # Walk through all the classes in the directories
    with open("results_full_100.txt", "w+") as f:
        for subdir, dirs, files in os.walk(root_dir):
            for dir in dirs:
                # Generate x
                x = np.empty([0, 3, 64, 64])

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

                cat = dir.split('_')[0]
                # filter the test signs
                test_signs = filter(lambda c:int(c[6]) != int(cat), test_rows)
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
                res.sort(key = lambda x:x[2])
#                results.append(min(res, key=lambda x:x[2]))
                results.append(res[10])
                print(num)
                num += 1

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
            dir = line[31:35]
            if dir[3] == '/':
                dir = dir[0:3]
            if dir not in images:
                images[dir] = [line]
            else:
                images[dir].append(line)

    test_rows = []
#    with open('Datasets/GTSRB/Test.csv', 'r') as file:
#        csvreader = csv.reader(file)
#        header = next(csvreader)
#        for row in csvreader:
#            test_rows.append(row)
    with open('Datasets/Images/test.csv', 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            test_rows.append(row)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DISTS().to(device)

    accs = []
    suitable = []
    classified = []
    for i in range(0, 20):
        acc = []
        suit = []
        fname = 'Datasets/GTSRB/Test/{:05d}.png'.format(i)
        correct_class = test_rows[i][1]

        j = 0
        print(i)
        for key in sorted(images):
            x = np.empty([0, 3, 32, 32])

            # for loop through the each class of sample images
            # key = 0_0
            for img_path in images[key]:
                x = np.append(x, img_resize_to_np(img_path), axis = 0)
            x = torch.tensor(x, dtype=torch.float)
            ref = x.to(device)
            y = img_resize_to_np(fname)
            y = torch.tensor(y, dtype=torch.float)
            dist = y.to(device)
            score = torch.mean(model(ref, dist))

            acc.append((i, key, score.item()))
#            print(acc[j])
            # since the results and the images have been sorted the sequence should be the same
            for result in results:
                if result[0] == key:
                    to_compare = result
            if score < to_compare[2]:
                print(to_compare, score)
                suit.append(key)
            j += 1

        acc.sort(key = lambda x:x[2])
        # min_score = list(min(acc, key = lambda x:x[2]))
        # include the lowest fifteen scores
        min_score = acc[0:30]

        min_score.append(correct_class)
        classified.append(min_score)
        print("This picture is classified as ", classified[i][0:3])
        print("correct", classified[i][30])
        accs.append(acc)
        suitable.append(suit)

    with open('accuracy_result_out.txt', 'w+') as file:
        accuracy = 0
        included = 0
        low_in = 0
        count = 0
        for id, classes in enumerate(classified):
            main_class = []
            for i in range(30):
                actual = classes[i][1][0:2]
                if actual[1] == '_':
                    actual = actual[0:1]
                if actual not in main_class:
                    count += 1
                    main_class.append(actual)
                if count > 10:
                    break
            print(main_class)
            if classes[30] in main_class:
                low_in += 1
            low = []
            low.append(classes[0][1])
            low.append(classes[1][1])
            low.append(classes[2][1])
            classify = list(classes[0])
            classify[1] = classify[1][0:2]
            if classify[1][1] == '_':
                classify[1] = classify[1][0:1]
                
            # get the correct class
            
            file.write("The estimated class for image " + str(classify[0]) + " is " + str(classify[1]) + \
            ". The correct images class is " + str(classes[30]) + "\n")
            if int(classify[1]) == int(classes[30]):
                accuracy += 1
            file.write("The class below threshold includes " + str(suitable[id]) + "\n")
            for key in suitable[id]:
                print(key)
                key = key[0:2]
                
#                suitable[id] = suitable[id][0:2]
                if key[1] == '_':
                    to_compare = key[0]
                else:
                    to_compare = key
                if to_compare == classes[30]:
                    included += 1
                    break
#            if int(classify[1]) in suitable[id]:
#                included += 1
            file.write("Lowest Three is " + str(classes[0][1]) + " " + str(classes[1][1]) + \
            " " + str(classes[2][1]) + "\n")
            for key in main_class:
                file.write(key + ", ");
            file.write("\n");
#            for i in range(15):
#                if classes[30] == classes[i][1][0:len(classes[30])]:
#                    low_in += 1
#                    break
        file.write("The accuracy for this sample is " + str(accuracy) + "/100.\n")
        file.write("Pictures included for this sample is " + str(included) + "/100.\n")
        file.write("Pictures in the lowest three is  " + str(low_in) + "/100.\n")
        for acc in accs:
            file.write(str(acc) + "\n")

def find_min():
    """Function for finding lowest class and compare"""
    test_rows = []
    with open('Datasets/GTSRB/Test.csv', 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            test_rows.append(row)

    with open('accuracy_result_only.txt') as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
        line = lines[0]
        count = 0
        for line in lines:
            count2 = 0
            line = line.strip()
            line = line.replace("[", "")
            line = line.replace("\'", "")
            line = line.replace("]", "")
            line = line.replace("(", "")
            line = line.replace(")", "")
            results = line.split(',')
            correct_class = test_rows[int(results[0])][6]
            i = 0
            acc = []
            if count2 > 0:
                break
            count2 += 1
            # store the category and score for comparing
            small = []
            for r in results:
                if i % 3 == 1:
                    r = r.strip()
                    r = r[0:2]
                    if r[1] == '_':
                        r = r[0]
                    small.append(r)
                if i % 3 == 2:
                    small.append(float(r))
                    acc.append(small)
                    small = []
                i += 1
#            np_acc = np.array(acc).astype(float)
            acc.sort(key = lambda x:x[1])
            
            # count if the lowest 10 classes include the correct class
            category = []
            count2 = 0
            for a in acc:
                if count2 > 10:
                    break
                if a[0] not in category:
                    category.append(a[0])
                    count2 += 1
            if correct_class in category:
                count += 1
        print(count)
#            print("Mean for the " + str(count) + " image is " + str(np.mean(r, dtype=np.float64)) + " Correct image class is " +
#            correct_class + " Image score is " + str(acc[int(correct_class)]) + "\n")
#            if float(np.mean(r, dtype=np.float64)) > float(acc[int(correct_class)]):
#                count2 += 1
#            min = np.amin(np_acc)
#            res = int(np.where(np_acc == min)[0])
#            print("the lowest class is " + str(res) + " score is " + str(min))
#            count += 1
#        print(str(count2))
#
#    #prepare the device
#    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#    model = DISTS().to(device)
#
#    root_dir = 'Datasets/GTSRB/Selected_Train/0'
#    filepath2 = 'Datasets/GTSRB/Test/00195.png'
#    x = np.empty([0, 3, 64, 64])
#    i = 0
#    for subdir, dirs, files in os.walk(root_dir):
#        for file in files:
#            if i > 150:
#                break
#            print(os.path.join(root_dir, file), str(i))
#            i += 1
#            x = np.append(x, img_resize_to_np(os.path.join(root_dir, file)), axis = 0)
#    x = torch.tensor(x, dtype=torch.float)
#    ref = x.to(device)
#    y = img_resize_to_np(filepath2)
#    y = torch.tensor(y, dtype=torch.float)
#    dist = y.to(device)
#    arr = model(ref, dist)
#    score = torch.mean(arr)
#    print(score, torch.max(arr), torch.min(arr))

def calculate_inclass_comparison():
    root_dir = 'Datasets/GTSRB/Train'
    for subdir, dirs, files in os.walk(root_dir):
        for dir in dirs:
            names = []
            x = np.empty([0, 3, 64, 64])
            dirpath = os.path.join(root_dir, dir)
            # Randomly select 50 images as the sample iamges for comparing
            filenames = random.sample(os.listdir(dirpath), 50)
            for fname in filenames:
                x = np.append(x, img_resize_to_np(os.path.join(dirpath, fname)), axis = 0)

            tensor_x = torch.tensor(x, dtype=torch.float)
            ref = tensor_x.to(device)
            """Loop through the x and get the results by comparing each image to all of the others in the same class"""
            fname = "Datasets/GTSRB/inclass_comparison/inclass_comparison_{}.txt".format(dir)
            i = 0
            with open(fname, 'w+') as file:
                for finame in os.listdir(dirpath):
                    y = img_resize_to_np(os.path.join(dirpath, finame))
                    tensor_y = torch.tensor(y, dtype=torch.float)
                    dist = tensor_y.to(device)
                    arr = model(ref, dist)
                    print("{} ".format(os.path.join(dirpath, finame)), i)
                    file.write("{} ".format(os.path.join(dirpath, finame)))
                    for element in arr:
                        file.write("{} ".format(element))
                    file.write("\n")
                    i += 1
            # score = torch.mean(arr)
            # print("{} {}\n".format(score.item(), names[i]))
            # file.write("{} {}\n".format(score.item(), names[i]))
def shift_image():
    model = DISTS()
    img_path = 'Datasets/GTSRB/Train/0/00000_00000_00000.png'
    img_path2 = 'Datasets/GTSRB/Train/0/00000_00000_00029.png'
    x = Image.open(img_path).convert('RGB')
    y = Image.open(img_path2).convert('RGB')
#    y = ImageChops.offset(y, 10, 0)
    x = torch.tensor(img_resize_to_np(x), dtype=torch.float)
    y = torch.tensor(img_resize_to_np(y), dtype=torch.float)
    print(model(x, y))
    
if __name__ == '__main__':
#    calculate_threshold()
#    verify_result()
#    find_min()
    shift_image()


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
