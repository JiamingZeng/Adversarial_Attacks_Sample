# python code to load and find the properties of certain image (traffic signs)

# import necessary libraries
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
import csv
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
    img = img.resize((256, 256))

    # normalize
    img_normalized = normalize(img)

    # convert normalized image to numpy
    img_np = np.array(img_normalized)
    img_np = img_np.reshape(3, 256, 256)
    # return img_np.reshape(3, 256, 256)

    # Create a larger sample_case
    return np.expand_dims(img_np, axis = 0)


if __name__ == '__main__':

    # Read the data from the csv file
    rows = []
    with open('Datasets/GTSRB/Train.csv', 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            rows.append(row)

    # Filter the rows to stop signs
    stop_signs = filter(lambda c:int(c[6]) == 14, rows)

    # read test files
    test_rows = []
    with open('Datasets/GTSRB/Test.csv', 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            test_rows.append(row)

    # filter the test stop sign
    test_rows = [i + [int(i[7][5:10])] for i in test_rows]
    # print(test_rows[:10])
    stop_test_signs = filter(lambda c:int(c[6]) == 14, test_rows)
    stop_test_id = []
    for i in stop_test_signs:
        stop_test_id.append(i[8])
    # print(len(stop_test_id))

    # Get all the stop_signs from the training dataset
    x = np.empty([0, 3, 256, 256])
    count = 0
    for stop_sign in stop_signs:
        # limit the images to 3
        count += 1
        if count > 3:
            break
        img_path = 'Datasets/GTSRB/{}'.format(stop_sign[7])
        x = np.append(x, img_resize_to_np(img_path), axis = 0)
    x = np.append(x, img_resize_to_np('Datasets/GTSRB/Train/14/00014_00001_00019.png'), axis = 0)
    x = np.append(x, img_resize_to_np('Datasets/GTSRB/Train/14/00014_00002_00019.png'), axis = 0)
    x = np.append(x, img_resize_to_np('Datasets/GTSRB/Train/14/00014_00003_00019.png'), axis = 0)

    # conver tht stop_sign training np array to tensor for further calculation
    x = torch.tensor(x, dtype=torch.float)

    # prepare the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DISTS().to(device)
    ref = x.to(device)

    results = []

    # loop through all the images we need to calculate compare to the selected stop_signs
    for i in stop_test_id:
        # print('Datasets/GTSRB/Test/00{:03d}.png'.format(i))
        y = img_resize_to_np('Datasets/GTSRB/Test/{:05d}.png'.format(i))
        y = torch.tensor(y, dtype=torch.float)
        dist = y.to(device)
        score = torch.mean(model(ref, dist))
        results.append((i, score.item()))
        print((i, score.item()))

    # loop through the first 300 for comparing results
    for i in range(300):
        # print('Datasets/GTSRB/Test/00{:03d}.png'.format(i))
        if i in stop_test_id:
            continue
        y = img_resize_to_np('Datasets/GTSRB/Test/{:05d}.png'.format(i))
        y = torch.tensor(y, dtype=torch.float)
        dist = y.to(device)
        score = torch.mean(model(ref, dist))
        results.append((i, score.item()))
        print((i, score.item()))

    # Sort and write the results to file
    results.sort(key = lambda x:x[1])
    with open("results.txt", "w") as f:
        for result in results:
            if result[0] in stop_test_id:
                f.write(str(result[0]) + " " + str(result[1]) + " stop" + "\n")
            else:
                f.write(str(result[0]) + " " + str(result[1]) + "\n")

    stop_count = 0
    count = 0
    for result in results:
        count += 1
        if count > 300:
            break
        if result[0] in stop_test_id:
            stop_count += 1

    with open("results.txt", "a") as f:
        f.write("Stop signs include: {}/{}, Total include: {}/{}".format(stop_count, len(stop_test_id), 300, len(results)))



    # # stop_signs = Image.open('Datasets/GTSRB/Train/20/00020_00000_00000.png').convert('RGB')
    # # convert it into tensor
    # # torch.tensor(img_large, dtype=torch.float)
    # x = img_resize_to_np('Datasets/GTSRB/Train/20/00020_00000_00000.png')
    # x = np.append(x, img_resize_to_np('Datasets/GTSRB/Train/20/00020_00000_00001.png'), axis = 0)
    # x = np.append(x, img_resize_to_np('Datasets/GTSRB/Train/20/00020_00000_00000.png'), axis = 0)
    # x = torch.tensor(x, dtype=torch.float)
    #
    # print(x.shape)
    # # other = Image.open('Datasets/GTSRB/Train/20/00020_00000_00000.png').convert('RGB')
    # y = img_resize_to_np('Datasets/GTSRB/Train/20/00020_00000_00001.png')
    # # y = np.append(y, img_resize_to_np('Datasets/GTSRB/Train/20/00020_00000_00000.png'), axis = 0)
    # y = torch.tensor(y, dtype=torch.float)
    # print(D(x, y))
