# python code to load and find the properties of certain image (traffic signs)

# import necessary libraries
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
from swd import swd


# load the image
# img_path = 'standard_stop.png'
img_path = 'archive/Meta/14.png'
img = Image.open(img_path).convert('RGB')
img_test = Image.open('Archive/train/14/00014_00000_00000.png').convert('RGB')

# convert PIL image to numpy array
img = img.resize((256, 256))

# define custom transform function
transform = transforms.Compose([
    transforms.ToTensor()
])

img_tr = transform(img)

# calculate mean and std
mean, std = img_tr.mean([1,2]), img_tr.std([1,2])

# Define custom normalization function
transform_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# get normalized image
img_normalized = transform_norm(img)

# convert normalized image to numpy
# array
img_np = np.array(img_normalized)

img_np = img_np.reshape(3, 256, 256)

# Resize the image
img_test = img_test.resize((256, 256))
img_test_np = np.array(img_test)
img_test_np = img_test_np.reshape(3, 256, 256)

# Create a larger sample_case
img_large = np.expand_dims(img_np, axis = 0)
# for i in range(52, 65):
#     img2 = Image.open('traffic_sign_project/archive/images/road{}.png'.format(i)).convert('RGB')
#     img2 = img2.resize((256, 256))
#     img2_np = np.array(img2)
#     img2_np = img2_np.reshape(3, 256, 256)
#     np.append(img_large, img2_np)
img_test_np = np.expand_dims(img_test_np, axis = 0)
x = torch.tensor(img_large, dtype=torch.float)
y = torch.tensor(img_test_np, dtype=torch.float)

out = swd(x, y)
# print(x.size(), y.size())
print("distance: {:.3f}".format(out))

# Record all the distances
results = []

for j in range(3):
    for i in range(29):
        img_test = Image.open('archive/Train/14/00014_000{:02d}_000{:02d}.png'.format(j, i)).convert('RGB')

        # Resize the image
        img_test = img_test.resize((256, 256))

        img_tr = transform(img_test)

        # calculate mean and std
        mean, std = img_tr.mean([1,2]), img_tr.std([1,2])

        # Normalize the image
        img_test_normalized = transform_norm(img_test)
        img_test_np = np.array(img_test_normalized)
        img_test_np = img_test_np.reshape(3, 256, 256)

        img_test_np = np.expand_dims(img_test_np, axis = 0)
        y = torch.tensor(img_test_np, dtype=torch.float)
        results.append((i, swd(x, y)))

results.sort(key = lambda x:x[1])
print(results)
res_mean = np.percentile(results, 25)
new_result = []
num = 0

for id, result in enumerate(results):
    f.write(id)
    f.write(result)
    if result < res_mean:
        if id > 51 and id < 100:
            num += 1
        new_result.append(id)

# with open("results.txt", "w") as f:
#     f.write("Images remained: {} / 876, Stop sign remained: {} / 48".format(len(new_result), num))
