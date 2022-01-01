# python code to load and find the properties of certain image (traffic signs)

# import necessary libraries
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
from swd import swd


# load the image
img_path = 'standard_stop.png'
img = Image.open(img_path).convert('RGB')
img_test = Image.open('traffic_sign_project/archive/images/road0.png').convert('RGB')

# convert PIL image to numpy array
img = img.resize((256, 256))
img_np = np.array(img)
img_np = img_np.reshape(3, 256, 256)

# Resize the image
img_test = img_test.resize((256, 256))
img_test_np = np.array(img_test)
img_test_np = img_test_np.reshape(3, 256, 256)

img_large = np.expand_dims(img_np, axis = 0)
img_test_np = np.expand_dims(img_test_np, axis = 0)
x = torch.tensor(img_large, dtype=torch.float)
y = torch.tensor(img_test_np, dtype=torch.float)

out = swd(x, y)
# print(x.size(), y.size())
# print("distance: {:.3f}".format(out))

# Record all the distances
results = []

for i in range(876):
    img_test = Image.open('traffic_sign_project/archive/images/road{}.png'.format(i)).convert('RGB')

    # Resize the image
    img_test = img_test.resize((256, 256))
    img_test_np = np.array(img_test)
    img_test_np = img_test_np.reshape(3, 256, 256)

    # Resize the image
    img_test = img_test.resize((256, 256))
    img_test_np = np.array(img_test)
    img_test_np = img_test_np.reshape(3, 256, 256)

    img_test_np = np.expand_dims(img_test_np, axis = 0)
    y = torch.tensor(img_test_np, dtype=torch.float)
    results.append(swd(x, y))

results = np.array(results)
res_mean = np.percentile(results, 75)
new_result = []

for id, result in enumerate(results):
    if result < res_mean:
        new_result.append(id)

print(new_result)
