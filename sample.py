import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
import matplotlib.pyplot as plt

# Set up all the paramters for the model
# Total num of rounds of training
n_epochs = 3
# Num of reads in one training
batch_size_train = 64
batch_size_test = 1000
# How fast the gradient change with the back-propogation
learning_rate = 0.01
# It changes the weight increment with momentum times last time's weight increment
# improve the speed and accuracy of the model
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)
p = Path('/Users/jiamingzeng/Documents/UM/EECS/research/')


# Set up the transformation
transform = torchvision.transforms.Compose([
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize(
    (0.1307,), (0.3081,))
])

# Load the training and testing data
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST(p / "files", train=True, download=True,
                             transform=transform), batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST(p / "files", train=False, download=True,
                             transform=transform), batch_size=batch_size_test, shuffle=True)

# Define what device we are using
# print("CUDA Available: ",torch.cuda.is_available())
# device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# Build the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Use the same as MNIST Challenge
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        # self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=5)
        # Define to dropout zeros in convo2d
        self.conv2_drop = nn.Dropout2d()
        # Linear includes bias
        self.fc1 = nn.Linear(4 * 4 * 64, 1024)
        # self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        # 2*2 max pooling
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # print(x.shape)
        # Reshape the model for fitting into linear model
        x = x.view(-1, 4 * 4 * 64)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        # Normalize the logistic function of 10 dimension to probability distribution over predicted class
        return F.log_softmax(x)

network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)

# Training starts
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    # Need to set gradient to zero because pytorch accumulate gradient
    optimizer.zero_grad()

    # Compute the output of data
    output = network(data)

    # the loss is calculated using negative log-likelihood loss
    loss = F.nll_loss(output, target)

    # Feed back into the model to improve the weights and bias
    loss.backward()
    optimizer.step()

    # Every ten times record
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      torch.save(network.state_dict(), p /"results"/"model.pth")
      torch.save(optimizer.state_dict(), p /"results"/"optimizer.pth")

def test():
  network.eval()
  test_loss = 0
  correct = 0
  for data, target in test_loader:
     output = network(data)

     # Calculate the negative log-likelihood loss
     test_loss += F.nll_loss(output, target, size_average=False).item()

     # Predict and record the correct number
     # Fetch the max number of probability
     pred = output.data.max(1, keepdim=True)[1]
     correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

test()
for epoch in range(1, n_epochs + 1):
  train(epoch)
  test()

# fig = plt.figure()
# plt.plot(train_counter, train_losses, color='blue')
# plt.scatter(test_counter, test_losses, color='red')
# plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
# plt.xlabel('number of training examples seen')
# plt.ylabel('negative log likelihood loss')
# plt.show()

# Start Adversarial Attack using FGSM
# Follow the sign of gradient and thus maximize the Loss

def fgsm_attack(model, data, target, eps):
    # data = data.to(device)
    # target = target.to(device)
    data.requires_grad = True

    output = model(data)

    cost = F.nll_loss(output, target)
    model.zero_grad()
    cost.backward()
    data_grad = data.grad.data

    attack_images = data + eps*data_grad.sign()
    attack_images = torch.clamp(attack_images, 0, 1)

    return attack_images

# Execute the function and print out the results
# examples = enumerate(test_loader)
# batch_idx, (example_data, example_targets) = next(examples)
def attack_test(eps):
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        # data, target = data.to(device), target.to(device)
        attacked_data = fgsm_attack(network, data, target, eps)
        output = network(attacked_data)
        test_loss += F.nll_loss(output, target, size_average=False).item()

        # Predict and record the correct number
        # Fetch the max number of probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    # print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #         test_loss, correct, len(test_loader.dataset),
    #         100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)

# attacked_data = attacked_data.detach().numpy()
# plt.imshow(attacked_data[0][0], cmap='gray', interpolation='none')
# plt.show()

# Testing the program with different epsilons
epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
accuracies = []
for eps in epsilons:
    acc= attack_test(eps)
    accuracies.append(acc)
plt.plot(epsilons, accuracies, "*-")
plt.show()
