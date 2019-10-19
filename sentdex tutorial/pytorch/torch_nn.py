import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F

train = datasets.MNIST('', train=True, download=True, transform=transforms.Compose([transforms.ToTensor]))
test = datasets.MNIST('', train=False, download=True, transform=transforms.Compose([transforms.ToTensor]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super().__init__() #super corespond to nn.Modules, running the initialization for nn.Modules as well as whatever else we happen to in it. inherit the methods and attributes
        self.fc1 = nn.Linear(28*28, 64) #28*28=784 (input image flattened), 64 (neuron as an output). if using convolutional use nn.conv
        self.fc2 = nn.Linear(64, 64) 
        self.fc3 = nn.Linear(64, 64) 
        self.fc4 = nn.Linear(64, 10) #the output is 10 because we have 10 classes (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

    def forward(self, x): #feed-forward function, you could do logic in this forward function
        x = F.relu(self.fc1(x)) #F.relu is an activation function which run on output
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim = 1) #dim = dimension on which we want to apply softmax, dim = 1 is which thing is the probability distribution that we want to sum to one

net = Net()
print(net)

X = torch.rand((28, 28))
X = X.view(-1, 28*28) #-1 specifies that this input will be of an unknown shape
output = net(X) #the actual prediction
print(output)
