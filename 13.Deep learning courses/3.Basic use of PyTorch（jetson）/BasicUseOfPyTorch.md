3 Basic use of PyTorch
3.1 About PyTorch
3.1.1 Introduction

PyTorch is an open-source machine learning library based on the Torch library. It is widely used for deep learning applications and provides a wide range of tools and functions to make building and training neural networks easier. PyTorch is developed by Facebook's AI research group and is known for its simplicity, flexibility, and ease of use.
3.1.2 Features

Some of the key features of PyTorch are:

    Dynamic computation graphs that allow you to modify the network architecture on the fly
    Tensors with automatic differentiation, which make it easy to implement backpropagation and other gradient-based optimization algorithms
    A high-level interface for building and training neural networks
    A comprehensive set of tools for data processing and transformation, including support for loading and augmenting image data
    Integration with popular deep learning frameworks and tools like TensorFlow and Keras.

3.2 Tensors in PyTorch
3.2.1 Tensor

In PyTorch, a Tensor is a multi-dimensional matrix containing elements of a single data type. It is similar to NumPy's ndarray, but with additional support for GPUs. Tensors can be used to represent the inputs and outputs of a neural network, as well as the parameters of the network.
3.2.2 Create a Tensor

We can create a Tensor in PyTorch by using the torch.Tensor constructor. For example, to create a 2x3 Tensor of zeros, we can do the following:

python

import torch

# Create a 2x3 tensor of zeros
x = torch.Tensor(2, 3)
print(x)

Output:

lua

tensor([[0., 0., 0.],
        [0., 0., 0.]])

We can also create a Tensor from a list or a NumPy array:

python

import torch
import numpy as np

# Create a tensor from a list
x = torch.Tensor([1, 2, 3])
print(x)

# Create a tensor from a NumPy array
y = torch.from_numpy(np.array([1, 2, 3]))
print(y)

Output:

scss

tensor([1., 2., 3.])
tensor([1, 2, 3])

3.3 torchvision package introduction

torchvision is a library dedicated to processing images in PyTorch. It contains four main categories: datasets, models, transforms, and utils.

    datasets: contains popular datasets such as CIFAR10, MNIST, and ImageNet.
    models: contains popular pre-trained models for image classification and object detection.
    transforms: contains common image transformations such as random cropping, resizing, and normalization.
    utils: contains utility functions that are commonly used in computer vision tasks.

To use torchvision, we need to first install it by running pip install torchvision.
3.4 Convolutional Neural Networks

A neural network is a collection of neurons that are connected to each other. A Convolutional Neural Network (CNN) is a neural network that is designed to process data that has a grid-like structure, such as an image.

CNNs consist of multiple layers, with each layer performing a different type of operation. The layers include convolutional layers, pooling layers, and fully connected layers.

A convolutional layer applies a set of filters to an input image, producing a set of feature maps that highlight different aspects of the image. A pooling layer reduces the spatial size of the feature maps, while retaining the important information. Finally, the fully connected layers use the feature maps to classify the image.
3.5 Build the LeNet neural network and train the data set

In this section, we will use PyTorch to build the LeNet neural network and train it on the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.
3.5.1 Preparation before implementation

Before we can implement the LeNet neural network, we need to prepare the dataset and the required modules.

python

# Import related modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

# Load the dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

cifar_train_data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
cifar_test_data = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       transform=transform)

# Package the dataset
train_data_loader = torch.utils.data.DataLoader(cifar_train_data, batch_size=32,
                                          shuffle=True)
test_data_loader = torch.utils.data.DataLoader(cifar_test_data, batch_size=32,
                                         shuffle=True)

3.5.2 Implementation process

Now that we have prepared the dataset, we can proceed to build the LeNet neural network and train it on the CIFAR-10 dataset.

python

# Define the LeNet neural network
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
5. Configure the Loss Function and Optimizer for Training

Now that we have the neural network architecture defined, we need to specify how it will be trained. This involves specifying a loss function to measure the error between the predicted and actual output, and an optimizer to adjust the weights of the network to minimize the loss.

In this example, we'll use the cross-entropy loss function and the stochastic gradient descent (SGD) optimizer.

python

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)

The nn.CrossEntropyLoss() function computes the cross-entropy loss, which is commonly used for multi-class classification problems. It takes as input the output of the neural network and the true labels, and returns a scalar value representing the loss.

The optim.SGD() function creates an SGD optimizer, which updates the weights of the network in the direction of the negative gradient of the loss function. It takes as input the network parameters to be optimized, the learning rate (lr), and the momentum value (momentum). The learning rate controls how much the weights are adjusted with each update, and the momentum value helps to smooth out the updates and prevent oscillations.
6. Start Training and Testing
After preparing the data and defining the model, we can start training and testing our model using the following code:

python

# Training the Model
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_data_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

# Testing the Model
correct = 0
total = 0
with torch.no_grad():
    for data in test_data_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

In the code above, we first train the model for a specified number of epochs, which is defined as num_epochs in our code. For each epoch, we iterate over the train_data_loader and perform the following steps:

    Zero out the gradients using optimizer.zero_grad().
    Pass the input through the network to get the outputs.
    Calculate the loss between the predicted outputs and the true labels using the cross-entropy loss function defined earlier.
    Backpropagate the loss to compute the gradients using loss.backward().
    Update the model parameters using the gradients computed in the previous step with optimizer.step().

After training the model, we then test it on the test dataset to calculate its accuracy. We iterate over the test_data_loader and for each batch of data, we perform the following steps:

    Pass the input through the network to get the outputs.
    Find the class with the highest score in the outputs using torch.max().
    Compare the predicted classes with the true labels to calculate the accuracy.

The overall accuracy of the model on the test dataset is then printed out to the console.

Congratulations, you have successfully trained a convolutional neural network on the CIFAR-10 dataset using PyTorch! With this knowledge, you can build more complex neural networks for various tasks such as image classification, object detection, and even natural language processing.
PyTorch is also highly flexible and customizable, allowing researchers and developers to create their own deep learning models from scratch and incorporate them seamlessly into their existing projects.

Overall, PyTorch is an incredibly powerful tool for deep learning and has gained widespread popularity among researchers, developers, and hobbyists alike. With its intuitive interface and extensive library of pre-built functions and models, PyTorch makes it easier than ever to dive into the world of deep learning and create cutting-edge AI applications.
