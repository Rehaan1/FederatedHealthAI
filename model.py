import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Note the model and functions here defined do not have any FL-specific components.

# Resnet18 model with additional fully connected layers
class Net(nn.Module):
    def __init__(self, num_classes: int):
        super(Net, self).__init__()
        # Load pre-trained ResNet18 model
        self.resnet = models.resnet18(weights='IMAGENET1K_V1')
        # Freeze the parameters of the pre-trained model
        for param in self.resnet.parameters():
            param.requires_grad = False
        # Get the number of features from the ResNet backbone
        num_features = self.resnet.fc.in_features
        # Replace the last fully connected layer for binary classification
        self.resnet.fc = nn.Identity()  # Remove the original fully connected layer
        
        # Add additional fully connected layers
        self.fc1 = nn.Linear(num_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# Resnet34 model with additional fully connected layers
# class Net(nn.Module):
#     def __init__(self, num_classes: int):
#         super(Net, self).__init__()
#         # Load pre-trained ResNet34 model
#         self.resnet = models.resnet34(weights='IMAGENET1K_V1')
#         # Freeze the parameters of the pre-trained model
#         for param in self.resnet.parameters():
#             param.requires_grad = False
#         # Get the number of features from the ResNet backbone
#         num_features = self.resnet.fc.in_features
#         # Replace the last fully connected layer for binary classification
#         self.resnet.fc = nn.Identity()  # Remove the original fully connected layer
        
#         # Add additional fully connected layers
#         self.fc1 = nn.Linear(num_features, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, num_classes)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.resnet(x)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x    

# class Net(nn.Module):
#     """A simple CNN suitable for simple vision tasks."""

#     def __init__(self, num_classes: int) -> None:
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 53 * 53, 120)  # Adjust this line
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, num_classes)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(x.size(0), -1)  # Adjust this line
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
    
    
def train(net, trainloader, optimizer, epochs, device: str):
    """Train the network on the training set.

    This is a fairly simple training loop for PyTorch.
    """
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    net.to(device)
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()


def test(net, testloader, device: str):
    """Validate the network on the entire test set.

    and report loss and accuracy.
    """
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    net.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy