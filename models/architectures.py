import torch
import torch.nn as nn
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    """
    Custom PyTorch Dataset for handling data and labels.

    Args:
        data: numpy array or list containing the input data
        labels: numpy array or list containing the corresponding labels
    """
    def __init__(self, data, labels):
        self.data = torch.Tensor(data)
        self.labels = torch.Tensor(labels)

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a sample and its label at the specified index.

        Args:
            idx: int, index of the sample to retrieve

        Returns:
            tuple: (sample, label) - the data sample and its corresponding label
        """
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label


class InterdotCNN(nn.Module):
    """
    Convolutional Neural Network for interdot region classification.

    This CNN architecture is designed to identify interdot regions in quantum dot measurements.
    It consists of convolutional layers followed by fully connected layers.
    """
    def __init__(self):
        """
        Initialize the InterdotCNN model with convolutional and fully connected layers.
        """
        super(InterdotCNN, self).__init__()
        self.convolutional = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fully_connected = nn.Sequential(
            nn.Linear(in_features=64 * 5 * 5, out_features=128),
            nn.Tanh(),
            nn.Dropout(p=0.15),
            nn.Linear(in_features=128, out_features=32),
            nn.Tanh(),
            nn.Dropout(p=0.07),
            nn.Linear(in_features=32, out_features=1),
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: torch.Tensor, input tensor of shape [batch_size, 1, height, width]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, 1]
        """
        x = self.convolutional(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x


class LineCNN(nn.Module):
    """
    Convolutional Neural Network for line detection in quantum dot measurements.

    This CNN architecture uses an encoder-decoder structure to identify and extract
    line features from input images.
    """
    def __init__(self):
        """
        Initialize the LineCNN model with encoder and decoder components.
        """
        super(LineCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=13, stride=1, padding=6),
            nn.BatchNorm2d(num_features=8, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.AvgPool2d(2),  # [16, 50, 50]
            nn.Conv2d(8, 16, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.AvgPool2d(2),  # [16, 25, 25]
            nn.Conv2d(16, 32, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.AvgPool2d(2),  # [32, 12, 12]
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.AvgPool2d(2),  # [64, 6, 6]
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1),
            nn.ReLU(),  # [32, 12, 12]
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=0, output_padding=0),
            nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.1),
            nn.ReLU(),  # [32, 25, 25]
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(num_features=8, eps=1e-05, momentum=0.1),
            nn.ReLU(),  # [16, 50, 50]
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
            # [16, 100, 100]
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: torch.Tensor, input tensor of shape [batch_size, 1, height, width]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, 1, height, width] containing
                          the detected line features
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class AngleCNN(nn.Module):
    """
    Convolutional Neural Network for angle prediction in quantum dot measurements.

    This CNN architecture is designed to predict angles from line features extracted
    from quantum dot measurements.
    """
    def __init__(self):
        """
        Initialize the AngleCNN model with convolutional and fully connected layers.
        """
        super(AngleCNN, self).__init__()
        self.convolutional = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(4),
        )

        self.fully_connected = nn.Sequential(
            nn.Linear(in_features=64 * 6 * 6, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=8)
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: torch.Tensor, input tensor of shape [batch_size, 1, height, width]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, 8] containing
                          the predicted angle features
        """
        x = self.convolutional(x)
        x = x.view(-1, 64 * 6 * 6)
        x = self.fully_connected(x)
        return x
