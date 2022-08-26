import torch
import ssl

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import datasets, transforms

from cifar10_api.ml_models.base_net import cifar10_labels, BaseNet

# Solve the ssl problem, do not remove this line
ssl._create_default_https_context = ssl._create_unverified_context


class ResnetCifar10(BaseNet):
    def __init__(self, batch_size, device=torch.device('cpu')) -> None:
        super().__init__(cifar10_labels, device)
        # Define a model
        self._model = None
        # Define image transforms
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(45),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        valid_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_transform = valid_transform
        self._transform = test_transform
        # Download the data
        big_train_set = datasets.CIFAR10('dataset', train=True, transform=train_transform, download=True)
        big_valid_set = datasets.CIFAR10('dataset', train=True, transform=valid_transform, download=True)
        test_set = datasets.CIFAR10('dataset', train=False, transform=test_transform, download=True)
        # Splitting the data
        num_indices = len(big_train_set)
        indices = list(range(num_indices))
        train_indices, valid_indices = train_test_split(indices)
        train_set = Subset(big_train_set, train_indices)
        valid_set = Subset(big_valid_set, valid_indices)
        # Create dataloaders
        self._batch_size = batch_size
        self._train_loader = DataLoader(train_set, batch_size, shuffle=True)
        self._valid_loader = DataLoader(valid_set, batch_size, shuffle=True)
        self._test_loader = DataLoader(test_set, batch_size, shuffle=True)

    def forward(self, img):
        return self._model(img)

    def train_model(self, filename=None):
        loss_fn = nn.CrossEntropyLoss()
        optim = torch.optim.Adam(self._model.fc.parameters(), lr=0.001)
        return super().base_train_model(self._train_loader, self._valid_loader, loss_fn, optim, eporchs=100,
                                        filename=filename, verbose=True)

    def test_model(self):
        loss_fn = nn.CrossEntropyLoss()
        return super().base_test_model(self._test_loader, loss_fn, verbose=True)
