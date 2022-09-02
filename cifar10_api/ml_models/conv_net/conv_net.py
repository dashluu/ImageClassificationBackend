import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from cifar10_api.ml_models.blocks import ConvBlock, ResidualBlock
from cifar10_api.ml_models.utils import Loader
from cifar10_api.ml_models.base_net import cifar10_labels, BaseNet


class ConvNetCifar10(BaseNet):
    def __init__(self, device=torch.device('cpu')):
        super().__init__(cifar10_labels, device)
        # In-channels: the depth of the input, for colored images, it is 3.
        # Out-channels: the number of filtered images (or the number of filters applied to the input,
        # or the depth of the convolutional layer).
        self.conv = nn.Sequential(
            # 32 x 32 x 64
            ConvBlock(3, 64),
            # 32 x 32 x 128
            ConvBlock(64, 128),
            # 16 x 16 x 128
            nn.MaxPool2d(kernel_size=2),
            ResidualBlock([ConvBlock(128, 128), ConvBlock(128, 128)]),
            # 16 x 16 x 256
            ConvBlock(128, 256),
            # 8 x 8 x 256
            nn.MaxPool2d(kernel_size=2),
            # 8 x 8 x 512
            ConvBlock(256, 512),
            # 4 x 4 x 512
            nn.MaxPool2d(kernel_size=2),
            ResidualBlock([ConvBlock(512, 512), ConvBlock(512, 512)]),
            # 1 x 1 x 512
            nn.MaxPool2d(kernel_size=4)
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 10)
        )

    def forward(self, img):
        output = self.conv(img)
        output = torch.flatten(output, start_dim=1)
        output = self.fc(output)
        return output

    def train_model(self, train_loader, valid_loader, epochs, filename=None):
        loss_fn = nn.CrossEntropyLoss()
        optim = torch.optim.SGD(self.parameters(), lr=0.02, momentum=0.85, weight_decay=1e-4)
        # scheduler = torch.optim.lr_scheduler.CyclicLR(optim, base_lr=0.1, max_lr=3)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=0.02, epochs=epochs,
                                                        steps_per_epoch=len(train_loader))
        return super().base_train_model(train_loader, valid_loader, epochs, loss_fn, optim, scheduler=scheduler,
                                        in_batch=True, filename=filename, verbose=True)

    def test_model(self, test_loader):
        loss_fn = nn.CrossEntropyLoss()
        return super().base_test_model(test_loader, loss_fn, verbose=True)

    def init_loader(self, batch_size):
        # Prepare data
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
        ])
        test_transform = valid_transform
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
        train_loader = DataLoader(train_set, batch_size, shuffle=True)
        valid_loader = DataLoader(valid_set, batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size, shuffle=True)
        return Loader(train_loader, valid_loader, test_loader)

    def predict(self, img):
        self.load_model('cifar10_api/ml_models/conv_net/convnet_cifar10_2.pt')
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
        ])
        m_img = tf(img).view(1, 3, 32, 32)
        return super().base_predict(m_img)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eporchs = 50
    convnetCifar10 = ConvNetCifar10(device).to_device()
    loader = convnetCifar10.init_loader(batch_size=512)
    result = convnetCifar10.train_model(loader.train_loader, loader.test_loader, eporchs,
                                        filename='convnet_cifar10.pt')
    print(f'validation mean accuracy: {result.accuracy: .3f}%')
    print(f'accuracy per class: {result.class_accuracy}')
    print(f'train distribution per class: {result.class_dist.train_dist}')
    print(f'valid distribution per class: {result.class_dist.valid_dist}')
    eporch_axis = np.arange(0, eporchs, 1)
    fig, axis = plt.subplots()
    train_line, = axis.plot(eporch_axis, result.train_loss, 'r', label='train')
    valid_line, = axis.plot(eporch_axis, result.valid_loss, 'b', label='valid')
    axis.legend(handles=[train_line, valid_line])
