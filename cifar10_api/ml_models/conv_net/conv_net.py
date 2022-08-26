import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import datasets, transforms
from cifar10_api.ml_models.blocks import ConvBlock
from cifar10_api.ml_models.utils import Loader
from cifar10_api.ml_models.base_net import cifar10_labels, BaseNet


class ConvNetCifar10(BaseNet):
    def __init__(self, device=torch.device('cpu')):
        super().__init__(cifar10_labels, device)
        # In-channels: the depth of the input, for colored images, it is 3.
        # Out-channels: the number of filtered images (or the number of filters applied to the input,
        # or the depth of the convolutional layer).
        # 32 x 32 x 64
        self.__conv1 = ConvBlock(3, 64)
        # 32 x 32 x 128
        self.__conv2 = ConvBlock(64, 128)
        # 16 x 16 x 128
        self.__pool1 = nn.MaxPool2d(kernel_size=2)
        self.__res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        # 16 x 16 x 256
        self.__conv3 = ConvBlock(128, 256)
        # 8 x 8 x 256
        self.__pool2 = nn.MaxPool2d(kernel_size=2)
        # 8 x 8 x 512
        self.__conv4 = ConvBlock(256, 512)
        # 4 x 4 x 512
        self.__pool3 = nn.MaxPool2d(kernel_size=2)
        self.__res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        # 1 x 1 x 512
        self.__pool4 = nn.MaxPool2d(kernel_size=4)
        self.__fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 10)
        )

    def forward(self, img):
        output = self.__conv1(img)
        output = self.__conv2(output)
        output = self.__pool1(output)
        output = self.__res1(output) + output
        output = self.__conv3(output)
        output = self.__pool2(output)
        output = self.__conv4(output)
        output = self.__pool3(output)
        output = self.__res2(output) + output
        output = self.__pool4(output)
        output = torch.flatten(output, start_dim=1)
        output = self.__fc(output)
        return output

    def train_model(self, train_loader, valid_loader, eporchs, filename=None):
        loss_fn = nn.CrossEntropyLoss()
        optim = torch.optim.Adam(self.parameters(), lr=0.001)
        return super().base_train_model(train_loader, valid_loader, loss_fn, optim, eporchs, filename, verbose=True)

    def test_model(self, test_loader):
        loss_fn = nn.CrossEntropyLoss()
        return super().base_test_model(test_loader, loss_fn, verbose=True)

    def init_loader(self, batch_size):
        # Prepare data
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(45),
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

    @staticmethod
    def __inference_transform(img):
        transform_seq = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
        ])
        return transform_seq(img).view(1, 3, 32, 32)

    def predict(self, img):
        self.load_model('cifar10_api/ml_models/conv_net/convnet_cifar10.pt')
        return super().base_predict(img, ConvNetCifar10.__inference_transform)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eporchs = 2
    convnetCifar10 = ConvNetCifar10(device).to_device()
    loader = convnetCifar10.init_loader(batch_size=256)
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


main()
