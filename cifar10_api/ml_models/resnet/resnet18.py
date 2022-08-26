import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from cifar10_api.ml_models.resnet.resnet import ResnetCifar10


class Resnet18Cifar10(ResnetCifar10):
    def __init__(self, batch_size, device=torch.device('cpu')) -> None:
        super().__init__(batch_size, device)
        self._model = resnet18(ResNet18_Weights.DEFAULT)
        # Not to require gradients for all layers
        for parameter in self._model.parameters():
            parameter.requires_grad = False
        # Require gradients for the fc layer
        self._model.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 10)
        )


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet18_model = Resnet18Cifar10(batch_size=64, device=device).to_device()
    resnet18_model.train_model('resnet18_cifar10.pt')


main()
