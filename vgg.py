from torch import Tensor, nn


class VGG5(nn.Module):
    def __init__(
        self, n_classes: int = 10, bias: bool = False, last_bn: bool = False
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.act2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=bias)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.act3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=bias)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.act4 = nn.ReLU(inplace=True)

        self.flatten = nn.Flatten(start_dim=1)

        self.fc5 = nn.Linear(2 * 2 * 256, 512, bias=bias)
        self.bn5 = nn.BatchNorm1d(512)
        self.act5 = nn.ReLU(inplace=True)

        self.fc6 = nn.Linear(512, n_classes, bias=True)

        # https://arxiv.org/pdf/2007.14234.pdf
        self.last_bn = nn.BatchNorm1d(n_classes) if last_bn else None

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.pool3(x)
        x = self.act3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.pool4(x)
        x = self.act4(x)

        x = self.flatten(x)

        x = self.fc5(x)
        x = self.bn5(x)
        x = self.act5(x)

        x = self.fc6(x)

        if self.last_bn is not None:
            x = self.last_bn(x)
        return x


class VGG7(nn.Module):
    """VGG7 model described in Ternary Weight Networks [1, 2].

    Full-precision should have 92.88% accuracy on CIFAR-10?
    Ternary should have 92.56% accuracy on CIFAR-10?

    [1]: https://arxiv.org/abs/1605.04711
    [2]: https://raw.githubusercontent.com/Thinklab-SJTU/twns/main/cls/litenet.py
    """

    def __init__(
        self, n_classes: int = 10, bias: bool = False, last_bn: bool = False
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 4 * 512, 1_024, bias=bias),
            nn.BatchNorm1d(1_024),
            nn.ReLU(inplace=True),
            nn.Linear(1_024, n_classes, bias=True),
        )
        # https://arxiv.org/pdf/2007.14234.pdf
        # Using BN after a last layer.
        self.last_bn = nn.BatchNorm1d(n_classes) if last_bn else None

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.linear(x)
        if self.last_bn is not None:
            x = self.last_bn(x)
        return x
