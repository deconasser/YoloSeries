import torch
from torch import nn

# Describing convolutional and max-pooling layers, as well as the number of repetitions for convolutional blocks.
architecture_config = [
    (7, 64, 2, 3),  # Convolutional block 1
    "M",            # Max-pooling layer 1
    (3, 192, 1, 1), # Convolutional block 2
    "M",            # Max-pooling layer 2
    (1, 128, 1, 0), # Convolutional block 3
    (3, 256, 1, 1), # Convolutional block 4
    (1, 256, 1, 0), # Convolutional block 5
    (3, 512, 1, 1), # Convolutional block 6
    "M",            # Max-pooling layer 3
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],  # Convolutional block 7 (repeated 4 times)
    (1, 512, 1, 0), # Convolutional block 8
    (3, 1024, 1, 1),# Convolutional block 9
    "M",            # Max-pooling layer 4
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],  # Convolutional block 10 (repeated 2 times)
    (3, 1024, 1, 1),# Convolutional block 11
    (3, 1024, 2, 1),# Convolutional block 12
    (3, 1024, 1, 1),# Convolutional block 13
    (3, 1024, 1, 1),# Convolutional block 14
]

# A convolutional block is defined with Conv2d, BatchNorm2d, and LeakyReLU layers.
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))

# The YOLOv1 model is defined with convolutional layers and fully connected layers (fcs).
class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    # Function to create convolutional layers based on the predefined architecture.
    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers += [
                    CNNBlock(
                        in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3],
                    )
                ]
                in_channels = x[1]

            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    # Function to create fully connected layers based on input parameters such as grid size, number of boxes, and number of classes.
    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5)),
        )