import torch

from torch import nn

LABELS = {
    'smartwatches': [
        "Amazfit GTR 3 Pro",
        "Amazfit T-Rex",
        "Apple Watch 8",
        "Apple Watch Ultra",
        "Fitbit Sense",
        "Fitbit Sense 2",
        "Garmin Lily",
        "Garmin Venu 2 Plus",
        "Garmin Vivoactive 4",
        "Google Pixel Watch",
        "Huawei Watch 3",
        "Mobvoi TicWatch E3",
        "Samsung Galaxy Watch 4 Classic",
        "Samsung Galaxy Watch 5",
        "Samsung Galaxy Watch 5 Pro",
        "Samsung Galaxy Watch Active 2"
    ], 'toothbrushes': [
        "7am2m AM105",
        "Happybrush Eco VIBE 3",
        "Lächen RM-H9",
        "Lächen RM-T8",
        "Olybo ST-A9",
        "Oral-B Genius X",
        "Oral-B iO 10",
        "Oral-B Pro3 3000",
        "Oral-B Pulsonic Slim Luxe 4500",
        "Oral-B Smart",
        "Philips Sonicare 3100 Series",
        "Philips Sonicare DiamondClean",
        "Philips Sonicare ExpertClean",
        "Philips Sonicare Prestige",
        "Philips Sonicare ProtectiveClean",
        "Phylian Sonic Electric Toothbrush",
        "Seago SG-507",
        "Tristan-Auron Sonic Toothbrush",
        "YUNCHI Y7"
    ]
}


class InceptionModule(nn.Module):
    def __init__(self, options: dict):
        super(InceptionModule, self).__init__()

        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(
                in_channels=options['in_channels'],
                out_channels=options['out_channels']['1x1'],
                kernel_size=1
            ),
            options['activation_function']()
        )

        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(
                in_channels=options['in_channels'],
                out_channels=options['out_channels']['3x3_reduction'],
                kernel_size=1
            ),
            options['activation_function'](),
            nn.Conv2d(
                in_channels=options['out_channels']['3x3_reduction'],
                out_channels=options['out_channels']['3x3'],
                kernel_size=3,
                padding=1
            ),
            options['activation_function']()
        )

        self.conv_5x5 = nn.Sequential(
            nn.Conv2d(
                in_channels=options['in_channels'],
                out_channels=options['out_channels']['5x5_reduction'],
                kernel_size=1
            ),
            options['activation_function'](),
            nn.Conv2d(
                in_channels=options['out_channels']['5x5_reduction'],
                out_channels=options['out_channels']['5x5'],
                kernel_size=5,
                padding=2
            ),
            options['activation_function']()
        )

        self.max_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            nn.Conv2d(
                in_channels=options['in_channels'],
                out_channels=options['out_channels']['max'],
                kernel_size=1
            ),
            options['activation_function']()
        )

    def forward(self, x):
        x_1x1 = self.conv_1x1(x)
        x_3x3 = self.conv_3x3(x)
        x_5x5 = self.conv_5x5(x)
        x_max = self.max_pool(x)
        y = torch.cat([x_1x1, x_3x3, x_5x5, x_max], dim=1)
        return y


class ResNetModule(nn.Module):
    def __init__(self, options: dict):
        super(ResNetModule, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=options['in_channels'],
                out_channels=options['out_channels'],
                stride=options['stride'],
                kernel_size=(3, 3),
                padding=1
            ),
            nn.BatchNorm2d(num_features=options['out_channels']),
            options['activation_function'](),
            nn.Conv2d(
                in_channels=options['out_channels'],
                out_channels=options['out_channels'],
                kernel_size=(3, 3),
                padding=1
            ),
            nn.BatchNorm2d(num_features=options['out_channels'])
        )

        self.downsample = None

        if options['stride'] != 1 or options['in_channels'] != options['out_channels']:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=options['in_channels'],
                    out_channels=options['out_channels'],
                    stride=options['stride'],
                    kernel_size=(1, 1)
                ),
                nn.BatchNorm2d(num_features=options['out_channels'])
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        y = self.net(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        y += identity
        y = self.relu(y)

        return y


class Classifier(nn.Module):
    def __init__(self, out_features=19):
        super(Classifier, self).__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), ceil_mode=True),
            nn.LocalResponseNorm(size=5),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), ceil_mode=True)
        )

        self.resnet_1 = ResNetModule(options={
            'in_channels': 192,
            'out_channels': 32,
            'stride': 1,
            'activation_function': nn.ReLU
        })

        self.resnet_2 = ResNetModule(options={
            'in_channels': 192,
            'out_channels': 128,
            'stride': 1,
            'activation_function': nn.ReLU
        })

        self.inception_1 = InceptionModule(options={
            'in_channels': 192,
            'out_channels': {
                '1x1': 64,
                '3x3_reduction': 96,
                '3x3': 128,
                '5x5_reduction': 16,
                '5x5': 32,
                'max': 32
            },
            'activation_function': nn.ReLU
        })

        self.inception_2 = InceptionModule(options={
            'in_channels': 256,
            'out_channels': {
                '1x1': 128,
                '3x3_reduction': 128,
                '3x3': 192,
                '5x5_reduction': 32,
                '5x5': 96,
                'max': 64
            },
            'activation_function': nn.ReLU
        })

        self.resnet_3 = ResNetModule(options={
            'in_channels': 256,
            'out_channels': 256,
            'stride': 1,
            'activation_function': nn.ReLU
        })

        self.inception_3 = InceptionModule(options={
            'in_channels': 608,
            'out_channels': {
                '1x1': 256,
                '3x3_reduction': 256,
                '3x3': 32,
                '5x5_reduction': 64,
                '5x5': 32,
                'max': 128
            },
            'activation_function': nn.ReLU
        })

        self.resnet_4 = ResNetModule(options={
            'in_channels': 736,
            'out_channels': 736,
            'stride': 1,
            'activation_function': nn.ReLU
        })

        self.final = nn.Sequential(
            nn.AvgPool2d((7, 7)),
            nn.Dropout(p=0.4),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=(4 * 4 * 736), out_features=out_features)
        )

    def forward(self, x):
        branch1 = self.initial(x)

        branch2a = self.resnet_1(branch1)
        branch2b = self.resnet_2(branch1)
        branch2c = self.inception_1(branch1)

        branch3a = self.inception_2(branch2c)
        branch3b = self.resnet_3(branch2c)

        branch4 = torch.cat((branch2b, branch3a), dim=1)
        branch4 = self.inception_3(branch4)

        branch5 = torch.cat((branch2a, branch3b, branch4), dim=1)
        branch5 = self.resnet_4(branch5)
        branch5 = self.final(branch5)

        return branch5
