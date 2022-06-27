import torch.nn as nn


__all__ = [
    "ResnetGenerator",
    "resnet_generator_32x4x6",
    "resnet_generator_32x5x9",
    "resnet_generator_64x4x6",
    "resnet_generator_64x5x9"
]


class ResidualBlock(nn.Module):
    def __init__(self, n, norm=nn.InstanceNorm2d):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(n, n, 3),
            norm(n), nn.LeakyReLU(inplace=True),
            
            nn.ReflectionPad2d(1), nn.Conv2d(n, n, 3),
            norm(n),
        )

    def forward(self, x):
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    def __init__(self, c=[24, 32, 64, 128], blocks=4, norm=nn.InstanceNorm2d):
        super(ResnetGenerator, self).__init__()
        
        layers = [
            nn.ReflectionPad2d(3), nn.Conv2d(3, c[0], 7),
            norm(c[0]), nn.LeakyReLU(inplace=True),
        ]
        
        for i in range(len(c) - 1):
            layers += [
                nn.Conv2d(c[i], c[i + 1], 4, 2, 1),
                norm(c[i + 1]), nn.LeakyReLU(inplace=True),
            ]
        
        for _ in range(blocks):
            layers += [ResidualBlock(c[-1])]
        
        c = c[::-1]
        
        for i in range(len(c) - 1):
            layers += [
                nn.ConvTranspose2d(c[i], c[i + 1], 4, 2, 1),
                norm(c[i + 1]), nn.LeakyReLU(inplace=True),
            ]
        
        layers += [
            nn.ReflectionPad2d(3), nn.Conv2d(c[-1], 3, 7),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x): 
        return self.model(x) 


def resnet_generator_32x4x6():
    return ResnetGenerator(c=[32, 64, 128, 256], blocks=6)

def resnet_generator_32x5x9():
    return ResnetGenerator(c=[32, 64, 128, 256, 512], blocks=9)

def resnet_generator_64x4x6():
    return ResnetGenerator(c=[64, 128, 256, 512], blocks=6)

def resnet_generator_64x5x9():
    return ResnetGenerator(c=[64, 128, 256, 512, 1024], blocks=9)