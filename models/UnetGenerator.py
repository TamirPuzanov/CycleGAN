import torch.nn as nn
import torch


__all__ = [
    "UnetGenerator",
    "unet_generator_32x4x6",
    "unet_generator_32x5x9",
    "unet_generator_64x4x6",
    "unet_generator_64x5x9"
]


class ResidualBlock(nn.Module):
    def __init__(self, n):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n, n, 3, 1, 1), nn.InstanceNorm2d(n), 
            nn.LeakyReLU(), nn.Conv2d(n, n, 3, 1, 1)
        )
        
        self.norm = nn.InstanceNorm2d(n)
        self.f = nn.LeakyReLU()
        
    def forward(self, x):
        return self.f(self.norm(self.conv(x) + x))


class UnetUpConv(nn.Module):
    def __init__(self, n, c):
        super(UnetUpConv, self).__init__()
        layers = []
        c = c[::-1] + [n]
        
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        
        for i in range(len(c) - 1):
            layers += [nn.Sequential(
                nn.ReflectionPad2d(3), nn.Conv2d(c[i] + 3, c[i + 1], 7, 1, 0),
                nn.InstanceNorm2d(c[i + 1]), nn.LeakyReLU(),
                
                nn.Conv2d(c[i + 1], c[i + 1], 3, 1, 1),
                nn.InstanceNorm2d(c[i + 1]),
            )]
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, c, x):
        for i, m in enumerate(self.model):
            x = self.upsample(x)
            x = m(torch.cat((x, c[-(i + 1)]), dim=1))
        return x

class UnetDownConv(nn.Module):
    def __init__(self, n, c):
        super(UnetDownConv, self).__init__()
        layers = []
        c = [n] + c
        
        for i in range(len(c) - 1):
            layers += [nn.Sequential(
                nn.ReflectionPad2d(3), nn.Conv2d(c[i], c[i + 1], 7, 1, 0),
                nn.InstanceNorm2d(c[i + 1]), nn.LeakyReLU(),
                
                nn.Conv2d(c[i + 1], c[i + 1], 3, 1, 1),
                nn.InstanceNorm2d(c[i + 1]),
            )]
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        c = []
        for m in self.model:
            x = m(x)
            c.append(x[:,:3])
            
            x = torch.max_pool2d(x, 2)
        
        return c, x

class UnetGenerator(nn.Module):
    def __init__(self, c=[32, 64, 128, 256], blocks=6):
        super(UnetGenerator, self).__init__()
        self.down = UnetDownConv(3, c)
        self.up = UnetUpConv(3, c)
        
        self.res = nn.Sequential(*[
            ResidualBlock(c[-1])
            for _ in range(blocks)
        ])
    
    def forward(self, x):
        c, x = self.down(x)
        x = self.res(x)
        x = self.up(c, x)
        return x


def unet_generator_32x4x6():
    return UnetGenerator(c=[32, 64, 128, 256], blocks=6)

def unet_generator_32x5x9():
    return UnetGenerator(c=[32, 64, 128, 256, 512], blocks=9)

def unet_generator_64x4x6():
    return UnetGenerator(c=[64, 128, 256, 512], blocks=6)

def unet_generator_64x5x9():
    return UnetGenerator(c=[64, 128, 256, 512, 1024], blocks=9)