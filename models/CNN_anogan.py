# -*- coding:utf-8 -*-
# @TIME     : 2021/04/22
# @Author   : LiuHanbing
# @File     : CNN_anogan.py

'''
reference: https://github.com/qqsuhao/AnoGAN-MvTec-grid-
'''


import torch
import torch.nn as nn
from torchsummary import summary
import pdb

class Generator(nn.Module):
    def __init__(self, z_dim, c_dim, gf_dim, ksize):
        super(Generator, self).__init__()
        self.fc = nn.Linear(z_dim, gf_dim*4*ksize*ksize, bias=True)
        self.layer1 = nn.Sequential(
            nn.BatchNorm2d(gf_dim*4),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(gf_dim*4, gf_dim*2, ksize, 2, 1, bias=False),
            nn.BatchNorm2d(gf_dim*2),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(gf_dim*2, gf_dim, ksize, 2, 1, bias=False),
            nn.BatchNorm2d(gf_dim),
            nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(gf_dim, 1, ksize, 2, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.layer5 = nn.Sequential(
            nn.Linear(32*32, 12*12, bias=True),
            nn.Tanh(),
        )

    # ConvTranspose2d o = k - 2*p + (i-1)*s + op
    def forward(self, inputs):   # inputs [n,100,1,1]
        h1 = inputs.reshape(-1,100)
        h1 = self.fc(h1)
        h1 = h1.reshape(-1, 128, 4, 4) 
        h1 = self.layer1(h1)     # [n,64,4,4]
        h2 = self.layer2(h1)     # [n,32,8,8]
        h3 = self.layer3(h2)     # [n,16,16,16]
        h4 = self.layer4(h3)     # [n,1,32,32]
        h4 = h4.reshape(-1, 1*32*32)
        h5 = self.layer5(h4)     # [n,144]
        outputs = h5.reshape(-1, 1, 12, 12)
        return outputs


class Discriminator(nn.Module):
    def __init__(self, c_dim, df_dim):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(c_dim, df_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(df_dim, df_dim*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(df_dim*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(df_dim*2, df_dim*4, 3, 1, 0, bias=False),
            nn.BatchNorm2d(df_dim*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            #nn.Sigmoid(),
        )
        self.layer4 = nn.Sequential(
            nn.Linear(df_dim*4, 1, bias=True),
            nn.Sigmoid(),
        )

    # Conv2d o = (i-k+2*p) / s + 1
    def forward(self, inputs):    # inputs [n,1,12,12]
        h1 = self.layer1(inputs)  # [n,16,6,6]
        h2 = self.layer2(h1)      # [n,32,3,3]
        h3 = self.layer3(h2)      # [n,64,1,1]
        h3 = h3.reshape([-1,64])
        outputs = self.layer4(h3) # [n,1]
        return h2, outputs.squeeze(1)        # by squeeze, get just float not float Tenosor



def print_net():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = Generator(100, 1, 32, 4).to(device)
    D = Discriminator(1, 16).to(device)
    summary(G, (100, 1, 1))
    summary(D, (1, 12, 12))


print_net()