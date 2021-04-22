# -*- coding:utf8 -*-
# @TIME     : 2021/04/22
# @Author   : LiuHanbing
# @File     : anogan.py

'''
reference:
'''


from __future__ import print_function
import os
import sys
import pdb

#current_dir = os.getcwd()
sys.path.append('..')
import tqdm
import torch
from torch.utils.data import DataLoader
from models.CNN_anogan import Generator, Discriminator
from dataload.DNdata import load_data
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", default=r"../experiments/0409addg", help="path to save experiments results")
parser.add_argument("--n_epoches", type=int, default=20000, help="number of epoches of training")
parser.add_argument("--batchSize", type=int, default=144, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--nz", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--imageSize", type=int, default=12, help="size of each image dimension")
parser.add_argument("--nc", type=int, default=1, help="number of image channels")
parser.add_argument("--gf_dim", type=int, default=32, help="channels of middle layers for generator")
parser.add_argument("--df_dim", type=int, default=16, help="channels of middle layers for discriminator")
parser.add_argument("--sample_interval", type=int, default=500, help="interval between image sampling")
parser.add_argument("--smooth", type=float, default=0.1, help="label smothing scalar for real data")
opt = parser.parse_args()
print(opt)
os.makedirs(opt.experiment, exist_ok=True)

## cudnn
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

## device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

## dataset
train_data_list, train_data_cnt, max_train_data, mean_train_data = load_data(batch_size = opt.batchSize, train = True)

## model
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)

## model
gen = Generator(opt.nz, opt.nc, opt.gf_dim, ksize=4).to(device)
disc = Discriminator(opt.nc, opt.df_dim).to(device)

gen.apply(weights_init)
disc.apply(weights_init)

## adversarial loss
gen_optimizer = optim.Adam(gen.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
disc_optimizer = optim.Adam(disc.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
gen_criteria = nn.BCELoss()
disc_criteria = nn.BCELoss()

## record results
import time
now = time.localtime()
strnow = time.strftime("%m%d-%H%M",now)
writer = SummaryWriter("../runs{0}/".format(opt.experiment[1:]) + strnow, comment=opt.experiment[1:])

## Gaussian Distribution
def gen_z_gauss(i_size, nz):
    return torch.randn(i_size, nz, 1, 1).to(device)

opt.dataSize = train_data_cnt

## Training
record = 0
with tqdm.tqdm(range(opt.n_epoches)) as t:
    for e in t:
        t.set_description(f"Epoch {e+1} /{opt.n_epoches} Per epoch {opt.dataSize}")

        gen_epoch_loss = 0.0
        disc_epoch_loss = 0.0

        for i, inputs in enumerate(train_data_list): # inputs torch.size([n, 1, 12, 12])  torch.float32
            batch_size = inputs.size(0)
            inputs = inputs.to(device)
            label_smoothing = opt.smooth * torch.rand(batch_size).to(device)
            label_real = torch.ones(batch_size).to(device)
            label_fake = torch.zeros(batch_size).to(device)

            # Update 'G' : max log(D(G(z)))
            gen_optimizer.zero_grad()
            noise = gen_z_gauss(batch_size, opt.nz)
            outputs = gen(noise)
            _, D_fake = disc(outputs)
            pdb.set_trace()
            gen_loss = gen_criteria(D_fake, label_real)
            gen_loss.backward()
            gen_optimizer.step()
            gen_epoch_loss += gen_loss.item() * batch_size

            if (record + 1) % 2 == 0:
                # Update "D": max log(D(x)) + log(1-D(G(z))
                disc_optimizer.zero_grad()

                _, D_real = disc(inputs)
                disc_loss_real = disc_criteria(D_real, label_real - label_smoothing)
                disc_loss_real.backward()

                noise = gen_z_gauss(batch_size, opt.nz)
                outputs = gen(noise)
                _, D_fake = disc(outputs.detach())
                disc_loss_fake = disc_criteria(D_fake, label_fake + label_smoothing)
                disc_loss_fake.backward()

                disc_loss = (disc_loss_fake + disc_loss_real) * 0.5
                disc_optimizer.step()
                disc_epoch_loss += disc_loss.item() * batch_size
                    
            ## record results
            if (record + 1) % opt.sample_interval == 0:
                single_out = outputs[0].unsqueeze(dim=0)
                single_out = F.interpolate(single_out, (120, 120), mode='bilinear', align_corners = True)
                single_input = inputs[0].unsqueeze(dim=0)
                single_input = F.interpolate(single_input, (120, 120), mode='bilinear', align_corners = True)                
                vutils.save_image(single_out,'{0}/outputs_{1}.png'.format(opt.experiment, record+1), normalize=True)
                vutils.save_image(single_input,'{0}/inputs_{1}.png'.format(opt.experiment, record+1), normalize=True)
            
            record += 1
        
        ## End of epoch
        gen_epoch_loss /= opt.dataSize
        disc_epoch_loss /= opt.dataSize
        disc_epoch_loss /= 2.0
        t.set_postfix(gen_epoch_loss=gen_epoch_loss, disc_epoch_loss=disc_epoch_loss, train_cnt=record)

        writer.add_scalar("gen_epoch_loss", gen_epoch_loss, e)
        writer.add_scalar("disc_epoch_loss", disc_epoch_loss, e)

        if (e+1) % 100 == 0:
        # save model parameters
            torch.save(gen.state_dict(), '{0}/gen_{1}.pth'.format(opt.experiment, e+1))
            torch.save(disc.state_dict(), '{0}/disc_{1}.pth'.format(opt.experiment, e+1))
        
        if (e+1) % 200 == 0:
        # save outputs data
            outputs = outputs.view(-1, int(opt.nc) * int(opt.imageSize) * int(opt.imageSize))
            outputs = outputs.detach().cpu().numpy()
            real_outputs = (outputs * max_train_data) + mean_train_data
            np.savetxt('{0}/outputs_data_{1}.csv'.format(opt.experiment, e+1), real_outputs, delimiter=',')

writer.close()