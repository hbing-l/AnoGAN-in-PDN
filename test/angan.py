# -*- coding:utf8 -*-
# @TIME     : 2021/04/22
# @Author   : LiuHanbing
# @File     : test_angan.py


from __future__ import print_function
import os
import sys
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
import cv2
import pdb
import math

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", default=r"../experiments/0407updatedata_test", help="path to save experiments results")
parser.add_argument("--n_epoches", type=int, default=5000, help="number of epoches of training")
parser.add_argument("--batchSize", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--nz", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--imageSize", type=int, default=12, help="size of each image dimension")
parser.add_argument("--nc", type=int, default=1, help="number of image channels")
parser.add_argument("--gf_dim", type=int, default=16, help="channels of middle layers for generator")
parser.add_argument("--df_dim", type=int, default=16, help="channels of middle layers for discriminator")
parser.add_argument("--sample_interval", type=int, default=10, help="interval between image sampling")
parser.add_argument("--ano_param", type=float, default=0.1, help="weights of reconstruction error and disc loss")
parser.add_argument("--gen_pth", default=r"../experiments/0407updatedata/gen_6000.pth", help="pretrained model of gen")
parser.add_argument("--disc_pth", default=r"../experiments/0407updatedata/disc_6000.pth", help="pretrained model of disc")
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
test_data_list, test_data_cnt, max_test_data, mean_test_data = load_data(batch_size = opt.batchSize, train = False)

## model
gen = Generator(opt.nz, opt.nc, opt.gf_dim, ksize=4).to(device)
disc = Discriminator(opt.nc, opt.df_dim).to(device)
gen.load_state_dict(torch.load(opt.gen_pth))
disc.load_state_dict(torch.load(opt.disc_pth))
print("Pretrained models have been loaded.")

## Gaussian Distribution
def gen_z_gauss(i_size, nz):
    return torch.randn(i_size, nz, 1, 1, requires_grad=True, device=device)


opt.dataSize = test_data_cnt

## record results
import time
now = time.localtime()
strnow = time.strftime("%m%d-%H%M",now)
writer = SummaryWriter("../runs{0}/".format(opt.experiment[1:]) + strnow, comment=opt.experiment[1:])

## testing
gen.eval()
disc.eval()
loss = []
z = []
# tqdm_loader = tqdm.tqdm(test_data_list)
for i, test_input in enumerate(test_data_list):  # test_input torch.size([1,1,12,12])
    print(f"Test Sample {i+1} / {opt.dataSize}\n")
    test_input = test_input.to(device)
    z_inputs = gen_z_gauss(test_input.size(0), opt.nz).to(device)
    ones = torch.ones(test_input.size(0), 1).to(device)
    optimizerZ = optim.Adam([z_inputs], lr=opt.lr, betas=(opt.b1, opt.b2))
    ## inference
    with tqdm.tqdm(range(opt.n_epoches)) as t:
        for e in t:
            t.set_description(f"Epoch {e+1} /{opt.n_epoches}")
            ##
            optimizerZ.zero_grad()
            ano_G = gen(z_inputs)
            feature_ano_G, _ = disc(ano_G)
            feature_input, _ = disc(test_input)

            residual_loss = torch.sum(torch.abs(test_input-ano_G), dim=[1,2,3])
            disc_loss = torch.sum(torch.abs(feature_ano_G-feature_input), dim=[1,2,3])
 
            total_loss = (1.0-opt.ano_param)*residual_loss + (opt.ano_param)*disc_loss
            total_loss = total_loss.squeeze()
            #total_loss = total_loss.view(test_input.size(0), 1)
            ##
            final_l = total_loss.detach().cpu().flatten()
            score = 1 / math.exp(0.3*final_l) * 100

            writer.add_scalar("total loss{}".format(i), total_loss, e)
            total_loss.backward()
            optimizerZ.step()
            z.append(z_inputs)
            t.set_postfix(total_loss=final_l, grad=z_inputs[0,0,0,0].item(), score=score)


    loss.append(final_l)
    ano_G = gen(z_inputs)
    residule = torch.abs(test_input-ano_G)

    # vutils.save_image(torch.cat((test_input, ano_G, residule), dim=0), '{0}/{1}-0.png'.format(opt.experiment, i))
    
    single_out = F.interpolate(ano_G, (120, 120), mode='bilinear', align_corners = True)
    single_input = F.interpolate(test_input, (120, 120), mode='bilinear', align_corners = True)
    single_residule = F.interpolate(residule, (120, 120), mode='bilinear', align_corners = True)     
    vutils.save_image(torch.cat((single_input, single_out), dim=0), '{0}/{1}-0.png'.format(opt.experiment, i), normalize=True)
    # vutils.save_image(single_out,'{0}/{1}-out.png'.format(opt.experiment, i), normalize=True)
    # vutils.save_image(single_input,'{0}/{1}-in.png'.format(opt.experiment, i), normalize=True)
    vutils.save_image(single_residule,'{0}/{1}-score{2}.png'.format(opt.experiment, i, score), normalize=True)

    z_g = ano_G.detach().cpu().numpy()
    z_g = z_g.reshape(-1,144)
    if i == 0:
        generate_z = z_g
    else:
        generate_z = np.concatenate((generate_z, z_g), axis=0)

    residule = single_residule.detach().cpu().numpy()
    residule = (residule - np.min(residule)) / (np.max(residule) - np.min(residule)) * 255
    residule = residule.astype("uint8")
    residule = np.transpose(residule[0], [1, 2, 0])
    residule = cv2.applyColorMap(residule, cv2.COLORMAP_HOT)
    cv2.imwrite('{0}/{1}-1.png'.format(opt.experiment, i), residule)

# real_gen_z = (generate_z * max_test_data) + mean_test_data
# np.savetxt('{0}/generate_data.csv'.format(opt.experiment), real_gen_z, delimiter=',')