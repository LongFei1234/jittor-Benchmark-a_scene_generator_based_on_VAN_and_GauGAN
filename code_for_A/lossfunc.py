import jittor as jt
from jittor import init
from jittor import nn
import numpy as np

class OriGANLoss_D(nn.Module):
    def __init__(self):
        super().__init__()

    def execute(self, real_output_D_list, fake_output_D_list, criterion):
        loss_D_tot, D_x, D_G_z1 = 0, 0, 0
        n = len(real_output_D_list)
        for i in range(n):
            real_output_D = real_output_D_list[i][-1]
            fake_output_D = fake_output_D_list[i][-1]
            loss_D_real = criterion(real_output_D.squeeze(1), jt.ones(real_output_D.size()).squeeze(1))
            loss_D_fake = criterion(fake_output_D.squeeze(1), jt.zeros(real_output_D.size()).squeeze(1))
            loss_D_tot += (loss_D_real + loss_D_fake) / 2 
        return loss_D_tot / n

class OriGANLoss_G(nn.Module):
    def __init__(self):
        super().__init__()

    def execute(self, fake_output_D_list, criterion):
        loss_G = 0
        n = len(fake_output_D_list)
        for i in range(n):
            fake_output_D = fake_output_D_list[i][-1]
            loss_G += criterion(fake_output_D.squeeze(1), jt.ones(fake_output_D.size()).squeeze(1))
        return loss_G / n

class Pix2PixLoss_G(nn.Module):
    def __init__(self):
        super().__init__()

    def execute(self, real_output_D_list, fake_output_D_list, criterion):
        loss_G_GAN_Feat = 0
        n = len(real_output_D_list)
        for j in range(n):
            real_output_D = real_output_D_list[j]
            fake_output_D = fake_output_D_list[j]
            for i in range(len(real_output_D)-1):
                loss_G_GAN_Feat += criterion(real_output_D[i].detach(), fake_output_D[i])
        return loss_G_GAN_Feat / n

class KLDLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def execute(self, mu, logvar):
        return -0.5 * jt.sum(1 + logvar - mu.pow(2) - logvar.exp())