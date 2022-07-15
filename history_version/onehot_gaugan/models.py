import jittor as jt
from jittor import init
from jittor import nn
from jittor import models as jtmd
import numpy as np

def start_grad(model):
    for param in model.parameters():
        param.start_grad()

def stop_grad(model):
    for param in model.parameters():
        param.stop_grad()

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        jt.init.gauss_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        jt.init.gauss_(m.weight, 1.0, 0.02)
        jt.init.constant_(m.bias, 0.0)


class SPADE(nn.Module):
    def __init__(self, norm_nc):
        super(SPADE, self).__init__()
        self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        nhidden = 128
        kernal_size = 3
        padding = kernal_size // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(29, nhidden, kernel_size=kernal_size, padding=padding),
            nn.LeakyReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=kernal_size, padding=padding)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=kernal_size, padding=padding)

    def execute(self, x, segmap):
        normalized = self.param_free_norm(x)
        segmap = nn.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta
        return out

class SPADEResBlock(nn.Module):
    def __init__(self, fin, fout):
        super(SPADEResBlock, self).__init__()
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.bn_0 = nn.BatchNorm2d(fmiddle)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        self.bn_1 = nn.BatchNorm2d(fout)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)
            self.bn_s = nn.BatchNorm2d(fout)
        self.norm_0 = SPADE(fin)
        self.norm_1 = SPADE(fmiddle) 
        if self.learned_shortcut:
            self.norm_s = SPADE(fin)

    def execute(self, x, seg):
        x_s = self.shortcut(x, seg)
        dx = self.bn_0(self.conv_0(self.relu(self.norm_0(x, seg))))
        dx = self.bn_1(self.conv_1(self.relu(self.norm_1(dx, seg))))
        out = x_s + dx
        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.bn_s(self.conv_s(self.norm_s(x, seg)))
        else:
            x_s = x
        return x_s

    def relu(self, x):
        return nn.leaky_relu(x, 2e-1)

class Generator(nn.Module):
    def __init__(self, pad_size, random_input_hid_dim):
        super(Generator, self).__init__()
        self.random_input_hid_dim = random_input_hid_dim
        self.pad_size = pad_size
        self.kernel_size = pad_size * 2 + 1
        self.up = nn.Upsample(scale_factor=2)
        self.FC1 = nn.Linear(random_input_hid_dim, 6144)
        self.SPADE_layer1 = SPADEResBlock(512, 1024)
        self.SPADE_layer2 = SPADEResBlock(1024, 1024)
        self.SPADE_layer2_2 = SPADEResBlock(1024, 1024)
        self.SPADE_layer3 = SPADEResBlock(1024, 512)
        self.SPADE_layer3_2 = SPADEResBlock(512, 512)
        self.SPADE_layer4 = SPADEResBlock(512, 256)
        self.SPADE_layer5 = SPADEResBlock(256, 128)
        self.SPADE_layer6 = SPADEResBlock(128, 64)
        self.final_cov = nn.Conv2d(64, 3, 3, padding=1)
        
    def execute(self, random_input_vec, seg_img):
        x = self.FC1(random_input_vec)
        x = x.view(x.size(0), 512, 3, 4)
        x = self.up(x)
        print(x.shape)
        print(seg_img.shape)
        exit()
        x = self.SPADE_layer1(x, seg_img)
        x = self.up(x)
        x = self.SPADE_layer2(x, seg_img)
        x = self.SPADE_layer2_2(x, seg_img)
        x = self.up(x)
        x = self.SPADE_layer3(x, seg_img)
        x = self.up(x)
        x = self.SPADE_layer3_2(x, seg_img)
        x = self.up(x)
        x = self.SPADE_layer4(x, seg_img)
        x = self.up(x)
        x = self.SPADE_layer5(x, seg_img)
        x = self.up(x)
        x = self.SPADE_layer6(x, seg_img)
        x = self.final_cov(nn.leaky_relu(x, 2e-1))
        x = jt.tanh(x)
        return x


class ConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = 64
        self.conv1 = nn.Conv2d(3, ndf, kw, stride=2, padding=pw)
        self.bn1 = nn.BatchNorm2d(ndf)
        self.conv2 = nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw)
        self.bn2 = nn.BatchNorm2d(ndf * 2)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw)
        self.bn3 = nn.BatchNorm2d(ndf * 4)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw)
        self.bn4 = nn.BatchNorm2d(ndf * 8)
        self.conv5 = nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw)
        self.bn5 = nn.BatchNorm2d(ndf * 8)
        self.conv6 = nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw)
        self.bn6 = nn.BatchNorm2d(ndf * 8)
        
        self.so = s0 = 4
        self.fc_mu = nn.Linear(ndf * 8 * s0 * s0, 256)
        self.fc_var = nn.Linear(ndf * 8 * s0 * s0, 256)

        self.actvn = nn.LeakyReLU(0.2)

    def execute(self, x):
        if x.size(2) != 256 or x.size(3) != 256:
            x = nn.interpolate(x, size=(256, 256), mode='bilinear')
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(self.actvn(x)))
        x = self.bn3(self.conv3(self.actvn(x)))
        x = self.bn4(self.conv4(self.actvn(x)))
        x = self.bn5(self.conv5(self.actvn(x)))
        x = self.bn6(self.conv6(self.actvn(x)))
        x = self.actvn(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        return mu, logvar


class SubDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        padw = int(np.ceil((4 - 1.0) / 2))
        self.con1 = nn.Conv2d(32, 128, kernel_size=4, stride=2, padding=padw)
        self.bn1 = nn.BatchNorm2d(128)
        self.activation1 = nn.gelu
        self.con1_2 = nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=padw)
        self.bn1_2 = nn.BatchNorm2d(128)
        self.activation1_2 = nn.gelu
        self.con2 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=padw)
        self.bn2 = nn.BatchNorm2d(256)
        self.activation2 = nn.gelu
        self.con3 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=padw)
        self.bn3 = nn.BatchNorm2d(512)
        self.activation3 = nn.gelu
        self.con4 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=padw)
        self.bn4 = nn.BatchNorm2d(512)
        self.activation4 = nn.gelu
        self.con5 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=padw)
        self.bn5 = nn.BatchNorm2d(512)
        self.activation5 = nn.gelu
        self.con6 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=padw)
        self.bn6 = nn.BatchNorm2d(512)
        self.activation6 = nn.gelu
        self.con7 = nn.Conv2d(512, 1, kernel_size=4, stride=2, padding=padw)
        self.bn7 = nn.BatchNorm2d(1)
        self.activation7 = nn.gelu
        self.output_s = nn.Sigmoid()

    def execute(self, seg_imgs, real_imgs):
        x = jt.contrib.concat([seg_imgs, real_imgs], 1)
        x = self.bn1(self.con1(x))
        x = self.activation1(x)
        x = self.bn1_2(self.con1_2(x))
        x = self.activation1_2(x)
        OP0 = x
        x = self.bn2(self.con2(x))
        x = self.activation2(x)
        OP1 = x
        x = self.bn3(self.con3(x))
        x = self.activation3(x)
        OP2 = x
        x = self.bn4(self.con4(x))
        x = self.activation4(x)
        OP3 = x
        x = self.bn5(self.con5(x))
        x = self.activation5(x)
        OP4 = x
        x = self.bn6(self.con6(x))
        x = self.activation6(x)
        x = self.con7(x)
        OP5 = self.output_s(x)
        return [OP0, OP1, OP2, OP3, OP4, OP5]

class Discriminator(nn.Module):
    def __init__(self, subnet_num):
        super().__init__()
        self._num = subnet_num
        self.subnet1 = SubDiscriminator()
        self.subnet2 = SubDiscriminator()
        self.subnet3 = SubDiscriminator()

    def execute(self, seg_imgs, real_imgs):
        return [subnet(seg_imgs, real_imgs) for subnet in self.children()]

class VGG19(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = jtmd.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def execute(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]

if __name__ == '__main__':
    D = Discriminator(3)
    G = Generator(1, 256)
    E = ConvEncoder()
    V = VGG19()
    print(D)
    print(G)
    print(E)
    print(V)