import jittor as jt
from jittor import init
from jittor import nn
import jittor.transform as transform
import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
import cv2
import time

from models import *
from datasets import *
from lossfunc import *

from tensorboardX import SummaryWriter

import warnings
warnings.filterwarnings("ignore")

jt.flags.use_cuda = 1

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--data_path", type=str, default="./jittor_landscape_200k")
parser.add_argument("--output_path", type=str, default="./results/flickr")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=384, help="size of image height")
parser.add_argument("--img_width", type=int, default=512, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument(
    "--sample_interval", type=int, default=500, help="interval between sampling of images from generators"
)
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints")
opt = parser.parse_args()
print(opt)

# TODO
def save_image(img, path, nrow=10):
    N,C,W,H = img.shape
    if N % 2 == 0:
        nrow = N / 2
    elif N < nrow:
        nrow = N
    img=img.transpose((1,0,2,3))
    ncol=int(N/nrow)
    img2=img.reshape([img.shape[0],-1,H])
    img=img2[:,:W*ncol,:]
    for i in range(1,int(img2.shape[1]/W/ncol)):
        img=np.concatenate([img,img2[:,W*ncol*i:W*ncol*(i+1),:]],axis=2)
    min_=img.min()
    max_=img.max()
    img=(img-min_)/(max_-min_)*255
    img=img.transpose((1,2,0))
    if C==3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path,img)
    return img

os.makedirs(f"{opt.output_path}/images/", exist_ok=True)
os.makedirs(f"{opt.output_path}/saved_models/", exist_ok=True)

writer = SummaryWriter(opt.output_path)

# Loss functions
criterion_GAN = nn.BCELoss()
criterion_pixelwise = nn.L1Loss()

# Loss weight of L1 pixel-wise loss between CONVed image and real image
lambda_pixel = 10

# Loss weight of vgg loss
lambda_vgg = 10

# Loss weight of origin picture L1 loss
lambda_L1 = 20


# Initialize generator and discriminator
discriminator = Discriminator(3)
generator = Generator(1, 256)
encoder = ConvEncoder()
vgg = VGG19()
stop_grad(vgg)
VGG_weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

if opt.epoch != 0:
    # Load pretrained models
    print('load models')
    generator.load(f"{opt.output_path}/saved_models/generator_{opt.epoch}.pkl")
    discriminator.load(f"{opt.output_path}/saved_models/discriminator_{opt.epoch}.pkl")
    encoder.load(f"{opt.output_path}/saved_models/encoder_{opt.epoch}.pkl")
    

# Optimizers
optimizer_G = jt.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = jt.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_E = jt.optim.Adam(encoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

def reparameterize(mu, logvar):
    std = jt.exp(0.5 * logvar)
    eps = jt.randn_like(std)
    return jt.multiply(eps, std) + mu

def encode_z(real_image):
    mu, logvar = encoder(real_image)
    z = reparameterize(mu, logvar)
    return z, mu, logvar


# Configure dataloaders
transforms = [
    transform.Resize(size=(opt.img_height, opt.img_width), mode=Image.BICUBIC),
    transform.ToTensor(),
    transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]

dataloader = ImageDataset(opt.data_path, mode="train", transforms=transforms).set_attrs(
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)
val_dataloader = ImageDataset(opt.data_path, mode="val", transforms=transforms).set_attrs(
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=1,
)


# TODO
@jt.single_process_scope()
def eval(epoch, writer):
    os.makedirs(f"{opt.output_path}/images/test_fake_imgs/epoch_{epoch}", exist_ok=True)
    for i, (_, real_A, photo_id) in enumerate(val_dataloader):
        random_vec = jt.randn(real_A.shape[0], 256)
        fake_B = generator(random_vec, real_A)
        
        if i == 0:
            # visual image result
            img_sample = np.concatenate([real_A.data, fake_B.data], -2)
            img = save_image(img_sample, f"{opt.output_path}/images/epoch_{epoch}_sample.png", nrow=5)
            writer.add_image('val/image', img.transpose(2,0,1), epoch)

        fake_B = ((fake_B + 1) / 2 * 255).numpy().astype('uint8')
        for idx in range(fake_B.shape[0]):
            cv2.imwrite(f"{opt.output_path}/images/test_fake_imgs/epoch_{epoch}/{photo_id[idx]}.jpg", fake_B[idx].transpose(1,2,0)[:,:,::-1])

warmup_times = -1
run_times = 3000
total_time = 0.
cnt = 0

# ----------
#  Training
# ----------

prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    for i, (real_B, real_A, _) in enumerate(dataloader):
        # Adversarial ground truths
        z, mu, logvar = encode_z(real_B)
        fake_B = generator(z, real_A)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        start_grad(discriminator)
        D_output_real = discriminator(real_A, real_B)
        D_output_fake = discriminator(real_A, fake_B.detach())
        L = OriGANLoss_D()
        loss_D = L(D_output_real, D_output_fake, criterion_GAN)
        optimizer_D.step(loss_D)
        writer.add_scalar('train/loss_D', loss_D.item(), epoch * len(dataloader) + i)

        # ------------------
        #  Train Generators
        # ------------------
        stop_grad(discriminator)
        optimizer_G.zero_grad()
        optimizer_E.zero_grad()

        D_output_real = discriminator(real_A, real_B)
        D_output_fake = discriminator(real_A, fake_B)

        L = KLDLoss()
        KLD_loss = L(mu, logvar) * 0.05
        L = OriGANLoss_G()
        loss_G_GAN = L(D_output_fake, criterion_GAN)
        L = Pix2PixLoss_G()
        loss_G_Feat = L(D_output_real, D_output_fake, criterion_pixelwise)
        vgg_real = vgg(real_B)
        vgg_fake = vgg(fake_B)
        loss_vgg = 0
        L = nn.L1Loss()
        for j in range(5):
            loss_vgg += L(vgg_real[j].detach(),vgg_fake[j]) * VGG_weights[j]
        loss_G_L1 = L(fake_B, real_B)
        loss_G = (loss_G_GAN + lambda_pixel * loss_G_Feat + lambda_vgg * loss_vgg + lambda_L1 * loss_G_L1) / 4

        optimizer_G.backward(loss_G)
        optimizer_G.step()
        optimizer_E.backward(KLD_loss) 
        optimizer_E.step()

        writer.add_scalar('train/loss_G', loss_G.item(), epoch * len(dataloader) + i)

        jt.sync_all(True)

        if jt.rank == 0:
            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            jt.sync_all()
            if batches_done % 5 == 0:
                sys.stdout.write(
                    "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, pixel: %f, vgg: %f, L1: %f] ETA: %s"
                    % (
                        epoch,
                        opt.n_epochs,
                        i,
                        len(dataloader),
                        loss_D.numpy()[0],
                        loss_G.numpy()[0],
                        loss_G_GAN.numpy()[0],
                        loss_G_Feat.numpy()[0],
                        loss_vgg.numpy()[0],
                        loss_G_L1.numpy()[0],
                        time_left,
                    )   
                )
        
        if jt.rank == 0 and (i+1) % 500 == 0:
            generator.save(os.path.join(f"{opt.output_path}/saved_models/generator_{epoch}.pkl"))
            encoder.save(os.path.join(f"{opt.output_path}/saved_models/encoder_{epoch}.pkl"))
            discriminator.save(os.path.join(f"{opt.output_path}/saved_models/discriminator_{epoch}.pkl"))

    if jt.rank == 0 and opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # eval(epoch, writer)
        # Save model checkpoints
        generator.save(os.path.join(f"{opt.output_path}/saved_models/generator_{epoch}.pkl"))
        encoder.save(os.path.join(f"{opt.output_path}/saved_models/encoder_{epoch}.pkl"))
        discriminator.save(os.path.join(f"{opt.output_path}/saved_models/discriminator_{epoch}.pkl"))
        eval(epoch, writer)