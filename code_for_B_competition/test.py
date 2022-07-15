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

import warnings
warnings.filterwarnings("ignore")

jt.flags.use_cuda = 1

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--input_path", type=str, default="./data/val/labels")
parser.add_argument("--output_path", type=str, default="./results")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
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

# Initialize generator and discriminator
generator = Generator(1, 256)

# Load pretrained models
print('load models')
generator.load('checkpoints/generator.pkl')

# Configure dataloaders
transforms = [
    transform.Resize(size=(opt.img_height, opt.img_width), mode=Image.BICUBIC),
    transform.ToTensor(),
    transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]
val_dataloader = ImageDataset(opt.input_path, mode="val", transforms=transforms).set_attrs(
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=1,
)

# TODO
@jt.single_process_scope()
def eval():
    os.makedirs(f"{opt.output_path}", exist_ok=True)
    for i, (_, real_A, photo_id) in enumerate(val_dataloader):
        random_vec = jt.randn(real_A.shape[0], 256)
        fake_B = generator(random_vec, real_A)
        fake_B = ((fake_B + 1) / 2 * 255).numpy().astype('uint8')
        for idx in range(fake_B.shape[0]):
            cv2.imwrite(f"{opt.output_path}/{photo_id[idx]}.jpg", fake_B[idx].transpose(1,2,0)[:,:,::-1])

eval()
