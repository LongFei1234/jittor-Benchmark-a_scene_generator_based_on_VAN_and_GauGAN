import glob
import random
import os
import numpy as np
import jittor as jt
from jittor import init

from jittor.dataset.dataset import Dataset
import jittor.transform as transform
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, root, trans_real, trans_seg, mode="train"):
        super().__init__()
        self.transforms_real = transform.Compose(trans_real)
        self.transforms_seg = transform.Compose(trans_seg)
        self.mode = mode
        if self.mode == 'train':
            self.files = sorted(glob.glob(os.path.join(root, mode, "imgs") + "/*.*"))
        self.labels = sorted(glob.glob(os.path.join(root, mode, "labels") + "/*.*"))
        self.set_attrs(total_len=len(self.labels))
        print(f"from {mode} split load {self.total_len} images.")

    def __getitem__(self, index):
        # img_B: seg_img 
        # img_A: real_img
        
        label_path = self.labels[index % len(self.labels)]
        photo_id = label_path.split('/')[-1][:-4]
        img_B = Image.open(label_path)
        # img_B = Image.fromarray(np.array(img_B).astype("uint8")[:, :, np.newaxis].repeat(3,2))
        if self.mode == "train":
            img_A = Image.open(self.files[index % len(self.files)])
            #print(img_A)
            # if np.random.random() < 0.5:
            #     img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
            #     img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")
            img_A = self.transforms_real(img_A)
        else:
            img_A = np.empty([1])
        img_B = self.transforms_seg(img_B)

        return img_A, img_B, photo_id


def get_one_hot(x, class_num=29):
    return jt.zeros((x.shape[0], class_num, x.shape[2], x.shape[3])).scatter_(1, x, jt.float32(1.0))


if __name__ == '__main__':

    trans_real = [
        #transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transform.Resize(size=(384, 512), mode=Image.BICUBIC),
        transform.ToTensor(),
        transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
    trans_seg = [
        transform.Resize(size=(384, 512)),
        transform.ToTensor(),
    ]
    dataloader = ImageDataset('data', mode="train", trans_real=trans_real, trans_seg=trans_seg).set_attrs(
        batch_size=2,
        shuffle=True,
        num_workers=8,
    )
    import time
    for data_ in dataloader:
        imgA = data_[0]
        imgB = data_[1]
        # print('imgA:')
        # print(imgA.shape)
        # print(imgA[0])
        # print('imgB:')
        # print(imgB.shape)
        # print(imgB)
        st = time.time()
        y = get_one_hot(imgB)
        ed = time.time()
        print(ed-st)
        # print(y[0][3])
        # print(y.shape)
        break
