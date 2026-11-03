import os
import numpy as np
import torch
import random
from skimage import io
from torch.utils import data
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
# from osgeo import gdal_array
import cv2

num_classes = 6
ST_COLORMAP = [[255,255,255], [0,128,0], [255,0,0], [0,255,0], [128,0,0], [128,128,128]]
ST_CLASSES = ['unchanged', 'cropland', 'road', 'forest', 'building','ground']

MEAN_A = np.array([113.40, 114.08, 116.45])
STD_A  = np.array([48.30,  46.27,  48.14])
MEAN_B = np.array([111.07, 114.04, 118.18])
STD_B  = np.array([49.41,  47.01,  47.94])

root = ''

colormap2label = np.zeros(256 ** 3)
for i, cm in enumerate(ST_COLORMAP):
    colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i

def Colorls2Index(ColorLabels):
    IndexLabels = []
    for i, data in enumerate(ColorLabels):
        IndexMap = Color2Index(data)
        IndexLabels.append(IndexMap)
    return IndexLabels

def Color2Index(ColorLabel):
    data = ColorLabel.astype(np.int32)
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    IndexMap = colormap2label[idx]
    #IndexMap = 2*(IndexMap > 1) + 1 * (IndexMap <= 1)
    IndexMap = IndexMap * (IndexMap < num_classes)
    return IndexMap

def Index2Color(pred):
    colormap = np.asarray(ST_COLORMAP, dtype='uint8')
    x = np.asarray(pred, dtype='int32')
    return colormap[x, :]

def showIMG(img):
    plt.imshow(img)
    plt.show()
    return 0

def normalize_image(im, time='A'):
    assert time in ['A', 'B']
    if time=='A':
        im = (im - MEAN_A) / STD_A
    else:
        im = (im - MEAN_B) / STD_B
    return im

def normalize_images(imgs, time='A'):
    for i, im in enumerate(imgs):
        imgs[i] = normalize_image(im, time)
    return imgs

def read_RSimages(mode):
    #assert mode in ['train', 'val', 'test']
    img_A_dir = os.path.join(root, mode, 'im1')
    img_B_dir = os.path.join(root, mode, 'im2')
    label_A_dir = os.path.join(root, mode, 'label1')
    label_B_dir = os.path.join(root, mode, 'label2')
    # To use rgb labels:
    #label_A_dir = os.path.join(root, mode, 'label1_rgb')
    #label_B_dir = os.path.join(root, mode, 'label2_rgb')

    data_list = os.listdir(img_A_dir)
    imgs_list_A, imgs_list_B, labels_A, labels_B = [], [], [], []
    count = 0
    for it in data_list:
        # print(it)
        if (it[-4:]=='.tif'):
            img_A_path = os.path.join(img_A_dir, it)
            img_B_path = os.path.join(img_B_dir, it)
            label_A_path = os.path.join(label_A_dir, it.replace('.tif','.png'))
            label_B_path = os.path.join(label_B_dir, it.replace('.tif','.png'))

            imgs_list_A.append(img_A_path)
            imgs_list_B.append(img_B_path)

            label_A = io.imread(label_A_path)
            label_B = io.imread(label_B_path)
            #for rgb labels:
            # label_A = Color2Index(label_A)
            # label_B = Color2Index(label_B)
            labels_A.append(label_A)
            labels_B.append(label_B)
        count+=1
        if not count%500: print('%d/%d images loaded.'%(count, len(data_list)))
    print(labels_A[0].shape)
    print(str(len(imgs_list_A)) + ' ' + mode + ' images' + ' loaded.')

    return imgs_list_A, imgs_list_B, labels_A, labels_B

def photometric_augment(img, p=0.8):
    """
    img: float32 HWC image, typically in the 0-255 range before normalization.
    """
    if random.random() > p:
        return img
    out = img.copy()
    # 1) brightness/contrast
    if random.random() < 0.7:
        alpha = random.uniform(0.9, 1.1)     # contrast
        beta  = random.uniform(-8, 8)      # brightness
        out = out * alpha + beta
    # 2) gamma
    if random.random() < 0.3:
        gamma = random.uniform(0.95, 1.05)
        out = np.clip(out, 0, 255) / 255.0
        out = (out ** gamma) * 255.0
    if random.random() < 0.3:
        ch_scale = np.array([
            random.uniform(0.95, 1.01),
            random.uniform(0.95, 1.01),
            random.uniform(0.95, 1.01),
        ], dtype=np.float32)
        out = out * ch_scale
    # 4) Gaussian noise
    if random.random() < 0.15:
        sigma = random.uniform(1.0, 3.0)
        noise = np.random.normal(0, sigma, out.shape).astype(np.float32)
        out = out + noise
    # 5) blur
    if random.random() < 0.1:
        k = random.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0)
    return np.clip(out, 0, 255).astype(np.float32)

def change_aware_crop(img_A, img_B, label_A, label_B, crop_h, crop_w, num_trials=10):
    h, w = img_A.shape[:2]

    if h == crop_h and w == crop_w:
        return img_A, img_B, label_A, label_B
    change_mask = (label_A != label_B).astype(np.uint8)
    best_h = 0
    best_w = 0
    best_score = -1.0
    for _ in range(num_trials):
        sh = random.randint(0, h - crop_h)
        sw = random.randint(0, w - crop_w)
        crop_change = change_mask[sh:sh + crop_h, sw:sw + crop_w]
        score = crop_change.mean()
        if score > best_score:
            best_score = score
            best_h = sh
            best_w = sw

    img_A = img_A[best_h:best_h + crop_h, best_w:best_w + crop_w]
    img_B = img_B[best_h:best_h + crop_h, best_w:best_w + crop_w]
    label_A = label_A[best_h:best_h + crop_h, best_w:best_w + crop_w]
    label_B = label_B[best_h:best_h + crop_h, best_w:best_w + crop_w]

    return img_A, img_B, label_A, label_B

def joint_augment(img_A, img_B, label_A, label_B, p=0.5):
    h, w = img_A.shape[:2]

    if random.random() < p:
        scale = random.uniform(0.9, 1.1)
        new_h, new_w = int(h * scale), int(w * scale)

        img_A = cv2.resize(img_A, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        img_B = cv2.resize(img_B, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        label_A = cv2.resize(label_A, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        label_B = cv2.resize(label_B, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        if scale > 1.0:

            img_A, img_B, label_A, label_B = change_aware_crop(
                img_A, img_B, label_A, label_B, h, w, num_trials=10
            )
        else:

            pad_h = max(0, h - new_h)
            pad_w = max(0, w - new_w)

            img_A = cv2.copyMakeBorder(img_A, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            img_B = cv2.copyMakeBorder(img_B, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

            label_A = cv2.copyMakeBorder(label_A, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
            label_B = cv2.copyMakeBorder(label_B, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)

            img_A = img_A[:h, :w]
            img_B = img_B[:h, :w]
            label_A = label_A[:h, :w]
            label_B = label_B[:h, :w]

    if random.random() < 0.5:
        img_A = np.fliplr(img_A)
        img_B = np.fliplr(img_B)
        label_A = np.fliplr(label_A)
        label_B = np.fliplr(label_B)

    k = random.randint(0, 3)
    if k > 0:
        img_A = np.rot90(img_A, k)
        img_B = np.rot90(img_B, k)
        label_A = np.rot90(label_A, k)
        label_B = np.rot90(label_B, k)

    swapped = False

    img_A = photometric_augment(img_A, p=0.6)
    img_B = photometric_augment(img_B, p=0.6)

    return img_A.copy(), img_B.copy(), label_A.copy(), label_B.copy(), swapped

class Data(data.Dataset):
    def __init__(self, mode, augment=False):
        self.augment = augment
        self.imgs_list_A, self.imgs_list_B, self.labels_A, self.labels_B = read_RSimages(mode)

    def get_mask_name(self, idx):
        mask_name = os.path.split(self.imgs_list_A[idx])[-1]
        return mask_name

    def __getitem__(self, idx):
        img_A = io.imread(self.imgs_list_A[idx]).astype(np.float32)
        img_B = io.imread(self.imgs_list_B[idx]).astype(np.float32)
        label_A = self.labels_A[idx].copy()
        label_B = self.labels_B[idx].copy()

        if self.augment:
            img_A, img_B, label_A, label_B, swapped = joint_augment(img_A, img_B, label_A, label_B)
        else:
            swapped = False

        if swapped:
            img_A = normalize_image(img_A, 'B')
            img_B = normalize_image(img_B, 'A')
        else:
            img_A = normalize_image(img_A, 'A')
            img_B = normalize_image(img_B, 'B')

        label_A = label_A.astype(np.int64)
        label_B = label_B.astype(np.int64)
        return F.to_tensor(img_A).float(), F.to_tensor(img_B).float(), torch.from_numpy(label_A).long(), torch.from_numpy(label_B).long()

    def __len__(self):
        return len(self.imgs_list_A)

class Data_test(data.Dataset):
    def __init__(self, test_dir):
        self.imgs_list_A = []
        self.imgs_list_B = []
        self.mask_name_list = []
        imgA_dir = os.path.join(test_dir, 'im1')
        imgB_dir = os.path.join(test_dir, 'im2')
        data_list = os.listdir(imgA_dir)
        for it in data_list:
            if (it[-4:]=='.tif'):
                self.imgs_list_A.append(os.path.join(imgA_dir, it))
                self.imgs_list_B.append(os.path.join(imgB_dir, it))
                self.mask_name_list.append(it)
        self.len = len(self.imgs_list_A)

    def get_mask_name(self, idx):
        return self.mask_name_list[idx]

    def __getitem__(self, idx):
        img_A = io.imread(self.imgs_list_A[idx]).astype(np.float32)
        img_B = io.imread(self.imgs_list_B[idx]).astype(np.float32)
        if img_A.max() > 255 or img_B.max() > 255:
            img_A = (img_A - img_A.min()) / (img_A.max() - img_A.min() + 1e-8) * 255.0
            img_B = (img_B - img_B.min()) / (img_B.max() - img_B.min() + 1e-8) * 255.0
        img_A = normalize_image(img_A, 'A')
        img_B = normalize_image(img_B, 'B')
        return F.to_tensor(img_A).float(), F.to_tensor(img_B).float()

    def __len__(self):
        return self.len
