import os
import numpy as np
import torch
from skimage import io
from torch.utils import data
import utils.transform as transform
import matplotlib.pyplot as plt
from skimage.transform import rescale
from torchvision.transforms import functional as F
import cv2
import torchvision.transforms as T
import random
import albumentations as A
from PIL import Image

num_classes = 7
ST_COLORMAP = [[255, 255, 255], [0, 0, 255], [128, 128, 128], [0, 128, 0], [0, 255, 0], [128, 0, 0], [255, 0, 0]]
ST_CLASSES = ['unchanged', 'water', 'ground', 'low vegetation', 'tree', 'building', 'sports field']

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
    im = im.astype(np.float32)
    if time=='A':
        im = (im - MEAN_A) / STD_A
    else:
        im = (im - MEAN_B) / STD_B
    return im

def normalize_images(imgs, time='A'):
    for i, im in enumerate(imgs):
        imgs[i] = normalize_image(im, time)
    return imgs

def read_RSimages(mode, rescale=False):
    img_A_dir = os.path.join(root, mode, 'im1')
    img_B_dir = os.path.join(root, mode, 'im2')
    label_A_dir = os.path.join(root, mode, 'label1_idx')
    label_B_dir = os.path.join(root, mode, 'label2_idx')

    data_list = os.listdir(img_A_dir)
    imgs_list_A, imgs_list_B, labels_A, labels_B = [], [], [], []
    count = 0
    for it in data_list:
        if (it[-4:]=='.png'):
            img_A_path = os.path.join(img_A_dir, it)
            img_B_path = os.path.join(img_B_dir, it)
            label_A_path = os.path.join(label_A_dir, it.replace('.tif','.png'))
            label_B_path = os.path.join(label_B_dir, it.replace('.tif','.png'))

            imgs_list_A.append(img_A_path)
            imgs_list_B.append(img_B_path)

            label_A = io.imread(label_A_path)
            label_B = io.imread(label_B_path)

            labels_A.append(label_A)
            labels_B.append(label_B)
        count+=1
        if not count%500: print('%d/%d images loaded.'%(count, len(data_list)))

    print(labels_A[0].shape)
    print(str(len(imgs_list_A)) + ' ' + mode + ' images' + ' loaded.')

    return imgs_list_A, imgs_list_B, labels_A, labels_B

class Data(data.Dataset):
    def __init__(self, mode, random_flip=False):
        self.mode = mode
        self.random_flip = random_flip
        self.imgs_list_A, self.imgs_list_B, self.labels_A, self.labels_B = read_RSimages(mode)

        self.crop_size = 512

        # Apply the same geometric transform to both images and both masks.
        self.geo_transform = A.Compose([

            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),

            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=6, p=0.5),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
                A.OpticalDistortion(distort_limit=0.2, p=0.5),
            ], p=0.3),

            A.Resize(height=self.crop_size, width=self.crop_size, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
        ],

        additional_targets={'image_b': 'image', 'mask_b': 'mask'})

        self.photo_transform = A.Compose([

            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),

            A.GaussNoise(std_range=(0.02, 0.08),  mean_range=(0.0, 0.0), p=0.3),

            A.OneOf([
                A.MotionBlur(blur_limit=3, p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.1),

            A.ToGray(p=0.1),

        ])

        self.shared_color_aug = A.ColorJitter(
            brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05, p=1
        )

    def get_mask_name(self, idx):
        mask_name = os.path.split(self.imgs_list_A[idx])[-1]
        return mask_name

    def __getitem__(self, idx):

        img_A = io.imread(self.imgs_list_A[idx])
        img_B = io.imread(self.imgs_list_B[idx])
        label_A = self.labels_A[idx]
        label_B = self.labels_B[idx]

        if self.random_flip:

            transformed = self.geo_transform(image=img_A, image_b=img_B, mask=label_A, mask_b=label_B)
            img_A = transformed['image']
            img_B = transformed['image_b']
            label_A = transformed['mask']
            label_B = transformed['mask_b']

            # Sample photometric augmentation independently for each timestamp.
            img_A = self.photo_transform(image=img_A)['image']
            img_B = self.photo_transform(image=img_B)['image']

        elif self.mode != 'train':

             resizer = A.Resize(height=self.crop_size, width=self.crop_size)
             transformed = resizer(image=img_A, image_b=img_B, mask=label_A, mask_b=label_B)
             img_A, img_B = transformed['image'], transformed['image_b']
             label_A, label_B = transformed['mask'], transformed['mask_b']

        # Use timestamp-specific normalization statistics.
        img_A = normalize_image(img_A, 'A')
        img_B = normalize_image(img_B, 'B')

        return F.to_tensor(img_A).float(), F.to_tensor(img_B).float(), torch.from_numpy(label_A).long(), torch.from_numpy(label_B).long()

    def __len__(self):
        return len(self.imgs_list_A)

class Data_test(data.Dataset):
    def __init__(self, test_dir):
        self.imgs_A = []
        self.imgs_B = []
        self.mask_name_list = []
        imgA_dir = os.path.join(test_dir, 'im1')
        imgB_dir = os.path.join(test_dir, 'im2')
        data_list = os.listdir(imgA_dir)
        for it in data_list:
            if (it[-4:]=='.png'):
                img_A_path = os.path.join(imgA_dir, it)
                img_B_path = os.path.join(imgB_dir, it)
                self.imgs_A.append(io.imread(img_A_path))
                self.imgs_B.append(io.imread(img_B_path))
                self.mask_name_list.append(it)
        self.len = len(self.imgs_A)

    def get_mask_name(self, idx):
        return self.mask_name_list[idx]

    def __getitem__(self, idx):
        img_A = self.imgs_A[idx]
        img_B = self.imgs_B[idx]
        # Use timestamp-specific normalization statistics.
        img_A = normalize_image(img_A, 'A')
        img_B = normalize_image(img_B, 'B')
        return F.to_tensor(img_A), F.to_tensor(img_B)

    def __len__(self):
        return self.len