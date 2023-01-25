import os
import re
import sys
from pathlib import Path
from typing import Optional, Callable, Any

import torch
import numpy as np
from PIL import Image
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import warnings
from torchvision import transforms
from torchvision.datasets import VisionDataset

warnings.filterwarnings("ignore")

original_img_regex = ".*frame_[0-9]+_endo.png"

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve


def load_url(url, model_dir='./pretrained', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)


def find_images_in_directory(img_directory, img_regex):
    result = []
    for subdir, dirs, files in os.walk(img_directory):
        for file in files:
            if re.match(img_regex, file):
                result.append("/".join(subdir.split("\\")[1:] + [file]).replace(".png", ""))
    return result


def load_image(filename):
    return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = r'{}'.format(str(mask_dir) + "/" + str(idx) + mask_suffix + ".png")
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = '_watershed_mask'):
        self.images_dir = images_dir
        self.mask_dir = mask_dir
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.ids = find_images_in_directory("archive", original_img_regex)
        print(f'Creating dataset with {len(self.ids)} examples')

        self.mask_values = [
            [50, 50, 50],
            [11, 11, 11],
            [21, 21, 21],
            [13, 13, 13],
            [12, 12, 12],
            [31, 31, 31],
            [23, 23, 23],
            [24, 24, 24],
            [25, 25, 25],
            [32, 32, 32],
            [22, 22, 22],
            [33, 33, 33],
            [5, 5, 5]
        ]

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = f"{self.mask_dir}/{name}{self.mask_suffix}.png"
        img_file = f"{self.mask_dir}/{name}.png"
        mask = load_image(mask_file)
        img = load_image(img_file)

        # img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        # mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)
        return torch.as_tensor(np.asarray(img).copy()).float().contiguous(), torch.as_tensor(
            np.asarray(mask).copy()).long().contiguous()


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)
    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)
    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def count_dice_coeff(val_loader, net):
    dice_score = 0
    for i, data in enumerate(val_loader, 0):
        data, mask_true = data
        mask_pred = net(data)
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
        mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
        dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)


def count_dice_coeff_torchvision(val_loader, net):
    dice_score = 0
    for i, data in enumerate(val_loader, 0):
        data, mask_true = data
        mask_pred = net(data)['out']
        mask_true = F.one_hot(mask_true, 13).permute(0, 3, 1, 2).float()
        mask_pred = F.one_hot(mask_pred.argmax(dim=1), 13).permute(0, 3, 1, 2).float()
        dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)


class SegmentationDataset(VisionDataset):
    def __init__(self,
                 root: str,
                 image_folder: str = "archive",
                 mask_folder: str = "archive",
                 transforms: Optional[Callable] = None,
                 image_color_mode: str = "rgb",
                 mask_color_mode: str = "grayscale") -> None:

        super().__init__(root, transforms)
        self.image_names = find_images_in_directory(image_folder, original_img_regex)
        self.mask_names = [f"{x}_watershed_mask" for x in self.image_names]
        self.image_color_mode = image_color_mode
        self.mask_color_mode = mask_color_mode
        print("self.image_names[:10]", self.image_names[:10])

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int) -> Any:
        image_path = self.image_names[index]
        mask_path = self.mask_names[index]
        with open(f"archive/{image_path}.png", "rb") as image_file, open(f"archive/{mask_path}.png", "rb") as mask_file:
            image = Image.open(image_file)
            if self.image_color_mode == "rgb":
                image = image.convert("RGB")
            elif self.image_color_mode == "grayscale":
                image = image.convert("L")
            mask = Image.open(mask_file)
            if self.mask_color_mode == "rgb":
                mask = mask.convert("RGB")
            elif self.mask_color_mode == "grayscale":
                mask = mask.convert("L")
            sample = {"image": image, "mask": mask}
            if self.transforms:
                sample["image"] = self.transforms(sample["image"])
                sample["mask"] = self.transforms(sample["mask"])
            return sample


def get_dataloader_single_folder(data_dir: str,
                                 image_folder: str = 'archive',
                                 mask_folder: str = 'archive',
                                 fraction: float = 0.2,
                                 batch_size: int = 4):
    data_transforms = transforms.Compose([transforms.ToTensor()])

    image_datasets = {
        x: SegmentationDataset(data_dir,
                               image_folder=image_folder,
                               mask_folder=mask_folder,
                               transforms=data_transforms)
        for x in ['Train', 'Test']
    }
    dataloaders = {
        x: DataLoader(image_datasets[x],
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=8)
        for x in ['Train', 'Test']
    }
    return dataloaders
