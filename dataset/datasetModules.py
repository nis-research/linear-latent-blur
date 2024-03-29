from typing import Optional
import pandas as pd
import cv2
import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from PIL import Image
import sys
sys.path.append("/gpfs/home3/imazilu/defocus_blur/")
from models.config import INPUT_CROP_SIZE, MEAN_W1, STD_W1, MEAN_W2, STD_W2, STACKS, DEBLUR_INPUTS_FILE
from variables import INPUT_PATH


def get_stats(slide_type):
    return (MEAN_W1, STD_W1) if "w1" in slide_type else (MEAN_W2, STD_W2)


class CustomDataset:

    def __init__(self, data_csv, training_dir="train", use_ten_crop=True, return_alpha=False,
                 compute_stats=False, normalize=True, input_path=""):
        self.dir_path = os.path.dirname(os.path.abspath(__file__))
        if input_path == "":
            self.train_dir = os.path.join(self.dir_path, training_dir, "processed")
        else:
            self.train_dir = os.path.join(input_path, "dataset", training_dir, "processed")
        self.train_df = pd.read_csv(os.path.join(self.dir_path, data_csv), header=0)
        self.compute_stats = compute_stats
        self.use_ten_crop = use_ten_crop
        self.normalize = normalize
        self._set_transformations()
        self.return_alpha = return_alpha
        self.mean, self.std = get_stats(data_csv)

    def __getitem__(self, index):

        # Get the image path
        input1_path = os.path.join(self.train_dir, self.train_df.iat[index, 1].replace("\\", "/"))
        input2_path = os.path.join(self.train_dir, self.train_df.iat[index, 2].replace("\\", "/"))
        target_path = os.path.join(self.train_dir, self.train_df.iat[index, 3].replace("\\", "/"))

        # Reading and processing the image
        input1 = self._process_image(input1_path)
        input2 = self._process_image(input2_path)
        target = self._process_image(target_path)

        # Apply image transformations
        if self.transform is not None:
            input1 = self.stack_crops(self.transform(input1))
            input2 = self.stack_crops(self.transform(input2))
            target = self.stack_crops(self.transform(target))
        # the last value returned is the interpolation parameter
        if not self.return_alpha:
            return input1, input2, target
        return input1, input2, target, self.train_df.iat[index, 4]

    def __len__(self):
        return len(self.train_df)

    def stack_crops(self, crops):
        if not self.use_ten_crop or self.compute_stats:
            return crops
        if self.normalize:
            return torch.stack([transforms.Normalize(self.mean, self.std)(transforms.ToTensor()(crop)) for crop in crops])
        return torch.stack([transforms.ToTensor()(crop) for crop in crops])

    def _set_transformations(self):
        if self.compute_stats:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
        elif self.use_ten_crop:
            self.transform = transforms.Compose([
                transforms.TenCrop(INPUT_CROP_SIZE),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop(INPUT_CROP_SIZE),
            ])

    def _process_image(self, image_path):
        """
        Reads the image at the given path and returns it.
        """
        image = cv2.imread(image_path, -1)
        try:
            return Image.fromarray(image)
        except AttributeError:
            print(image_path)
            raise


# class PretrainDataset:
#
#     def __init__(self, data_csv="pretrain_train_all.csv", training_dir="train", use_ten_crop=True):
#         self.dir_path = os.path.dirname(os.path.abspath(__file__))
#         self.train_df = pd.read_csv(os.path.join(self.dir_path, data_csv), header=0)
#         self.train_dir = os.path.join(self.dir_path, training_dir, "processed")
#         self.use_ten_crop = use_ten_crop
#         self._set_transformations()
#         self.mean, self.std = get_stats(data_csv)
#
#     def __getitem__(self, index):
#         # Get the image path
#         image_path = os.path.join(self.train_dir, self.train_df.iat[index, 1].replace("\\", "/"))
#         # Reading and processing the image
#         image = self._process_image(image_path)
#         # Apply image transformations
#         if self.transform is not None:
#             image = self.stack_crops(self.transform(image))
#         return image
#
#     def __len__(self):
#         return len(self.train_df)
#
#     def stack_crops(self, crops):
#         return torch.stack([transforms.Normalize(self.mean, self.std)(transforms.ToTensor()(crop)) for crop in crops])
#
#     def _set_transformations(self):
#         if self.use_ten_crop:
#             self.transform = transforms.Compose([
#                 transforms.TenCrop(INPUT_CROP_SIZE),
#             ])
#         else:
#             self.transform = transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.CenterCrop(INPUT_CROP_SIZE),
#             ])
#
#     def _process_image(self, image_path):
#         """
#         Reads the image at the given path and returns it.
#         """
#         image = cv2.imread(image_path, -1)
#         return Image.fromarray(image)


# class PretrainDataModule(pl.LightningDataModule):
#
#     def __init__(self, batch_size, slide_type):
#         super().__init__()
#         self.batch_size = batch_size
#         self.slide_type = slide_type
#
#     def setup(self, stage: Optional[str] = None):
#         self.train_ds = PretrainDataset(data_csv=f"pretrain_train_{self.slide_type}.csv", training_dir="train")
#         self.val_ds = PretrainDataset(data_csv=f"pretrain_val_{self.slide_type}.csv", training_dir="val")
#         self.test_ds = PretrainDataset(data_csv=f"pretrain_test_{self.slide_type}.csv", training_dir="test")
#
#     def prepare_data(self):
#         pass
#
#     def train_dataloader(self):
#         return DataLoader(self.train_ds, shuffle=True, batch_size=self.batch_size, num_workers=1,
#                           pin_memory=True)
#
#     def val_dataloader(self):
#         return DataLoader(self.val_ds, shuffle=True, batch_size=self.batch_size, num_workers=1)
#
#     def test_dataloader(self):
#         return DataLoader(self.test_ds, shuffle=False, batch_size=self.batch_size, num_workers=1)


class TrainDataModule(pl.LightningDataModule):

    def __init__(self, batch_size, slide_type: str, normalize=True, use_ten_crop=True, input_path=""):
        super().__init__()
        self.batch_size = batch_size
        self.slide_type = slide_type
        self.normalize = normalize
        self.use_ten_crop = use_ten_crop
        self.input_path = input_path

    def setup(self, stage: Optional[str] = None):
        self.train_ds = CustomDataset(data_csv=f"train_{self.slide_type}.csv", normalize=self.normalize,
                                      use_ten_crop=self.use_ten_crop, input_path=self.input_path)
        self.val_ds = CustomDataset(data_csv=f"val_{self.slide_type}.csv", training_dir="val", normalize=self.normalize,
                                      use_ten_crop=self.use_ten_crop, input_path=self.input_path)
        # by default, we only use a center crop from the test set
        self.test_ds = CustomDataset(data_csv=f"test_{self.slide_type}.csv", training_dir="test",
                                     normalize=self.normalize, use_ten_crop=False, input_path=self.input_path)


    def prepare_data(self):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True, batch_size=self.batch_size, num_workers=8,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, shuffle=False, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_ds, shuffle=False, batch_size=1, num_workers=8)


def input_for_visualization(slide_type, input_path="", normalize=True):
    """
    Returns sets of 9 images, where each set contains a transition from z0 to z16 for one of the images in the
    `images` list.
    :param slide_type: which type of slides to be used ("w1" or "w2")
    :return: a tensor with sets of 9 images each
    """
    if normalize:
        mean_set, std_set = get_stats(slide_type)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean_set, std_set),
            transforms.CenterCrop(INPUT_CROP_SIZE),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(INPUT_CROP_SIZE),
        ])
    images = [f"mcf-z-stacks-03212011_a04_s1_{slide_type}.tif", f"mcf-z-stacks-03212011_a07_s1_{slide_type}.tif",
              f"mcf-z-stacks-03212011_a10_s1_{slide_type}.tif"]
    dir_path = os.path.dirname(os.path.abspath(__file__))
    z_stacks = []
    for z in STACKS:
        if input_path == "":
            dir_name = os.path.join(dir_path, "test", "processed", f"z{z}")
        else:
            dir_name = os.path.join(input_path, "dataset", "test", "processed", f"z{z}")
        for img in images:
            img_path = os.path.join(dir_name, img)
            if not os.path.exists(img_path):
                print(f"path {img_path} does not exist")
                print(sorted(os.listdir(dir_name))[0])
            img = cv2.imread(img_path, -1)
            img = transform(img)
            try:
                z_stacks[z//2].append(img)
            except IndexError:
                z_stacks.append([img])
    for idx, zs in enumerate(z_stacks):
        z_stacks[idx] = torch.stack(zs, dim=0)
    return z_stacks


def test_images(slide_type, normalize=False, input_path=""):
    """
    Returns a tensor which contains subsets of 9 images for each image in the test set (transition from z0 to z16)
    :param slide_type: the type of slides to be used
    :return: a tensor with images
    """
    if normalize:
        mean_set, std_set = get_stats(slide_type)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean_set, std_set),
            transforms.CenterCrop(INPUT_CROP_SIZE),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(INPUT_CROP_SIZE),
        ])
    dir_path = os.path.dirname(os.path.abspath(__file__))  # in our case it is /path_to_project/dataset
    result, img_class, stacks = [], [], []
    if input_path == "":
        dir_name = os.path.join(dir_path, "test", "processed", f"z{0}")
    else:
        dir_name = os.path.join(input_path, "dataset", "test", "processed", f"z{0}")
    for idx, img_name in enumerate(sorted(os.listdir(dir_name))):
        if slide_type in img_name:
            for z in STACKS:
                if input_path == "":
                    stack_dir = os.path.join(dir_path, "test", "processed", f"z{z}")
                else:
                    # in our setup, the data was located at /input_path/dataset/...
                    stack_dir = os.path.join(input_path, "dataset", "test", "processed", f"z{z}")
                img_path = os.path.join(stack_dir, img_name)
                img = cv2.imread(img_path, -1)
                img = transform(img)
                result.append(img)
                img_class.append(idx)
                stacks.append(z)
    result_tensor = torch.stack(result, dim=0)
    return result_tensor, img_class, stacks


def compute_dataset_stats(slide_type):
    """
    Computes the mean and std for the train set of `slide_type` slides.
    :param slide_type: the type of slides ("w1" or "w2")
    :return: the mean and std of the set of images chosen
    """
    ds = CustomDataset(f"train_{slide_type}.csv", compute_stats=True)
    idx = 0
    imgs = []
    while True:
        try:
            i1, i2, i3 = ds.__getitem__(idx)
            idx += 1
            imgs.extend([i1, i2, i3])
        except IndexError:
            break
    imgs = torch.stack(imgs, dim=0)
    return torch.mean(imgs, dim=[0, 1, 2, 3]), torch.std(imgs, dim=[0, 1, 2, 3])


def merge_batch_crops(images):
    bs, nrcrops, c, h, w = images.size()
    return images.view(-1, c, h, w)


def input_for_deblurring(slide_type, use_ten_crop=False, input_path="", img_idx=-1):
    """
    Returns the left and right inputs for a deblurring operation, alongside the target sharp image and the corresponding
    interpolation parameter alpha.
    :param slide_type: the type of slides ("w1" or "w2").
    :param use_ten_crop: whether to generate 10 crops from each image, or only a center crop.
    :return: tensors representing the left and right inputs for deblurring, the target sharp images and a list of alpha
    parameters.
    """
    ds = CustomDataset(data_csv=f"{DEBLUR_INPUTS_FILE}_{slide_type}.csv", training_dir="test",
                       use_ten_crop=use_ten_crop, return_alpha=True, input_path=input_path)
    i = 0
    left_imgs, right_imgs, target_imgs, alphas = [], [], [], []
    while True:
        try:
            print(f"Trying to read image {i}")
            if i == img_idx or img_idx == -1:
                left, right, target, alpha = ds.__getitem__(i)
                if use_ten_crop:
                    alphas.extend([alpha] * left.shape[0])
                else:
                    alphas.append(alpha)
                left_imgs.append(left)
                right_imgs.append(right)
                target_imgs.append(target)
                if i == img_idx:
                    break
            i += 1
        except IndexError:
            break
    if not use_ten_crop:
        return torch.stack(left_imgs, dim=0), torch.stack(right_imgs, dim=0), torch.stack(target_imgs, dim=0), alphas
    return torch.cat(left_imgs, dim=0), torch.cat(right_imgs, dim=0), torch.cat(target_imgs, dim=0), alphas



