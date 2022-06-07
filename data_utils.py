import os
import math
import random

import torch
import numpy as np
from torch import nn
from PIL import Image
from tqdm import tqdm
from torch.utils.data.sampler import Sampler
from torchvision import transforms, datasets
from torchvision.transforms import functional as F
import torch.utils.data as data

# channel means and standard deviations of kaggle dataset, computed by origin author
MEAN = [108.64628601 / 255, 75.86886597 / 255, 54.34005737 / 255]
STD = [70.53946096 / 255, 51.71475228 / 255, 43.03428563 / 255]

# for color augmentation, computed by origin author
U = torch.tensor([[-0.56543481, 0.71983482, 0.40240142],
                  [-0.5989477, -0.02304967, -0.80036049],
                  [-0.56694071, -0.6935729, 0.44423429]], dtype=torch.float32)
EV = torch.tensor([1.65513492, 0.48450358, 0.1565086], dtype=torch.float32)

# set of resampling weights that yields balanced classes, computed by origin author
BALANCE_WEIGHTS = torch.tensor([1.3609453700116234, 14.378223495702006,
                                6.637566137566138, 40.235967926689575,
                                49.612994350282484], dtype=torch.double)
FINAL_WEIGHTS = torch.as_tensor([1, 2, 2, 2, 2], dtype=torch.double)


def generate_stem_dataset(data_path, input_size, data_aug):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            size=input_size,
            scale=data_aug['scale'],
            ratio=data_aug['stretch_ratio']
        ),
        transforms.RandomAffine(
            degrees=data_aug['ratation'],
            translate=data_aug['translation_ratio'],
            scale=None,
            shear=None
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(tuple(MEAN), tuple(STD)),
        KrizhevskyColorAugmentation(sigma=data_aug['sigma'])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        transforms.Normalize(tuple(MEAN), tuple(STD))
    ])

    def load_image(x):
        return Image.open(x)

    return generate_dataset(data_path, load_image, ('jpg','png', 'jpeg'), train_transform, test_transform)


# def my_collate(batch):
#   batch = filter(lambda img: img is not None, batch)
  
#   return data.dataloader.default_collate(list(batch))

def my_collate(examples):
    pixel_values = torch.stack([example[0] for example in examples])
    labels = torch.tensor([example[1] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def generate_dataset(data_path, loader, extensions, train_transform, test_transform):
    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')
    val_path = os.path.join(data_path, 'val')

    train_dataset = datasets.DatasetFolder(train_path, loader, extensions, transform=train_transform)
    test_dataset = datasets.DatasetFolder(test_path, loader, extensions, transform=test_transform)
    val_dataset = datasets.DatasetFolder(val_path, loader, extensions, transform=test_transform)

    return train_dataset, test_dataset, val_dataset


class ScheduledWeightedSampler(Sampler):
    def __init__(self, num_samples, train_targets, initial_weight=BALANCE_WEIGHTS,
                 final_weight=FINAL_WEIGHTS, replacement=True):
        self.num_samples = num_samples
        self.train_targets = train_targets
        self.replacement = replacement

        self.epoch = 0
        self.w0 = initial_weight
        self.wf = final_weight
        self.train_sample_weight = torch.zeros(len(train_targets), dtype=torch.double)
        self.assign_weight(initial_weight)

    def step(self):
        self.epoch += 1
        factor = 0.975**(self.epoch - 1)
        self.weights = factor * self.w0 + (1 - factor) * self.wf
        self.assign_weight(self.weights)

    def assign_weight(self, weights):
        for i, _class in enumerate(self.train_targets):
            self.train_sample_weight[i] = weights[_class]

    def __iter__(self):
        return iter(torch.multinomial(self.train_sample_weight, self.num_samples, self.replacement).tolist())

    def __len__(self):
        return self.num_samples


class KrizhevskyColorAugmentation(object):
    def __init__(self, sigma=0.5):
        self.sigma = sigma
        self.mean = torch.tensor([0.0])
        self.deviation = torch.tensor([sigma])

    def __call__(self, img, color_vec=None):
        sigma = self.sigma
        if color_vec is None:
            if not sigma > 0.0:
                color_vec = torch.zeros(3, dtype=torch.float32)
            else:
                color_vec = torch.distributions.Normal(self.mean, self.deviation).sample((3,))
            color_vec = color_vec.squeeze()

        alpha = color_vec * EV
        noise = torch.matmul(U, alpha.t())
        noise = noise.view((3, 1, 1))
        return img + noise

    def __repr__(self):
        return self.__class__.__name__ + '(sigma={})'.format(self.sigma)


