from os.path import isdir, join
from glob import glob
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.utils import encode_img


DEFAULT_TARGET_SIZE = (64, 64)


class ImageDataset:

    def __init__(self, root, target_size=DEFAULT_TARGET_SIZE, transform=None, hint=""):
        assert isdir(root)
        self.root = root
        self.output_size = target_size
        self.sample_ptr = 0
        self.transform = transform
        self.img_paths = glob(join(root, "*.jpg"))
        self.imgs = list(map(self.make_tensor, self.img_paths))
        self.dataset_hint = hint

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return [self[i] for i in range(*item.indices(len(self)))]
        elif isinstance(item, int):
            if self.transform:
                return self.transform(self.imgs[item])
            else:
                return self.imgs[item]
        else:
            raise ValueError("Invalid index(-ices) to __get_item__ method")

    def __iter__(self):
        return self

    def __next__(self):
        if self.sample_ptr < len(self.imgs):
            out = self.sample_ptr
            self.sample_ptr += 1
            return self.__getitem__(out)
        else:
            self.sample_ptr = 0
            raise StopIteration

    def __str__(self):
        return f"ImageDataset{self.dataset_hint}"

    def make_tensor(self, path_to_image):
        image = cv2.imread(path_to_image)
        resize_image = (image.shape[1], image.shape[0]) != self.output_size
        if resize_image:
            image = cv2.resize(image, self.output_size)
        return encode_img(image)


class AddGaussianNoise(object):

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        if type(tensor) == torch.Tensor:
            return tensor + torch.randn(tensor.size(), device=tensor.device) * self.std + self.mean
        elif type(tensor) == np.ndarray:
            return tensor + np.random.normal(loc=self.mean, scale=self.std, size=tensor.shape).astype(dtype=np.float32)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def make_dataloader(dset, batch_size=1, shuffle_dataset=True):
    random_seed = 42

    dataset_size = len(dset)
    indices = list(range(dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    dataloader = torch.utils.data.DataLoader(dset, batch_size=batch_size)
    return dataloader
