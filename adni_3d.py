import os
import pathlib
import torch
import numpy as np
import nibabel as nib
from skimage.transform import resize
from torch.utils.data.dataset import Dataset
from typing import Tuple
from loguru import logger

from utils import find_classes


class ADNIDataset3D(Dataset):
    def __init__(
        self, root="data\\ADNI1\\ADNI1-Screening-Nifti", domain=None, transform=None
    ) -> None:
        logger.info("Loading Image and Label data from ADNI1".center(70, "+"))
        self.data_path = os.path.join(os.getcwd(), root, domain)
        self.paths = list(pathlib.Path(self.data_path).glob("*/*.nii.gz"))

        logger.info(f"Loading Label Data".center(70, "+"))
        self.transform = transform
        self.classes, self.class_to_idx = find_classes(self.data_path)

    def load_image(self, index: int):
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        image = nib.load(image_path)
        image = np.array(image.get_fdata()[:, :]).squeeze().astype(np.float32)
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        return resize(image, (224, 218, 224), mode="constant")

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        class_name = self.paths[
            index
        ].parent.name  # expects path in data_folder/class_name/image.jpeg
        class_idx = self.class_to_idx[class_name]

        img = customToTensor(img)

        return [img, class_idx]

    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)


def customToTensor(pic):
    if isinstance(pic, np.ndarray):
        img = torch.from_numpy(pic)
        img = torch.unsqueeze(img, 0)
        return img.float()
