"""
Contains functionality for creating PyTorch DataLoaders
"""

import os

from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from loguru import logger

from adni_2d import ADNIDataset2D
from adni_3d import ADNIDataset3D

NUM_WORKERS = os.cpu_count()
SEED = 42


def create_dataloaders_mri_2d(
    root: str,
    source_1: str,
    source_2: str,
    source_3: str,
    transform_train: transforms.Compose,
    transform_test_val: transforms.Compose,
    batch_size_train: int,
    batch_size_test_val: int,
    pin_memoery: bool = True,
    gen_test_val: bool = False,
    num_workers: int = NUM_WORKERS,
):

    data_output = {}
    for key, data in {
        "Source_1": source_1,
        "Source_2": source_2,
        "Source_3": source_3,
    }.items():
        # Split the Source Data into Train and Val
        train_source_split = ADNIDataset2D(
            root=root, domain=data, split='train', transform=transform_train
        )
        test_source_split = ADNIDataset2D(
            root=root, domain=data, split='test', transform=transform_test_val
        )

        # Get class names
        class_names = train_source_split.classes

        if gen_test_val:
            val_source_split = ADNIDataset2D(
            root=root, domain=data, split='val', transform=transform_test_val
        )
            #logger.info(
            #    f"The length of {key} Val set is: {len(val_source_split)}".center(
            #        70, "+"
            #    )
            #)
            val_dataloader = DataLoader(
                val_source_split,
                batch_size=batch_size_test_val,
                shuffle=False,
                num_workers=1,
                pin_memory=False,
            )

        # Turn images into data loaders
        #logger.info(
        #    f"The length of {key} Train set is: {len(train_source_split)}".center(
        #        70, "+"
        #    )
        #)
        train_dataloader = DataLoader(
            train_source_split,
            batch_size=batch_size_train,
            shuffle=True,
            num_workers=1,
            pin_memory=pin_memoery,
        )

        #logger.info(
        #    f"The length of {key} Test set is: {len(test_source_split)}".center(70, "+")
        #)
        test_dataloader = DataLoader(
            test_source_split,
            batch_size=batch_size_test_val,
            shuffle=False,
            num_workers=1,
            pin_memory=False,
        )

        if gen_test_val:
            data_output[key] = {
                "Train": train_dataloader,
                "Val": val_dataloader,
                "Test": test_dataloader,
            }
        else:
            data_output[key] = {
                "Train": train_dataloader,
                "Test": test_dataloader,
            }

    return data_output, class_names


def create_dataloaders_mri_3d(
    root: str,
    source_1: str,
    transform_train: transforms.Compose,
    transform_test_val: transforms.Compose,
    batch_size_train: int,
    batch_size_test_val: int,
    pin_memoery: bool = True,
    gen_test_val: bool = False,
    num_workers: int = NUM_WORKERS,
):
    data_output = {}
    for key, data in {"Source_1": source_1}.items():
        # Split the Source Data into Train and Val
        train_source_split = ADNIDataset3D(
            root=root, domain=data, split='train', transform=transform_train
        )
        test_source_split = ADNIDataset3D(
            root=root, domain=data, split='test', transform=transform_test_val
        )

        # Get class names
        class_names = train_source_split.classes

        if gen_test_val:
            val_source_split = ADNIDataset3D(
            root=root, domain=data, split='val', transform=transform_test_val
        )
            #logger.info(
            #    f"The length of {key} Val set is: {len(val_source_split)}".center(
            #        70, "+"
            #    )
            #)
            val_dataloader = DataLoader(
                val_source_split,
                batch_size=batch_size_test_val,
                shuffle=False,
                num_workers=1,
                pin_memory=False,
            )

        # Turn images into data loaders
        #logger.info(
        #    f"The length of {key} Train set is: {len(train_source_split)}".center(
        #        70, "+"
        #    )
        #)
        train_dataloader = DataLoader(
            train_source_split,
            batch_size=batch_size_train,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memoery,
        )

        #logger.info(
        #    f"The length of {key} Test set is: {len(test_source_split)}".center(70, "+")
        #)
        test_dataloader = DataLoader(
            test_source_split,
            batch_size=batch_size_test_val,
            shuffle=False,
            num_workers=1,
            pin_memory=False,
        )

        if gen_test_val:
            data_output[key] = {
                "Train": train_dataloader,
                "Val": val_dataloader,
                "Test": test_dataloader,
            }
        else:
            data_output[key] = {
                "Train": train_dataloader,
                "Test": test_dataloader,
            }

    return data_output, class_names
