"""
Contains functionality for creating PyTorch DataLoaders
"""

import os

from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from adni_2d import ADNIDataset2D
from adni_3d import ADNIDataset3D

NUM_WORKERS = os.cpu_count()
SEED = 42


def create_dataloaders_mri_2d(
    root: str,
    source_1: str,
    source_2: str,
    source_3: str,
    transform: transforms.Compose,
    batch_size: int,
    pin_memoery: bool = True,
    val_size: float = 0.3,
    test_size: float = 0.5,
    gen_val_test: bool = False,
    num_workers: int = NUM_WORKERS,
):

    # Use ImageFolder to create dataset(s)
    source_data_1 = ADNIDataset2D(root=root, domain=source_1, transform=transform)
    source_data_2 = ADNIDataset2D(root=root, domain=source_2, transform=transform)
    source_data_3 = ADNIDataset2D(root=root, domain=source_3, transform=transform)

    # Get class names
    class_names = source_data_1.classes

    data_output = {}
    for key, data in {
        "Source_1": source_data_1,
        "Source_2": source_data_2,
        "Source_3": source_data_3,
    }.items():
        # Split the Source Data into Train and Val
        train_idx, val_idx = train_test_split(
            list(range(len(data))), test_size=val_size, random_state=SEED
        )

        ## generate subset based on indices
        train_source_split = Subset(data, train_idx)
        val_source_split = Subset(data, val_idx)

        if gen_val_test:
            # Split the Source Data into Test and Val
            test_idx, val_idx = train_test_split(
                range(len(val_source_split)), test_size=test_size, random_state=SEED
            )
            ## generate subset based on indices
            test_source_split = Subset(val_source_split, test_idx)
            val_source_split = Subset(val_source_split, val_idx)

            test_dataloader = DataLoader(
                test_source_split,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memoery,
            )

        # Turn images into data loaders
        train_dataloader = DataLoader(
            train_source_split,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memoery,
        )

        val_dataloader = DataLoader(
            val_source_split,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memoery,
        )

        if gen_val_test:
            data_output[key] = {
                "Train": train_dataloader,
                "Val": val_dataloader,
                "Test": test_dataloader,
            }
        else:
            data_output[key] = {
                "Train": train_dataloader,
                "Test": val_dataloader,
            }

    return data_output, class_names


def create_dataloaders_mri_3d(
    root: str,
    source_1: str,
    transform: transforms.Compose,
    batch_size: int,
    pin_memoery: bool = True,
    val_size: float = 0.3,
    test_size: float = 0.15,
    gen_val_test: bool = False,
    num_workers: int = NUM_WORKERS,
):

    # Use ImageFolder to create dataset(s)
    source_data_1 = ADNIDataset3D(root=root, domain=source_1, transform=transform)

    # Get class names
    class_names = source_data_1.classes

    data_output = {}
    for key, data in {"Source_1": source_data_1}.items():
        # Split the Source Data into Train and Val
        train_idx, val_idx = train_test_split(
            list(range(len(data))), test_size=val_size, random_state=SEED
        )

        ## generate subset based on indices
        train_source_split = Subset(data, train_idx)
        val_source_split = Subset(data, val_idx)

        if gen_val_test:
            # Split the Source Data into Test and Val
            test_idx, val_idx = train_test_split(
                range(len(val_source_split)), test_size=test_size, random_state=SEED
            )
            ## generate subset based on indices
            test_source_split = Subset(val_source_split, test_idx)
            val_source_split = Subset(val_source_split, val_idx)

            test_dataloader = DataLoader(
                test_source_split,
                batch_size=1,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memoery,
            )

        # Turn images into data loaders
        train_dataloader = DataLoader(
            train_source_split,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memoery,
        )

        val_dataloader = DataLoader(
            val_source_split,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memoery,
        )
        if gen_val_test:
            data_output[key] = {
                "Train": train_dataloader,
                "Val": val_dataloader,
                "Test": test_dataloader,
            }
        else:
            data_output[key] = {
                "Train": train_dataloader,
                "Test": val_dataloader,
            }

    return data_output, class_names
