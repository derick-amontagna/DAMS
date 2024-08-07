"""
Contains functionality for utily
"""

import os
import pathlib
import random

import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Dict
from loguru import logger


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folder names in a target directory.

    Assumes target directory is in standard image classification format.

    Args:
        directory (str): target directory to load classnames from.

    Returns:
        Tuple[List[str], Dict[str, int]]: (list_of_class_names, dict(class_name: idx...))

    Example:
        find_classes("food_images/train")
        >>> (["class_1", "class_2"], {"class_1": 0, ...})
    """
    # 1. Get the class names by scanning the target directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

    # 2. Raise an error if class names not found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")

    # 3. Create a dictionary of index labels (computers prefer numerical rather than string labels)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def plot_loss_curves(results: dict):
    # Get the loss values of the results dictionary (training and val)
    loss = results["train_loss"]
    val_loss = results["val_loss"]

    # Get the accuracy values of the results dictionary (training and val)
    accuracy = results["train_acc"]
    val_accuracy = results["val_acc"]

    # Figure out how many epochs there were
    epochs = range(len(results["train_loss"]))

    # Setup a plot
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, val_accuracy, label="val_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()


def print_train_time(start: float, end: float, device: torch.device = None):
    time_elapsed = end - start
    logger.info(
        f"Train time on {device}: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
    )
    return f"{time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def imshow_dataloader(data_dataloader, class_names, num_images=3):
    """Imshow for Tensor."""
    inputs, classes = next(iter(data_dataloader))
    inp = torchvision.utils.make_grid(inputs[0:num_images])
    title = title = [class_names[x] for x in classes[0:num_images]]
    imshow(inp, title)


def visualize_model(model, dataloaders, device, class_names, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloaders):
            y = y.type(torch.LongTensor)  # casting to long
            inputs, labels = X.to(device), y.to(device)
            outputs = model(inputs)

            target_image_pred_probs = torch.softmax(outputs, dim=1)
            target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis("off")
                ax.set_title(
                    f"Pred: {class_names[target_image_pred_label[j]]} | Prob: {target_image_pred_probs[j].max():.3f} | True: {class_names[labels[j]]}"
                )
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def set_parameter_requires_grad(model, feature_extracting, num_layers=None):
    # Freeze all base layers in the "features" section of the model (the feature extractor) by setting feature_extracting=True and choose the num_layers too freeze. None to Freeze all.
    if feature_extracting:
        if num_layers is not None:
            for param in model.features[0:-num_layers].parameters():
                param.requires_grad = True
        else:
            for param in model.features.parameters():
                param.requires_grad = False
    return model


class EarlyStoppingMinimizeLoss:
    def __init__(self, patience=5):

        self.patience = patience
        self.counter = 0
        self.minor_loss = np.inf
        self.early_stop = False

    def __call__(self, loss):
        if loss < self.minor_loss:
            self.minor_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class ModelCheckpointBestAcc:
    def __init__(self, name, path_save="models"):
        self.path_save = os.path.join(os.getcwd(), path_save)
        self.best_acc = 0.0
        self.model_name = name
        self.model_save_path = os.path.join(self.path_save, self.model_name)

    def __call__(self, model: torch.nn.Module, acc=0.0):
        if acc > self.best_acc:
            self.best_acc = acc

            # Verify model save path
            assert self.model_name.endswith(".pth") or self.model_name.endswith(
                ".pt"
            ), "model_name should end with '.pt' or '.pth'"

            # Save the model state_dict()
            logger.info(
                f"[ModelCheckpoint] Saving model to: {self.path_save} with acc {self.best_acc*100:.4f}"
            )
            torch.save(obj=model.state_dict(), f=self.model_save_path)

    def load_best_model_weights(self):
        return torch.load(f=self.model_save_path), self.best_acc


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError("Boolean value expected.")


def set_random_seed(seed=47):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_directory(type_path: str, args):
    if type_path == "logs":
        directory = os.path.join(
            os.getcwd(), "logs", args.exp_number, args.transfer_loss
        )
    elif type_path == "models":
        directory = os.path.join(
            os.getcwd(), "models", args.exp_number, args.transfer_loss
        )
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    logger.info(f"Directory created successfully! - {directory}")
