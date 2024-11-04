# Implementation taken from https://github.com/easezyc/deep-transfer-learning/blob/master/MUDA/MFSAN/MFSAN_2src/mfsan.py
# Zhu Y, Zhuang F, Wang D. Aligning domain-specific distribution and classifier for cross-domain classification from multiple sources[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2019, 33: 5989-5996.

from __future__ import print_function
import torch
import os
import math
import resnet as models
import copy
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset_train2D import dataset_train, RandomGenerator
from dataset_test3D import dataset3D

# from dataset_test3D import dataset3D
from loader import create_dataloaders_mri_2d, create_dataloaders_mri_3d

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Training settings
batch_size = 32
iteration = 10000
lr = [0.001, 0.01]
momentum = 0.9
cuda = True
seed = 8
log_interval = 20
l2_decay = 5e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
source1_name = "Siemens"
source2_name = "Philips"
target_name = "GE"
dataset = "ADNI2-T1"
type_training = "w_DA"

IMG_PATH = "./Dataset/ADNI1"
results_dir = "./Results"
img_size = 224
num_workers = 4
W_DA = True

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)


data_source_2d, class_name_2d = create_dataloaders_mri_2d(
    root=f"data\\preprocess\\{dataset}\\6_step_nifti_2d",
    source_1=source1_name,
    source_2=source2_name,
    source_3=target_name,
    transform_train=transforms.Compose(
        [RandomGenerator(output_size=[img_size, img_size])]
    ),
    transform_test_val=None,
    batch_size_train=batch_size,
    batch_size_test_val=1,
    pin_memoery=True,
    gen_test_val=False,
    num_workers=num_workers,
)

data_source_2d_concat, class_name_2d_concat = create_dataloaders_mri_2d(
    root=f"data\\preprocess\\{dataset}\\6_step_nifti_2d",
    source_1=source1_name,
    source_2=source2_name,
    source_3=target_name,
    transform_train=transforms.Compose(
        [RandomGenerator(output_size=[img_size, img_size])]
    ),
    transform_test_val=None,
    batch_size_train=batch_size,
    batch_size_test_val=1,
    pin_memoery=True,
    gen_test_val=False,
    num_workers=num_workers,
    concat_s1_s2=True,
)

data_source_3d, class_name_3d = create_dataloaders_mri_3d(
    root=f"data\\preprocess\\{dataset}\\5_step_class_folders",
    source_1=target_name,
    transform_train=transforms.Compose(
        [RandomGenerator(output_size=[img_size, img_size])]
    ),
    transform_test_val=None,
    batch_size_train=batch_size,
    batch_size_test_val=1,
    pin_memoery=True,
    gen_test_val=True,
    num_workers=num_workers,
)


data_source_3d_source1, __ = create_dataloaders_mri_3d(
    root=f"data\\preprocess\\{dataset}\\5_step_class_folders",
    source_1=source1_name,
    transform_train=transforms.Compose(
        [RandomGenerator(output_size=[img_size, img_size])]
    ),
    transform_test_val=None,
    batch_size_train=batch_size,
    batch_size_test_val=1,
    pin_memoery=True,
    gen_test_val=True,
    num_workers=num_workers,
)

data_source_3d_source2, __ = create_dataloaders_mri_3d(
    root=f"data\\preprocess\\{dataset}\\5_step_class_folders",
    source_1=source2_name,
    transform_train=transforms.Compose(
        [RandomGenerator(output_size=[img_size, img_size])]
    ),
    transform_test_val=None,
    batch_size_train=batch_size,
    batch_size_test_val=1,
    pin_memoery=True,
    gen_test_val=True,
    num_workers=num_workers,
)

data_source_3d_source_concat, __ = create_dataloaders_mri_3d(
    root=f"data\\preprocess\\{dataset}\\5_step_class_folders",
    source_1=source1_name,
    source_2=source2_name,
    transform_train=transforms.Compose(
        [RandomGenerator(output_size=[img_size, img_size])]
    ),
    transform_test_val=None,
    batch_size_train=batch_size,
    batch_size_test_val=1,
    pin_memoery=True,
    gen_test_val=True,
    num_workers=num_workers,
    concat_s1_s2=True
)

source1_loader = data_source_2d["Source_1"]["Train"]
source2_loader = data_source_2d["Source_2"]["Train"]

source1_concat_loader = data_source_2d_concat["Source_1"]["Train"]


target_train_loader = data_source_2d["Source_3"]["Train"]
target_test_loader = data_source_3d["Source_1"]["Test"]
target_valid_loader = data_source_3d["Source_1"]["Val"]

source1_test_loader = data_source_3d_source1["Source_1"]["Test"]
source2_test_loader = data_source_3d_source2["Source_1"]["Test"]
source_concat_test_loader = data_source_3d_source_concat["Source_1"]["Test"]


def train(model):
    if W_DA:
        source1_iter = iter(source1_concat_loader)
        target_iter = iter(target_train_loader)
    else:
        source1_iter = iter(source1_loader)
        source2_iter = iter(source2_loader)
        target_iter = iter(target_train_loader)
    correct = 0
    if W_DA:
        optimizer = torch.optim.SGD(
            [
                {"params": model.sharedNet.parameters()},
                {"params": model.cls_fc_son1.parameters(), "lr": lr[1]},
                {"params": model.sonnet1.parameters(), "lr": lr[1]},
            ],
            lr=lr[0],
            momentum=momentum,
            weight_decay=l2_decay,
        )
    else:
        optimizer = torch.optim.SGD(
            [
                {"params": model.sharedNet.parameters()},
                {"params": model.cls_fc_son1.parameters(), "lr": lr[1]},
                {"params": model.cls_fc_son2.parameters(), "lr": lr[1]},
                {"params": model.sonnet1.parameters(), "lr": lr[1]},
                {"params": model.sonnet2.parameters(), "lr": lr[1]},
            ],
            lr=lr[0],
            momentum=momentum,
            weight_decay=l2_decay,
        )

    for i in range(1, iteration + 1):
        model.train()
        if W_DA:
            optimizer.param_groups[0]["lr"] = lr[0] / math.pow(
                (1 + 10 * (i - 1) / (iteration)), 0.75
            )
            optimizer.param_groups[1]["lr"] = lr[1] / math.pow(
                (1 + 10 * (i - 1) / (iteration)), 0.75
            )
            optimizer.param_groups[2]["lr"] = lr[1] / math.pow(
                (1 + 10 * (i - 1) / (iteration)), 0.75
            )
        else:
            optimizer.param_groups[0]["lr"] = lr[0] / math.pow(
                (1 + 10 * (i - 1) / (iteration)), 0.75
            )
            optimizer.param_groups[1]["lr"] = lr[1] / math.pow(
                (1 + 10 * (i - 1) / (iteration)), 0.75
            )
            optimizer.param_groups[2]["lr"] = lr[1] / math.pow(
                (1 + 10 * (i - 1) / (iteration)), 0.75
            )
            optimizer.param_groups[3]["lr"] = lr[1] / math.pow(
                (1 + 10 * (i - 1) / (iteration)), 0.75
            )
            optimizer.param_groups[4]["lr"] = lr[1] / math.pow(
                (1 + 10 * (i - 1) / (iteration)), 0.75
            )

        try:
            sampled_batch = next(source1_iter)
            source_data, source_label = sampled_batch["image"], sampled_batch["label"]
        except Exception as err:
            source1_iter = iter(source1_loader)
            sampled_batch = next(source1_iter)
            source_data, source_label = sampled_batch["image"], sampled_batch["label"]

        try:
            sampled_batch = next(target_iter)
            target_data, __ = sampled_batch["image"], sampled_batch["label"]
        except Exception as err:
            target_iter = iter(target_train_loader)
            sampled_batch = next(target_iter)
            target_data, __ = sampled_batch["image"], sampled_batch["label"]

        if cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
            target_data = target_data.cuda()

        source_data, source_label = Variable(source_data), Variable(source_label)
        target_data = Variable(target_data)

        optimizer.zero_grad()
        ce_loss, joint_loss, l1_loss = model(
            source_data, target_data, source_label, mark=1
        )
        if W_DA:
            gamma = 2 / (1 + math.exp(-10 * (i) / (iteration))) - 1
            loss = ce_loss
        else:
            gamma = 2 / (1 + math.exp(-10 * (i) / (iteration))) - 1
            loss = ce_loss + gamma * (joint_loss + l1_loss)
        loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            print(
                "Train source1 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tCE_Loss: {:.6f}\tjoint_Loss: {:.6f}\tl1_Loss: {:.6f}".format(
                    i,
                    100.0 * i / iteration,
                    loss.item(),
                    ce_loss.item(),
                    joint_loss.item(),
                    l1_loss.item(),
                )
            )

        if not W_DA:
            try:
                sampled_batch = next(source2_iter)
                source_data, source_label = (
                    sampled_batch["image"],
                    sampled_batch["label"],
                )
            except Exception as err:
                source2_iter = iter(source2_loader)
                sampled_batch = next(source2_iter)
                source_data, source_label = (
                    sampled_batch["image"],
                    sampled_batch["label"],
                )

            try:
                sampled_batch = next(target_iter)
                target_data, __ = sampled_batch["image"], sampled_batch["label"]
            except Exception as err:
                target_iter = iter(target_train_loader)
                sampled_batch = next(target_iter)
                target_data, __ = sampled_batch["image"], sampled_batch["label"]

            if cuda:
                source_data, source_label = source_data.cuda(), source_label.cuda()
                target_data = target_data.cuda()

            source_data, source_label = Variable(source_data), Variable(source_label)
            target_data = Variable(target_data)

            optimizer.zero_grad()
            ce_loss, joint_loss, l1_loss = model(
                source_data, target_data, source_label, mark=2
            )
            if W_DA:
                gamma = 2 / (1 + math.exp(-10 * (i) / (iteration))) - 1
                loss = ce_loss + gamma * l1_loss
            else:
                gamma = 2 / (1 + math.exp(-10 * (i) / (iteration))) - 1
                loss = ce_loss + gamma * (joint_loss + l1_loss)
            loss.backward()
            optimizer.step()

            if i % log_interval == 0:
                print(
                    "Train source2 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tCE_Loss: {:.6f}\tjoint_Loss: {:.6f}\tl1_Loss: {:.6f}".format(
                        i,
                        100.0 * i / iteration,
                        loss.item(),
                        ce_loss.item(),
                        joint_loss.item(),
                        l1_loss.item(),
                    )
                )

        if i % (log_interval * 10) == 0:
            t_correct = valid(model, target_valid_loader)
            if t_correct >= correct:
                correct = t_correct
                torch.save(
                    model.state_dict(),
                    results_dir
                    + "/"
                    + dataset
                    + "_"
                    + source1_name
                    + "_"
                    + source2_name
                    + "_to_"
                    + target_name
                    + "_type_"
                    + type_training
                    + "_max_accuracy.pth",
                )
                max_model = copy.deepcopy(model)
            print(
                "Best performance: ",
                dataset,
                source1_name,
                source2_name,
                "to",
                target_name,
                "max correct:",
                correct,
                "Accuracy: {}/{} ({:.0f}%)".format(
                    correct,
                    len(target_valid_loader.dataset),
                    100.0 * correct / len(target_valid_loader.dataset),
                ),
                "\n",
            )

    if W_DA:
        test_correct_source1 = valid(max_model, source_concat_test_loader)
        acc_s1 = test_correct_source1 / len(source_concat_test_loader.dataset)
        print(
            "Testset performance on Source: ",
            "Accuracy: {}/{} ({:.0f}%)".format(
                test_correct_source1,
                len(source_concat_test_loader.dataset),
                100.0 * acc_s1,
            ),
            "\n",
        )
    else:
        test_correct_source1 = valid(max_model, source1_test_loader)
        test_correct_source2 = valid(max_model, source2_test_loader)
        acc_s1 = test_correct_source1 / len(source1_test_loader.dataset)
        acc_s2 = test_correct_source2 / len(source2_test_loader.dataset)
        acc_s = (acc_s1 + acc_s2) / 2
        print(
            "Testset performance on Source: ",
            "Accuracy: {} ({:.0f}%)".format(
                acc_s,
                100.0 * acc_s,
            ),
            "\n",
        )

    test_correct = valid(max_model, target_test_loader)
    print(
        "Testset performance on Target: ",
        dataset,
        source1_name,
        source2_name,
        "to",
        target_name,
        "Testset correct:",
        test_correct,
        "Accuracy: {}/{} ({:.0f}%)".format(
            test_correct,
            len(target_test_loader.dataset),
            100.0 * test_correct / len(target_test_loader.dataset),
        ),
        "\n",
    )


def valid(model, loader):
    model.eval()
    test_loss = 0
    correct = 0
    correct1 = 0
    correct2 = 0
    total = 0

    with torch.no_grad():
        for data_3D, target in loader:
            temp_correct = 0
            temp_correct1 = 0
            temp_correct2 = 0
            if cuda:
                data_3D, target = data_3D.cuda(), target.cuda()
            for slice_number in range(
                94, 126
            ):  # this slice range in the 2D coronal plane used to train the model
                temp_x = data_3D[:, :, :, slice_number, :]
                temp_x = temp_x.repeat(1, 3, 1, 1)
                data, target = Variable(temp_x), Variable(target)
                if W_DA:
                    pred1 = model(data, mark=0)
                    pred1 = torch.nn.functional.softmax(pred1, dim=1)
                    pred = pred1
                    test_loss += F.nll_loss(F.log_softmax(pred, dim=1), target).item()
                    pred = pred.data.max(1)[1]
                    temp_correct += pred.eq(target.data.view_as(pred)).cpu().sum()
                    pred = pred1.data.max(1)[1]
                    temp_correct1 += pred.eq(target.data.view_as(pred)).cpu().sum()
                    temp_correct2 = 0
                else:
                    pred1, pred2 = model(data, mark=0)
                    pred1 = torch.nn.functional.softmax(pred1, dim=1)
                    pred2 = torch.nn.functional.softmax(pred2, dim=1)
                    pred = (pred1 + pred2) / 2
                    test_loss += F.nll_loss(F.log_softmax(pred, dim=1), target).item()
                    pred = pred.data.max(1)[1]
                    temp_correct += pred.eq(target.data.view_as(pred)).cpu().sum()
                    pred = pred1.data.max(1)[1]
                    temp_correct1 += pred.eq(target.data.view_as(pred)).cpu().sum()
                    pred = pred2.data.max(1)[1]
                    temp_correct2 += pred.eq(target.data.view_as(pred)).cpu().sum()

            if temp_correct >= 15:
                correct = correct + 1
            if temp_correct1 >= 15:
                correct1 = correct1 + 1
            if temp_correct2 >= 15:
                correct2 = correct2 + 1
            total = total + 1

        # print(total, correct, correct1, correct2)

        test_loss /= len(loader.dataset)
        print(
            target_name,
            " Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
                test_loss,
                correct,
                len(loader.dataset),
                100.0 * correct / len(loader.dataset),
            ),
        )
        print(
            "source1 correct: {}, source2 correct: {}, Average correct: {}".format(
                correct1, correct2, correct
            )
        )
    return correct


if __name__ == "__main__":
    model = models.DAMS(num_classes=2, W_DA=W_DA)
    if cuda:
        model.cuda()
    train(model)
