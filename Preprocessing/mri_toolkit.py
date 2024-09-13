"""
Contains functionality for mri 
"""

import os
import shutil
import glob

import ants
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from loguru import logger
from deepbet import run_bet
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split


BASE_DIR = os.path.dirname(os.path.abspath("__file__"))


def pass_step(func):
    def verify_step(*args, **kwargs):
        for k, v in kwargs.items():
            if k == "dst_path":
                if not os.path.exists(v):
                    func(*args, **kwargs)
                else:
                    logger.info("Already done... Pass to next Step")

    return verify_step


def dcm2nifti(dcm_path, nii_out_path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dcm_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    # Added a call to PermuteAxes to change the axes of the data
    image = sitk.PermuteAxes(image, [2, 1, 0])
    sitk.WriteImage(image, nii_out_path)


@pass_step
def transform_dicom2nifti(
    origin_path: str = os.path.join(BASE_DIR, "data", "ADNI1", "ADNI1-Screening-Dicom"),
    dst_path: str = os.path.join(BASE_DIR, "data", "ADNI1", "ADNI1-Screening-Nifti"),
):
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
        logger.info(f"The new directory is created! - {dst_path}")
    for dcm_path in tqdm(
        glob.glob(
            origin_path + f"{os.path.sep}*{os.path.sep}*{os.path.sep}*{os.path.sep}*"
        )
    ):
        id_dcm = dcm_path.split(os.path.sep)[-1]
        nifti_path = os.path.join(dst_path, id_dcm + ".nii.gz")
        dcm2nifti(dcm_path, nifti_path)


@pass_step
def transform_skull_stripping(
    origin_path: str = os.path.join(BASE_DIR, "data", "ADNI1", "ADNI1-Screening-Nifti"),
    dst_path: str = os.path.join(
        BASE_DIR, "data", "ADNI1", "ADNI1-Screening-Nifti-SkullStripping"
    ),
):
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
        logger.info(f"The new directory is created! - {dst_path}")
    input_paths = []
    brain_paths = []
    for path in tqdm(glob.glob(os.path.join(origin_path, "*.nii.gz"), recursive=True)):
        input_paths.append(path)
        brain_paths.append(
            path.replace(
                origin_path.split(os.path.sep)[-1], dst_path.split(os.path.sep)[-1]
            )
        )
    run_bet(
        input_paths, brain_paths, None, None, threshold=0.5, n_dilate=0, no_gpu=False
    )


@pass_step
def transform_registration(
    origin_path: str = os.path.join(
        BASE_DIR, "data", "ADNI1", "ADNI1-Screening-Nifti-SkullStripping"
    ),
    template_path: str = os.path.join(
        BASE_DIR, "data", "Template", "sri24_spm8_T1_brain.nii"
    ),
    dst_path: str = os.path.join(
        BASE_DIR, "data", "ADNI1", "ADNI1-Screening-Nifti-Registered"
    ),
):
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
        logger.info(f"The new directory is created! - {dst_path}")
    for moving_image_path in tqdm(
        glob.glob(os.path.join(origin_path, "*.nii.gz"), recursive=True)
    ):
        moving_image = ants.image_read(moving_image_path)
        template_image = ants.image_read(template_path)
        transformation = ants.registration(
            fixed=template_image,
            moving=moving_image,
            type_of_transform="Affine",
            verbose=False,
        )
        registered_img_ants = transformation["warpedmovout"]
        registered_img_ants.to_file(
            os.path.join(
                moving_image_path.replace(
                    origin_path.split(os.path.sep)[-1], dst_path.split(os.path.sep)[-1]
                )
            )
        )


@pass_step
def transform_extract_relevant_slices(
    origin_path: str = os.path.join(
        BASE_DIR, "data", "ADNI1", "ADNI1-Screening-Nifti-Registered"
    ),
    dst_path: str = os.path.join(
        BASE_DIR, "data", "ADNI1", "ADNI1-Screening-Nifti-Relevant-Slices"
    ),
    slice_thickness=None,
    top_discard=126,
    bottom_discard=94,
    voxel_resample=(1, 1, 1),
):
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
        logger.info(f"The new directory is created! - {dst_path}")

    for path in tqdm(glob.glob(os.path.join(origin_path, "*.nii.gz"), recursive=True)):
        image = ants.image_read(
            path, reorient="ASR"
        )  # Z, Y, X - IAL = Inferior-to-superior, Anterior-to-posterior, Left-to-right
        image_data = image.numpy()  # -> (Z,Y,X)
        image_shape = image_data.shape

        # Validate slice thickness or calculate based on image size (if no header or slice_thickness provided)
        if slice_thickness is None:
            if len(image_shape) != 3:
                raise ValueError("Image data must have 3 dimensions (z, y, x)")
            # Assuming axial slices (z-direction), estimate slice thickness based on total size
            slice_thickness = image.spacing[0]

        # Extract the desired slices
        extracted_data = image_data[bottom_discard:top_discard, :, :]

        # Create a new ANTsImage from the extracted data
        extracted_image = ants.from_numpy(extracted_data)
        mri_image_voxel = ants.resample_image(extracted_image, voxel_resample, False, 1)
        mri_image_voxel.to_file(
            os.path.join(
                path.replace(
                    origin_path.split(os.path.sep)[-1],
                    dst_path.split(os.path.sep)[-1],
                )
            )
        )


def train_val_test_split(
    df: pd.DataFrame = None,
    domain_column: str = "Manufacturer",
    column_class: str = "Research Group",
    test_val: bool = False,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: float = 42,
):
    dict_split = {}
    if domain_column:
        for domain in df[domain_column].unique():
            df_domain = df[df[domain_column] == domain]
            train_df, test_df = train_test_split(
                df_domain,
                test_size=test_size,
                stratify=df_domain[column_class],
                random_state=random_state,
            )
            if test_val:
                train_df, val_df = train_test_split(
                    train_df,
                    test_size=val_size,
                    stratify=train_df[column_class],
                    shuffle=True,
                    random_state=random_state,
                )
                print(
                    f"The split for the domain {domain} is: Train:{train_df.count().iloc[0]} / Val:{val_df.count().iloc[0]} / Test:{test_df.count().iloc[0]}"
                )
                dict_split[domain] = {
                    "train": {
                        class_type: train_df[train_df[column_class] == class_type][
                            "Image ID"
                        ].to_list()
                        for class_type in train_df[column_class].unique().tolist()
                    },
                    "val": {
                        class_type: val_df[val_df[column_class] == class_type][
                            "Image ID"
                        ].to_list()
                        for class_type in val_df[column_class].unique().tolist()
                    },
                    "test": {
                        class_type: test_df[test_df[column_class] == class_type][
                            "Image ID"
                        ].to_list()
                        for class_type in test_df[column_class].unique().tolist()
                    },
                }
            else:
                dict_split[domain] = {
                    "train": {
                        class_type: train_df[train_df[column_class] == class_type][
                            "Image ID"
                        ].to_list()
                        for class_type in train_df[column_class].unique().tolist()
                    },
                    "test": {
                        class_type: test_df[test_df[column_class] == class_type][
                            "Image ID"
                        ].to_list()
                        for class_type in test_df[column_class].unique().tolist()
                    },
                }
                print(
                    f"The split for the domain {domain} is: Train:{train_df.count().iloc[0]} / Test:{test_df.count().iloc[0]}"
                )

        return dict_split
    else:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df[column_class],
            random_state=random_state,
        )
        if test_val:
            train_df, val_df = train_test_split(
                train_df,
                test_size=val_size,
                stratify=train_df[column_class],
                shuffle=True,
                random_state=random_state,
            )
            print(
                f"The split for the domain All is: Train:{train_df.count().iloc[0]} / Val:{val_df.count().iloc[0]} / Test:{test_df.count().iloc[0]}"
            )
            dict_split["All"] = {
                "train": {
                    class_type: train_df[train_df[column_class] == class_type][
                        "Image ID"
                    ].to_list()
                    for class_type in train_df[column_class].unique().tolist()
                },
                "val": {
                    class_type: val_df[val_df[column_class] == class_type][
                        "Image ID"
                    ].to_list()
                    for class_type in val_df[column_class].unique().tolist()
                },
                "test": {
                    class_type: test_df[test_df[column_class] == class_type][
                        "Image ID"
                    ].to_list()
                    for class_type in test_df[column_class].unique().tolist()
                },
            }
        else:
            dict_split["All"] = {
                "train": {
                    class_type: train_df[train_df[column_class] == class_type][
                        "Image ID"
                    ].to_list()
                    for class_type in train_df[column_class].unique().tolist()
                },
                "test": {
                    class_type: test_df[test_df[column_class] == class_type][
                        "Image ID"
                    ].to_list()
                    for class_type in test_df[column_class].unique().tolist()
                },
            }
            print(
                f"The split for the domain All is: Train:{train_df.count().iloc[0]} / Test:{test_df.count().iloc[0]}"
            )
    return dict_split


def create_folder_and_move_image(
    origin_path: str = os.path.join(
        BASE_DIR, "data", "ADNI1", "ADNI1-Screening-Nifti-Relevant-Slices"
    ),
    dst_path: str = os.path.join(
        BASE_DIR, "data", "ADNI1", "ADNI1-Screening-Nifti-Class-Folders"
    ),
    df: pd.DataFrame = None,
    column_class: str = "Research Group",
    column_domain: str = "Manufacturer",
    split_data: dict = None,
    test_val: bool = False,
):
    if test_val:
        list_split = ["train", "val", "test"]
    else:
        list_split = ["train", "test"]
    list_class = df[column_class].unique().tolist()
    if column_domain:
        list_domain = df[column_domain].unique().tolist()
        for domain_name in list_domain:
            for class_name in list_class:
                for split_name in list_split:
                    dst_dir = os.path.join(
                        dst_path, domain_name, split_name, class_name
                    )
                    if not os.path.exists(dst_dir):
                        os.makedirs(dst_dir)
                    for id_mri in split_data[domain_name][split_name][class_name]:
                        shutil.copy(
                            os.path.join(origin_path, f"{id_mri}.nii.gz"),
                            os.path.join(dst_dir, f"{id_mri}.nii.gz"),
                        )
    else:
        for class_name in list_class:
            for split_name in list_split:
                dst_dir = os.path.join(dst_path, split_name, class_name)
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)
                    for id_mri in split_data["All"][split_name][class_name]:
                        shutil.copy(
                            os.path.join(origin_path, f"{id_mri}.nii.gz"),
                            os.path.join(dst_dir, f"{id_mri}.nii.gz"),
                        )


@pass_step
def gen_class_folders(
    origin_path: str = os.path.join(
        BASE_DIR, "data", "ADNI1", "ADNI1-Screening-Nifti-Relevant-Slices"
    ),
    dst_path: str = os.path.join(
        BASE_DIR, "data", "ADNI1", "ADNI1-Screening-Nifti-Class-Folders"
    ),
    df: pd.DataFrame = None,
    column_class: str = "Research Group",
    domain_manufacturer: bool = False,
    domain_model: bool = False,
    test_val: bool = False,
    test_size: float = 0.15,
    val_size: float = 0.15,
):
    df[["Manufacturer", "Model"]] = df["Imaging Protocol"].str.extract(
        "Manufacturer=(.+);Mfg Model=(.+)", expand=True
    )
    if domain_manufacturer or domain_model:
        logger.info("Create the With Domain")
        if domain_manufacturer:
            split_data = train_val_test_split(
                df.copy(),
                "Manufacturer",
                column_class=column_class,
                test_val=test_val,
                test_size=test_size,
                val_size=val_size,
            )
            create_folder_and_move_image(
                origin_path,
                dst_path,
                df,
                column_class,
                "Manufacturer",
                split_data,
                test_val,
            )
        else:
            split_data = train_val_test_split(
                df.copy(),
                "Model",
                column_class=column_class,
                test_val=test_val,
                test_size=test_size,
                val_size=val_size,
            )
            create_folder_and_move_image(
                origin_path, dst_path, df, column_class, "Model", split_data, test_val
            )
    else:
        logger.info("Create the Without Domain")
        split_data = train_val_test_split(
            df.copy(),
            None,
            column_class=column_class,
            test_val=test_val,
            test_size=test_size,
            val_size=val_size,
        )
        create_folder_and_move_image(
            origin_path, dst_path, df, column_class, None, split_data, test_val
        )


def get_path_folder_list(
    origin_path: str = os.path.join(
        BASE_DIR, "data", "ADNI1", "ADNI1-Screening-Nifti-Class-Folders"
    ),
    domain: bool = False,
):
    if domain:
        path_folder_list = list(
            np.unique(
                [
                    os.path.join(*path.split(os.path.sep)[11:14])
                    for path in glob.glob(
                        os.path.join(
                            origin_path, f"*{os.path.sep}*{os.path.sep}*{os.path.sep}"
                        ),
                        recursive=True,
                    )
                ]
            )
        )
    else:
        path_folder_list = list(
            np.unique(
                [
                    os.path.join(*path.split(os.path.sep)[11:13])
                    for path in glob.glob(
                        os.path.join(origin_path, f"*{os.path.sep}*{os.path.sep}"),
                        recursive=True,
                    )
                ]
            )
        )
    return path_folder_list


def gen_3d_to_2d(
    origin_path: str = os.path.join(
        BASE_DIR, "data", "ADNI1", "ADNI1-Screening-Nifti-Class-Folders"
    ),
    dst_path: str = os.path.join(BASE_DIR, "data", "ADNI1", "ADNI1-Screening-Nifti-2D"),
    path_folder_list: list = None,
):
    for class_name in path_folder_list:
        logger.info(f"Started copy the class! - {class_name}")
        class_dst_path = os.path.join(dst_path, class_name)
        if not os.path.exists(class_dst_path):
            os.makedirs(class_dst_path)
            logger.info(f"The new directory is created! - {class_dst_path}")
        for path_mri in tqdm(
            glob.glob(os.path.join(origin_path, class_name, "*.nii.gz"), recursive=True)
        ):
            name_id = path_mri.split(os.path.sep)[-1].split(".")[0]
            image = nib.load(path_mri)
            image_data = image.get_fdata()
            for i in range(image_data.shape[0]):
                dst_path_image = os.path.join(
                    class_dst_path, f"{name_id}_slice_{i}.nii.gz"
                )
                image_data_2D = image_data[:, i, :].astype(np.float32)
                nib.save(
                    nib.Nifti1Image(image_data_2D, affine=np.eye(4)), dst_path_image
                )


@pass_step
def transform_3d_to_2d(
    origin_path: str = os.path.join(
        BASE_DIR, "data", "ADNI1", "ADNI1-Screening-Nifti-Class-Folders"
    ),
    dst_path: str = os.path.join(BASE_DIR, "data", "ADNI1", "ADNI1-Screening-Nifti-2D"),
    domain: bool = False,
):
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
        logger.info(f"The new directory is created! - {dst_path}")
    gen_3d_to_2d(origin_path, dst_path, get_path_folder_list(origin_path, domain))


def preprocess_pipeline(
    origin_path: str = os.path.join(
        BASE_DIR, "data", "ADNI1", "Image", "ADNI1-Screening-Dicom"
    ),
    dataset_name: str = "ADNI1",
    df_path: str = os.path.join(
        BASE_DIR,
        "analytics",
        "ADNI1",
        "ADNI1-Screening-T1-Original-Collect.csv",
    ),
    gen_2d: bool = False,
    column_class: str = "Research Group",
    domain_manufacturer: bool = True,
    domain_model: bool = False,
    test_val: bool = False,
    test_size: float = 0.15,
    val_size: float = 0.15,
):
    int_dir = os.path.join(BASE_DIR, "data", "preprocess", dataset_name)
    df = pd.read_csv(df_path)
    df["Image ID"] = "I" + df["Image ID"].astype(str)
    if not os.path.exists(int_dir):
        os.makedirs(int_dir)
        logger.info(f"The new directory is created! - {int_dir}")
    logger.info(f"1 Step - Transform Dicom to Nifti")
    transform_dicom2nifti(
        origin_path=origin_path,
        dst_path=os.path.join(int_dir, "1_step_dicom2nifti"),
    )
    logger.info(f"2 Step - Apply Skull Stripping")
    transform_skull_stripping(
        origin_path=os.path.join(int_dir, "1_step_dicom2nifti"),
        dst_path=os.path.join(int_dir, "2_step_skull_stripping"),
    )
    logger.info(f"3 Step - Apply Registration")
    transform_registration(
        origin_path=os.path.join(int_dir, "2_step_skull_stripping"),
        dst_path=os.path.join(int_dir, "3_step_registration"),
        template_path=os.path.join(
            BASE_DIR, "Preprocessing", "Template", "MNI152_T1_1mm_Brain.nii"
        ),
    )
    logger.info(f"4 Step - Extract Relevant Slices")
    transform_extract_relevant_slices(
        origin_path=os.path.join(int_dir, "3_step_registration"),
        dst_path=os.path.join(int_dir, "4_step_relevant_slices"),
    )
    logger.info(f"5 Step - Generate class folders")
    gen_class_folders(
        origin_path=os.path.join(int_dir, "4_step_relevant_slices"),
        dst_path=os.path.join(int_dir, "5_step_class_folders"),
        df=df.copy(),
        column_class=column_class,
        domain_manufacturer=domain_manufacturer,
        domain_model=domain_model,
        test_val=test_val,
        test_size=test_size,
        val_size=val_size,
    )
    if gen_2d:
        logger.info(f"6 Step - Transform images 3D to 2D - Optional")
        transform_3d_to_2d(
            origin_path=os.path.join(int_dir, "5_step_class_folders"),
            dst_path=os.path.join(int_dir, "6_step_nifti_2d"),
            domain=domain_manufacturer or domain_model,
        )
