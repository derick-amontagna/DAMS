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
    # To remove the subfolder structure and move to the ID folder.
    for path_id in tqdm(glob.glob(origin_path + "\*")):
        path_del = glob.glob(path_id + "\*")[0]
        for path_image in glob.glob(path_id + "\*\*\*\*.dcm"):
            try:
                shutil.move(path_image, path_id)
            except:
                logger.info("Already done...")
        try:
            shutil.rmtree(path_del)
        except:
            logger.info("Already done...")

    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
        logger.info(f"The new directory is created! - {dst_path}")
    for dcm_path in tqdm(glob.glob(origin_path + "\*")):
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
        moving_image = ants.image_read(moving_image_path, reorient="ASR")
        template_image = ants.image_read(template_path, reorient="ASR")
        transformation = ants.registration(
            fixed=template_image,
            moving=moving_image,
            type_of_transform="Affine",
            verbose=True,
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


def create_folder_and_move_image(
    origin_path: str = os.path.join(
        BASE_DIR, "data", "ADNI1", "ADNI1-Screening-Nifti-Relevant-Slices"
    ),
    dst_path: str = os.path.join(
        BASE_DIR, "data", "ADNI1", "ADNI1-Screening-Nifti-Class-Folders"
    ),
    df: pd.DataFrame = None,
    column_class: str = "Research Group",
    domain: bool = False,
):
    # Filter the images for each domain if exist or this will take all
    filter_img = []
    for item in glob.glob(os.path.join(origin_path, "*.nii.gz"), recursive=True):
        filter_img.append(item.split("\\")[-1].replace(".nii.gz", ""))
    df = df[df["Subject ID"].isin(filter_img)]
    # Init the create folder and Move Image
    list_class = df[column_class].unique().tolist()
    for class_name in list_class:
        dst_dir = os.path.join(dst_path, class_name)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
            logger.info(f"The new directory is created! - {dst_dir}")

    for class_name in list_class:
        dst_dir = os.path.join(dst_path, class_name)
        list_id = df[df[column_class] == class_name]["Subject ID"].to_list()
        logger.info(f"Started copy the class! - {class_name}")
        for id_mri in tqdm(list_id):
            if domain:
                shutil.move(
                    os.path.join(origin_path, f"{id_mri}.nii.gz"),
                    os.path.join(dst_dir, f"{id_mri}.nii.gz"),
                )
            else:
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
):
    if domain_manufacturer or domain_model:
        if domain_manufacturer:
            logger.info("Create the Domain Manufacturer")
            df["Manufacturer"] = (
                df["Imaging Protocol"]
                .str.split(";", expand=True)[1]
                .str.split("=", expand=True)[1]
            )
            create_folder_and_move_image(origin_path, dst_path, df, "Manufacturer")
            for class_name in df["Manufacturer"].unique().tolist():
                create_folder_and_move_image(
                    os.path.join(dst_path, class_name),
                    os.path.join(dst_path, class_name),
                    df,
                    column_class,
                    domain=True,
                )
        else:
            logger.info("Create the Domain Model")
            df["Model"] = (
                df["Imaging Protocol"]
                .str.split(";", expand=True)[2]
                .str.split("=", expand=True)[1]
            )
            create_folder_and_move_image(origin_path, dst_path, df, "Model")
            for class_name in df["Model"].unique().tolist():
                create_folder_and_move_image(
                    os.path.join(dst_path, class_name),
                    os.path.join(dst_path, class_name),
                    df,
                    column_class,
                    domain=True,
                )
    else:
        logger.info("Create the Without Domain")
        create_folder_and_move_image(origin_path, dst_path, df, column_class)


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
                image_data_2D = image_data[i, :, :].astype(np.float32)
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
    if domain:
        path_folder_list = list(
            np.unique(
                [
                    os.path.join(*path.split(os.path.sep)[11:13])
                    for path in glob.glob(
                        os.path.join(origin_path, "*\*\*.nii.gz"), recursive=True
                    )
                ]
            )
        )
        logger.info("Create the Domain Manufacturer")
        gen_3d_to_2d(origin_path, dst_path, path_folder_list)

    else:
        logger.info("Create the Without Domain")
        path_folder_list = list(
            np.unique(
                [
                    os.path.join(*path.split(os.path.sep)[11:13])
                    for path in glob.glob(
                        os.path.join(origin_path, "*\*.nii.gz"), recursive=True
                    )
                ]
            )
        )
        gen_3d_to_2d(origin_path, dst_path, path_folder_list)


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
):
    int_dir = os.path.join(BASE_DIR, "data", "preprocess", dataset_name)
    df = pd.read_csv(df_path)
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
    )
    if gen_2d:
        logger.info(f"6 Step - Transform images 3D to 2D - Optional")
        transform_3d_to_2d(
            origin_path=os.path.join(int_dir, "5_step_class_folders"),
            dst_path=os.path.join(int_dir, "6_step_nifti_2d"),
            domain=domain_manufacturer or domain_model,
        )
