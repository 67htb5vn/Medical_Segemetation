"""
Convert LiTS 2017 (Liver Tumor Segmentation) data into UNet3+ data format
LiTS: https://competitions.codalab.org/competitions/17094
"""
import os
import sys
from glob import glob
from pathlib import Path
from tqdm import tqdm
import numpy as np
import multiprocessing as mp
import cv2
import nibabel as nib
import hydra
from omegaconf import DictConfig

sys.path.append(os.path.abspath("./"))
from utils.general_utils import create_directory, join_paths
from utils.images_utils import resize_image


# def read_nii(filepath):
#     """
#     Reads .nii file and returns pixel array
#     """
#     ct_scan = nib.load(filepath).get_fdata()
#     # in both train and test set, especially on train scan 130
#     ct_scan = np.rot90(np.array(ct_scan))
#     return ct_scan

# def read_nii(filepath):
#     scan = nib.load(filepath).get_fdata()
#     scan = np.asarray(scan)

#     # đảm bảo thứ tự (H, W, D)
#     if scan.shape[0] != scan.shape[1]:
#         scan = np.transpose(scan, (1, 0, 2))

#     return scan

def read_nii(filepath):
    scan = nib.load(filepath).get_fdata()
    scan = np.asarray(scan)

    # LiTS thường có dạng (D, H, W)
    if scan.shape[0] < scan.shape[1]:
        scan = np.transpose(scan, (1, 2, 0))  # -> (H, W, D)

    return scan

def crop_center(img, croph, cropw):
    """
    Center crop on given height and width
    """
    height, width = img.shape[:2]
    starth = height // 2 - (croph // 2)
    startw = width // 2 - (cropw // 2)
    return img[starth:starth + croph, startw:startw + cropw, :]


def linear_scale(img):
    """
    First convert image to range of 0-1 and them scale to 255
    """
    img = (img - img.min(axis=(0, 1))) / (img.max(axis=(0, 1)) - img.min(axis=(0, 1)))
    return img * 255


def clip_scan(img, min_value, max_value):
    """
    Clip scan to given range
    """
    return np.clip(img, min_value, max_value)


def resize_scan(scan, new_height, new_width, scan_type):
    H, W, D = scan.shape
    resized_scan = np.zeros((new_height, new_width, D), dtype=scan.dtype)

    for i in range(D):
        slice_2d = scan[:, :, i]

        if scan_type == "image":
            resized_slice = resize_image(
                slice_2d,
                new_height,
                new_width,
                cv2.INTER_CUBIC
            )
        else:  # mask
            resized_slice = cv2.resize(
                slice_2d,
                (new_width, new_height),
                interpolation=cv2.INTER_NEAREST
            )
            # ÉP NHÃN LẠI CHO CHẮC
            resized_slice = (resized_slice > 0).astype(np.uint8)

        resized_scan[:, :, i] = resized_slice

    return resized_scan


def save_images(scan, save_path, img_index):
    """
    Based on UNet3+ requirement "input image had three channels, including
    the slice to be segmented and the upper and lower slices, which was
    cropped to 320×320" save each scan as separate image with previous and
    next scan concatenated.
    """
    scan_shape = scan.shape
    for index in range(scan_shape[-1]):
        before_index = index - 1 if (index - 1) > 0 else 0
        after_index = index + 1 if (index + 1) < scan_shape[-1] else scan_shape[-1] - 1

        new_img_path = join_paths(save_path, f"image_{img_index}_{index}.png")
        new_image = np.stack(
            (
                scan[:, :, before_index],
                scan[:, :, index],
                scan[:, :, after_index]
            )
            , axis=-1)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)  # RGB to BGR
        cv2.imwrite(new_img_path, new_image)  # save the images as .png


def save_mask(scan, save_path, mask_index):
    """
    Save each scan as separate mask
    """
    for index in range(scan.shape[-1]):
        new_mask_path = join_paths(save_path, f"mask_{mask_index}_{index}.png")
        cv2.imwrite(new_mask_path, scan[:, :, index])  # save grey scale image


def extract_image(cfg, image_path, save_path, scan_type="image", ):
    """
    Extract image from given scan path
    """
    _, index = str(Path(image_path).stem).split("-")

    scan = read_nii(image_path)
    scan = resize_scan(
        scan,
        cfg.DATA_PREPARATION.RESIZED_HEIGHT,
        cfg.DATA_PREPARATION.RESIZED_WIDTH,
        scan_type
    )
    if scan_type == "image":
        scan = clip_scan(
            scan,
            cfg.DATA_PREPARATION.SCAN_MIN_VALUE,
            cfg.DATA_PREPARATION.SCAN_MAX_VALUE
        )
        scan = linear_scale(scan)
        scan = np.uint8(scan)
        save_images(scan, save_path, index)
    else:
        # 0 for background/non-lesion, 1 for liver, 2 for lesion/tumor
        # merging label 2 into label 1, because lesion/tumor is part of liver
        scan = (scan > 0).astype(np.uint8) * 255
        save_mask(scan, save_path, index)


def extract_images(cfg, images_path, save_path, scan_type="image"):
    for image_path in tqdm(images_path):
        extract_image(cfg, image_path, save_path, scan_type)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def preprocess_lits_data(cfg: DictConfig):
    """
    Preprocess LiTS 2017 (Liver Tumor Segmentation) data by extractions
    images and mask into UNet3+ data format
    """
    print("WORK_DIR =", cfg.WORK_DIR)
    train_images_names = glob(
        join_paths(
            cfg.WORK_DIR,
            cfg.DATA_PREPARATION.SCANS_TRAIN_DATA_PATH,
            "volume-*.nii"
        )
    )
    train_mask_names = glob(
        join_paths(
            cfg.WORK_DIR,
            cfg.DATA_PREPARATION.SCANS_TRAIN_DATA_PATH,
            "segmentation-*.nii"
        )
    )

    assert len(train_images_names) == len(train_mask_names), \
        "Train volumes and segmentations are not same in length"

    val_images_names = glob(
        join_paths(
            cfg.WORK_DIR,
            cfg.DATA_PREPARATION.SCANS_VAL_DATA_PATH,
            "volume-*.nii"
        )
    )
    val_mask_names = glob(
        join_paths(
            cfg.WORK_DIR,
            cfg.DATA_PREPARATION.SCANS_VAL_DATA_PATH,
            "segmentation-*.nii"
        )
    )
    assert len(val_images_names) == len(val_mask_names), \
        "Validation volumes and segmentations are not same in length"

    train_images_names = sorted(train_images_names)
    train_mask_names = sorted(train_mask_names)
    val_images_names = sorted(val_images_names)
    val_mask_names = sorted(val_mask_names)

    train_images_path = join_paths(
        cfg.WORK_DIR, cfg.DATASET.TRAIN.IMAGES_PATH
    )
    train_mask_path = join_paths(
        cfg.WORK_DIR, cfg.DATASET.TRAIN.MASK_PATH
    )
    val_images_path = join_paths(
        cfg.WORK_DIR, cfg.DATASET.VAL.IMAGES_PATH
    )
    val_mask_path = join_paths(
        cfg.WORK_DIR, cfg.DATASET.VAL.MASK_PATH
    )

    create_directory(train_images_path)
    create_directory(train_mask_path)
    create_directory(val_images_path)
    create_directory(val_mask_path)

    print("\nExtracting train images")
    extract_images(
        cfg, train_images_names, train_images_path, scan_type="image"
    )
    print("\nExtracting train mask")
    extract_images(
        cfg, train_mask_names, train_mask_path, scan_type="mask"
    )
    print("\nExtracting val images")
    extract_images(
        cfg, val_images_names, val_images_path, scan_type="image"
    )
    print("\nExtracting val mask")
    extract_images(
        cfg, val_mask_names, val_mask_path, scan_type="mask"
    )


if __name__ == '__main__':
    preprocess_lits_data()
