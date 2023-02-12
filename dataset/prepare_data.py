import logging
import os
import pandas as pd
import cv2
import numpy as np
import shutil
from models.config import TEST_RATIO, VALIDATION_RATIO, STACKS, DATASET_SIZE

delimiter = "_w"
dataset_dir = "all"
test_img_cnt = int(DATASET_SIZE * TEST_RATIO)
val_img_cnt = int(DATASET_SIZE * VALIDATION_RATIO)

logger = logging.getLogger("data_preparation")
logger.setLevel(logging.DEBUG)


def rename_files(dir_name, z_stacks):
    """
    Renames the files in the given directory and subdirectories.

    Renames the files at the path `dir_name/raw`, for each of the subdirectories in `z_stacks`.
    :param dir_name: The name of the directory.
    :param z_stacks: The stack-directories in which the files are renamed.
    """
    for directory in [os.path.join(dir_name, ("z" + str(i))) for i in z_stacks]:
        logger.info(f"Renaming files in {directory}...")
        for image_name in sorted(os.listdir(directory)):
            if image_name.endswith(".tif"):
                f_name, f_ext = os.path.splitext(image_name)
                parts = f_name.split(delimiter)
                f_new = parts[0] + delimiter + parts[1][0]
                new_name = f'{f_new}{f_ext}'
                new_path = os.path.join(directory, new_name)
                os.rename(os.path.join(directory, image_name), new_path)


def split_images(dir_name, z_stacks):
    """
    Creates a train-val-test split from the images in the given directory and subdirectories.

    Selects `TEST_RATIO`% images for testing, `VALIDATION_RATIO`% for validation and the remaining for training.
    Moves them to the corresponding directories (train/test/val).
    :param dir_name: the directory where the images are
    :param z_stacks: the z-stacks to be considered
    """
    for directory in [os.path.join(dir_name, ("z" + str(i))) for i in z_stacks]:
        logger.info(f"Creating TRAIN-VAL-TEST split from directory {directory}...")
        images = sorted(os.listdir(directory))
        for_test = images[:test_img_cnt]
        for_val = images[test_img_cnt:(test_img_cnt + val_img_cnt)]
        for_train = images[(test_img_cnt + val_img_cnt):]
        _move_images(directory, "test", for_test)
        _move_images(directory, "val", for_val)
        _move_images(directory, "train", for_train)


def _move_images(original_dir, move_to: str, images):
    """
    Moves images to the provided destination (a train/test/val directory). After moving, it also processes them for
    visualization.
    :param original_dir: the original directory of the images
    :param move_to: to which directory will the images be moved. Options: train/test/val
    :param images: the names of the images to be moved
    """
    logger.info(f"Moving images to {move_to} directory...")
    for img in images:
        complete_path = os.path.join(original_dir, img)
        path_parts = complete_path.split(os.path.sep)  # all/z0/img_name
        new_path = os.path.join(move_to, "raw", str(path_parts[-2]), str(path_parts[-1]))
        os.makedirs(os.path.join(move_to, "raw", str(path_parts[-2])), exist_ok=True)
        shutil.copy(complete_path, new_path)
        _process_image(new_path)


def _process_image(image_path):
    """
    Prepare the TIF-format images for processing.

    The image size is 696 x 520 pixels, in 16-bit TIF format, LZW compression.
    This method reads the image, converts it to 8 bits, applies normalization, and moves it to a directory of
    processed images.
    """
    image = cv2.imread(image_path, -1)
    # normalize each pixel between 0 and 1
    path_parts = image_path.split(os.path.sep)  # train/raw/z0/img_name.tif
    empty = np.zeros((696, 520))  # each image is 696x520
    image = cv2.normalize(image, empty, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    image = np.array(image, dtype="uint8")
    # new_sizes = (128, 128)
    # img_resized = cv2.resize(image, dsize=new_sizes, interpolation=cv2.INTER_CUBIC)
    # save the image in the provided directory, in a subdirectory representing the z-stack
    path = os.path.join(path_parts[0], "processed", path_parts[-2], path_parts[-1])
    os.makedirs(os.path.join(path_parts[0], "processed", path_parts[-2]), exist_ok=True)
    cv2.imwrite(path, image)
    return image


def prepare_splits(z_stacks, slide_type, task="train"):
    """
    Selects images for the `task` phase and writes them to a CSV file.
    :param z_stacks: which z-stacks are used
    :param slide_type: the type of slides ("w1" or "w2")
    :param task: for which task are the images selected. Options: train, test, val
    """
    logger.info(f"Preparing splits for slides {slide_type} and phase {task}.")
    if task not in ["train", "test", "val"]:
        raise Exception(f"Unknown value for task parameter: {task}. Must be on of train/test/val.")
    directory_prefix = "z"
    pretrain_ds = []
    for i in z_stacks:
        dir_name = os.path.join(task, "raw", directory_prefix + str(i))
        for image_name in sorted(os.listdir(dir_name)):
            pretrain_ds.append(os.path.join(directory_prefix + str(i), image_name))
    pd.DataFrame(pretrain_ds).to_csv(f"{task}_{slide_type}.csv")


def form_triplets(dir_name: str, triplets, slide_type, deblur=False):
    """
    Creates input triplets for training the network.
    :param dir_name: The directory (train/test/val) from which to choose images
    :param triplets: The z-stack triplets to form
    :return: a dataframe containing 3 columns, each representing a file in the triplet
    """
    assert slide_type in ["w1", "w2"]
    directory_prefix = "z"
    triplets_df = pd.DataFrame(columns=["Input1", "Input2", "Target"])
    first_input, second_input, target, alphas = [], [], [], []
    for triplet in triplets:
        logger.info(f"Processing triplet {triplet}...")
        dir_name_1 = os.path.join(dir_name, "raw", directory_prefix + str(triplet[0]))
        dir_name_2 = os.path.join(directory_prefix + str(triplet[1]))
        dir_name_3 = os.path.join(directory_prefix + str(triplet[2]))
        for image_name in sorted(os.listdir(dir_name_1)):
            if image_name.endswith(".tif") and slide_type in image_name:
                first_input.append(os.path.join(directory_prefix + str(triplet[0]), image_name))
                second_input.append(os.path.join(dir_name_2, image_name))
                target.append(os.path.join(dir_name_3, image_name))
                alphas.append(triplet[-1])
    triplets_df["Input1"] = first_input
    triplets_df["Input2"] = second_input
    triplets_df["Target"] = target
    triplets_df["Alpha"] = alphas
    triplets_df.to_csv(f"{dir_name}_deblur_{slide_type}.csv") if deblur else triplets_df.to_csv(f"{dir_name}_{slide_type}.csv")
    return triplets_df


if __name__ == "__main__":

    # rename_files("all", [*STACKS])
    # split_images("all", [*STACKS])
    #
    # # a triplet is of the form: lower_blur_level, higher_blur_level, intermediate_blur_level, interpolation_param
    triplets_train = [(0, 16, 8, 0.5), (0, 8, 4, 0.5), (8, 16, 12, 0.5), (0, 4, 2, 0.5), (4, 8, 6, 0.5),
                      (8, 12, 10, 0.5), (12, 16, 14, 0.5)]
    # Prepare train-test-val sets for the w1 and w2 sets of images
    for mode in ["train", "val", "test"]:
        for slide_type in ["w1", "w2"]:
            prepare_splits(STACKS, slide_type, mode)
            # define input triplets based on z-stack level. the last argument is the interpolation parameter.
            form_triplets(mode, triplets_train, slide_type)

    # triplets_train = [(0, 16, 2, 1/8), (0, 16, 4, 2/8), (0, 16, 6, 3/8), (0, 16, 8, 4/8), (0, 16, 10, 5/8),
    #                   (0, 16, 12, 6/8), (0, 16, 14, 7/8)]

    # Comment the lines below to generate a test set for deblurring
    # For deblurring, fix left input to z-stack0, vary the others from z-stack2 to z-stack14
    # triplets_deblur = [(0, 2, 16, 7/8), (0, 4, 16, 6/8), (0, 6, 16, 5/8), (0, 8, 16, 4/8), (0, 10, 16, 3/8),
    #                   (0, 12, 16, 2/8), (0, 14, 16, 1/8)]
    #
    # # x = a * xl + (1-a)*xr =>  xr = 1/(1-a)*x - a/(1-a) xl
    #
    # for mode in ["test"]:
    #     for slide_type in ["w1", "w2"]:
    #         form_triplets(mode, triplets_deblur, slide_type, deblur=True)
