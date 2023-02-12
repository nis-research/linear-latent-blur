import argparse
import logging

import torch.nn

from models.config import USE_TEN_CROP_TESTING
from models.model import AE
import pytorch_lightning as pl
from FocalFrequencyLoss import FocalFrequencyLoss as FFL
from metrics import *
from models.utils import create_img_folder, generate_interpolations_test, interpolate_for_deblur
from models.visualizations import *
from dataset import datasetModules as ds
from cleanfid import fid


# For a model we can:
# - compute metrics (PSNR, FID, FFL, LDS, APD)
# - visualize latent space codes (process encodings, visualize_2d_projections  --> plot like one used in the paper)
# - visualize how different models reconstruct the (original/generated) images (grid_comparison_reconstructions)
# - visualize deblurred images (with varying or fixed alpha values)

def init_logger(slide_type, experiment_name):
    test_logger = logging.getLogger("test_logger")
    test_logger.setLevel(logging.DEBUG)
    f_handler = logging.FileHandler(os.path.join(f"{slide_type}-experiments", experiment_name, f"{experiment_name}.log"))
    f_handler.setLevel(logging.DEBUG)
    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(logging.DEBUG)
    test_logger.addHandler(f_handler)
    return test_logger


def visualize_test_images():
    """
    Creates directories with the test images for w1 and w2 slides.
    """
    w1, _, _ = ds.test_images("w1")
    w2, _, _ = ds.test_images("w2")
    create_img_folder(w1, "w1-test-set")
    create_img_folder(w2, "w2-test-set")


def compute_fid():
    """
    Computes FID between reconstructed original images and their corresponding generated images.
    Use the `create_image_folder()` method to first generate the directories with the (original or generated)
    reconstructed images.
    """
    fid_score = fid.compute_fid("strong-64-1024-w1-ir", "strong-64-1024-w1-or", num_workers=0)
    logger.info(f"FID Strong-w1 {fid_score}")
    fid_score = fid.compute_fid("baseline-64-1024-w1-ir", "baseline-64-1024-w1-or", num_workers=0)
    logger.info(f"FID Baseline-w1 {fid_score}")
    fid_score = fid.compute_fid("weak-64-1024-w1-ir", "weak-64-1024-w1-or", num_workers=0)
    logger.info(f"FID Weak-w1 {fid_score}")
    fid_score = fid.compute_fid("baseline-64-1024-w2-ir", "baseline-64-1024-w2-or", num_workers=0)
    logger.info(f"FID Baseline-w2 {fid_score}")
    fid_score = fid.compute_fid("weak-64-1024-w2-ir", "weak-64-1024-w2-or", num_workers=0)
    logger.info(f"FID Weak-w2 {fid_score}")
    fid_score = fid.compute_fid("strong-64-1024-w2-ir", "strong-64-1024-w2-or", num_workers=0)
    logger.info(f"FID Strong-w2 {fid_score}")


def perform_blurring(ae, images_gt, experiment_name, slide_type):
    encodings, reconstructed_images = ae.encode_decode(images_gt)
    interpolations = generate_interpolations_test(encodings)
    generated_images = ae.decode(interpolations)
    torch.save(generated_images, os.path.join(f"{slide_type}-experiments", experiment_name,
                                              f"synthesised_blur_{experiment_name}.pt"))
    torch.save(encodings, os.path.join(f"{slide_type}-experiments", experiment_name,
                                       f"latent_codes_{experiment_name}.pt"))
    torch.save(reconstructed_images, os.path.join(f"{slide_type}-experiments", experiment_name,
                                                  f"reconstructed_blur_{experiment_name}.pt"))
    return encodings, interpolations, generated_images, reconstructed_images


def perform_deblurring(ae, experiment_name, slide_type):
    left_imgs, right_imgs, target_imgs, alphas = ds.input_for_deblurring(slide_type,
                                                                         use_ten_crop=USE_TEN_CROP_TESTING)
    left_encodings, left_rec = ae.encode_decode(left_imgs)
    right_encodings, right_rec = ae.encode_decode(right_imgs)
    interpolated_codes = interpolate_for_deblur(left_encodings, right_encodings, alphas)
    deblurred_imgs = ae.decode(interpolated_codes)
    _, target_reconstructed = ae.encode_decode(target_imgs)
    torch.save(target_reconstructed, os.path.join(f"{slide_type}-experiments", experiment_name,
                                                  f"reconstructed_sharp_{experiment_name}.pt"))
    torch.save(deblurred_imgs, os.path.join(f"{slide_type}-experiments", experiment_name,
                                            f"generated_sharp_{experiment_name}.pt"))
    torch.save(left_rec, os.path.join(f"{slide_type}-experiments", experiment_name,
                                      f"left_rec_by_{experiment_name}.pt"))
    torch.save(right_rec, os.path.join(f"{slide_type}-experiments", experiment_name,
                                       f"right_rec_by_{experiment_name}.pt"))
    if not os.path.exists(os.path.join(f"{slide_type}-experiments", f"deblurred_gt_{slide_type}.pt")):
        torch.save(target_imgs, os.path.join(f"{slide_type}-experiments", f"deblurred_gt_{slide_type}.pt"))
    return deblurred_imgs, target_reconstructed, target_imgs


def compute_metrics(experiment_name, slide_type, logger, vnum=None, epoch=None, step=None):
    logger.info(f"Computing evaluation metrics for experiment {experiment_name}.")
    assert slide_type in ["w1", "w2"]
    dir_path = os.path.dirname(os.path.abspath(__file__))
    images_gt, _, _ = ds.test_images(slide_type=slide_type)
    os.makedirs(os.path.join(f"{slide_type}-experiments", experiment_name), exist_ok=True)
    with torch.no_grad():
        try:
            latent_codes = torch.load(os.path.join(f"{slide_type}-experiments", experiment_name,
                                                   f"latent_codes_{experiment_name}.pt"), map_location="cpu")
            reconstructed_images = torch.load(os.path.join(f"{slide_type}-experiments", experiment_name,
                                                           f"reconstructed_blur_{experiment_name}.pt"),
                                              map_location="cpu")
            generated_images = torch.load(os.path.join(f"{slide_type}-experiments", experiment_name,
                                                       f"synthesised_blur_{experiment_name}.pt"), map_location="cpu")
            sharp_generated = torch.load(os.path.join(f"{slide_type}-experiments", experiment_name,
                                                      f"generated_sharp_{experiment_name}.pt"), map_location="cpu")
            sharp_reconstructed = torch.load(os.path.join(f"{slide_type}-experiments", experiment_name,
                                                          f"reconstructed_sharp_{experiment_name}.pt"),
                                             map_location="cpu")
            sharp_ground_truth = torch.load(os.path.join(f"{slide_type}-experiments", f"deblurred_gt_{slide_type}.pt"),
                                            map_location="cpu")
            interpolated_codes = generate_interpolations_test(latent_codes)
        except FileNotFoundError:
            logger.info("Encoding inputs...")
            pl.seed_everything(42)
            torch.backends.cudnn.determinstic = True
            torch.backends.cudnn.benchmark = False
            ckpt_path = os.path.join(dir_path, "checkpoints", "lightning_logs", f"version_{vnum}", "checkpoints",
                                     f"epoch={epoch}-step={step}.ckpt")
            ae = AE().load_from_checkpoint(ckpt_path)
            ae.eval()
            logger.info("Performing blur synthesis...")
            latent_codes, interpolated_codes, generated_images, reconstructed_images = perform_blurring(ae, images_gt,
                                                                                                        experiment_name,
                                                                                                        slide_type)
            logger.info("Performing deblurring...")
            sharp_generated, sharp_reconstructed, sharp_ground_truth = perform_deblurring(ae, experiment_name,
                                                                                          slide_type)
            ae.train()

        ############# METRICS ###########
        ffl_gen_vs_orig = FFL().forward(generated_images, images_gt)
        ffl_gen_vs_rec = FFL().forward(generated_images, reconstructed_images)
        logger.info(f"FFL (generated vs original) {ffl_gen_vs_orig}")
        logger.info(f"FFL (generated vs reconstructed) {ffl_gen_vs_rec}")
        lds = compute_lds(latent_codes, interpolated_codes)
        logger.info(f"LDS (original vs. interpolated latent codes) = {lds}")
        apd = compute_apd(latent_codes, interpolated_codes)
        logger.info(f"APD (original vs. interpolated latent codes) = {apd}")
        psnr_gen_vs_rec = eval_psnr(generated_images, reconstructed_images)
        psnr_gen_vs_orig = eval_psnr(generated_images, images_gt)
        logger.info(f"PSNR-blur-quality {psnr_gen_vs_orig}")
        logger.info(f"PSNR-blur {psnr_gen_vs_rec}")
        logger.info(f"Deblur PSNR-deblur-quality: {eval_psnr(sharp_generated, sharp_ground_truth)}")
        logger.info(f"Deblur PSNR-deblur: {eval_psnr(sharp_generated, sharp_reconstructed)}")

        return latent_codes, interpolated_codes, reconstructed_images, generated_images


def test(experiment_name=None, slide_type=None, vnum=None, epoch=None, step=None):
    if slide_type is not None:
        assert slide_type in ["w1", "w2"]
    pl.seed_everything(42)
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    save_to = os.path.join(f"{slide_type}-experiments", experiment_name)
    os.makedirs(save_to, exist_ok=True)
    with torch.no_grad():
        try:
            latent_codes = torch.load(os.path.join(f"{slide_type}-experiments", experiment_name,
                                                   f"latent_codes_{experiment_name}.pt"), map_location="cpu")
            reconstructed_blur = torch.load(os.path.join(f"{slide_type}-experiments", experiment_name,
                                                         f"reconstructed_blur_{experiment_name}.pt"),
                                            map_location="cpu")
            generated_blur = torch.load(os.path.join(f"{slide_type}-experiments", experiment_name,
                                                     f"synthesised_blur_{experiment_name}.pt"), map_location="cpu")
            # sharp_generated = torch.load(os.path.join(f"{slide_type}-experiments", experiment_name,
            #                                           f"generated_sharp_{experiment_name}.pt"), map_location="cpu")
            # sharp_reconstructed = torch.load(os.path.join(f"{slide_type}-experiments", experiment_name,
            #                                               f"reconstructed_sharp_{experiment_name}.pt"),
            #                                  map_location="cpu")
            # sharp_ground_truth = torch.load(os.path.join(f"{slide_type}-experiments", f"deblurred_gt_{slide_type}.pt")
            #                                   , map_location="cpu")
            interpolated_codes = generate_interpolations_test(latent_codes)
        except FileNotFoundError:
            print("File not found")
            latent_codes, interpolated_codes, reconstructed_blur, generated_blur = \
                compute_metrics(experiment_name, slide_type, vnum, epoch, step)

    create_img_folder(reconstructed_blur, experiment_name, slide_type)
    create_img_folder(generated_blur, experiment_name, slide_type, True)
    to_visualize = torch.cat([latent_codes, interpolated_codes], dim=0)
    ######### LATENT SPACE VISUALIZATIONS ################
    process_encodings(to_visualize, save_to)


if __name__ == "__main__":
    # First the compute_metrics() method must be called, which will generate the files with tensors corresponding to
    # reconstructed images or latent representations. Then, the test() method can be called to perform further
    # assessments through visualizations

    # parser = argparse.ArgumentParser()
    # parser.add_argument("-experiment", required=True, help="The name of the experiment to be tested.", )
    # parser.add_argument("--slide-type", required=True, choices=["w1", "w2"],
    #                     help="With which type of slides (w1 or w2) will the model be trained?")
    # parser.add_argument("-vnum", default=None)
    # parser.add_argument("-epoch", default=None)
    # parser.add_argument("-step", default=None)
    #
    # args = parser.parse_args()
    logger = init_logger("w1", "strong_w1")  # init_logger(args.experiment)
    # compute_metrics(args.experiment, args.slide_type, logger, args.vnum, args.epoch, args.step)
    compute_metrics("strong_w1", "w1", logger, 2, 29, 7080)
    # test(args.experiment, args.slide_type)
    test("strong_w1", "w1")
