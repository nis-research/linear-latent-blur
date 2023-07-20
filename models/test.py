import argparse
import logging
from torchvision.utils import make_grid, save_image
import torch.nn
from models.config import USE_TEN_CROP_TESTING
from models.model import AE
import pytorch_lightning as pl
from models.FocalFrequencyLoss import FocalFrequencyLoss as FFL
from metrics import *
from models.utils import create_img_folder, generate_interpolations_test, interpolate_for_deblur
from models.visualizations import *
from dataset import datasetModules as ds


# from cleanfid import fid

# For an experiment (model type + slide type combination) we can:
# - compute metrics (PSNR, FID, FFL, LDS, APD)
# - visualize 2D projections of latent space representations (process encodings, visualize_2d_projections
# --> plot like one shown in the paper)
# - visualize how different models reconstruct the (real/synthetic) images (grid_comparison_reconstructions)
# - visualize synthetic sharp images (with varying or fixed alpha values)


def init_logger(slide_type, experiment_name):
    test_logger = logging.getLogger("test_logger")
    test_logger.setLevel(logging.DEBUG)
    os.makedirs(os.path.join(f"{slide_type}-experiments", experiment_name), exist_ok=True)
    f_handler = logging.FileHandler(
        os.path.join(f"{slide_type}-experiments", experiment_name, f"{experiment_name}.log"))
    f_handler.setLevel(logging.DEBUG)
    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(logging.DEBUG)
    test_logger.addHandler(f_handler)
    return test_logger


# def visualize_test_images():
#     """
#     Creates directories with the test images for w1 and w1 slides.
#     """
#     w1, _, _ = ds.test_images("w1")
#     w1, _, _ = ds.test_images("w1")
#     create_img_folder(w1, "w1-test-set")
#     create_img_folder(w1, "w1-test-set")

def perform_blurring(ae, images_gt, experiment_name, slide_type="", save_to=""):
    """
    Encodes a set of images and generates interpolated representations from pairs of images (one pair = 2 different
    blur levels of one cell slide). The reconstructions of the input images in one pair, the target image and the
    synthetic image generated are saved for further usage.
    """
    encodings, reconstructed_images = ae.encode_decode(images_gt)
    interpolations = generate_interpolations_test(encodings)
    generated_images = ae.decode(interpolations)
    torch.save(generated_images, os.path.join(save_to, f"synthesised_blur_{experiment_name}.pt"))
    torch.save(encodings, os.path.join(save_to, f"latent_codes_{experiment_name}.pt"))
    torch.save(reconstructed_images, os.path.join(save_to, f"reconstructed_blur_{experiment_name}.pt"))
    return encodings, interpolations, generated_images, reconstructed_images


def perform_deblurring(ae, experiment_name, slide_type, save_to="", save_to_gt="", input_path=""):
    """
    Generates synthetic sharp images from 2 input images with different blur levels. The reconstructions of the 2 input
    images, original sharp image and synthetic sharp image are saved for further usage.
    """
    left_imgs, right_imgs, target_imgs, alphas = ds.input_for_deblurring(slide_type,
                                                                         use_ten_crop=USE_TEN_CROP_TESTING,
                                                                         input_path=input_path)
    left_encodings, left_rec = ae.encode_decode(left_imgs)
    right_encodings, right_rec = ae.encode_decode(right_imgs)
    interpolated_codes = interpolate_for_deblur(left_encodings, right_encodings, alphas)
    deblurred_imgs = ae.decode(interpolated_codes)
    _, target_reconstructed = ae.encode_decode(target_imgs)
    torch.save(target_reconstructed, os.path.join(save_to, f"reconstructed_sharp_{experiment_name}.pt"))
    torch.save(deblurred_imgs, os.path.join(save_to, f"generated_sharp_{experiment_name}.pt"))
    torch.save(left_rec, os.path.join(save_to, f"left_rec_by_{experiment_name}.pt"))
    torch.save(right_rec, os.path.join(save_to, f"right_rec_by_{experiment_name}.pt"))
    if not os.path.exists(os.path.join(save_to_gt, f"deblurred_gt_{slide_type}.pt")):
        torch.save(target_imgs, os.path.join(save_to_gt, f"deblurred_gt_{slide_type}.pt"))
    return deblurred_imgs, target_reconstructed, target_imgs, interpolated_codes


def deblur_with_alpha_range(image_idx, experiment_name, slide_type, model_type, checkpoint_dir=None, vnum=None,
                            epoch=None, step=None, output_path=""):
    left_imgs, right_imgs, target_imgs, alphas = ds.input_for_deblurring(slide_type=slide_type, img_idx=image_idx)
    if not output_path == "":
        save_to = os.path.join(output_path, f"{slide_type}-experiments", experiment_name)
        gt_save_to = os.path.join(output_path, f"{slide_type}-experiments")
    else:
        save_to = os.path.join(f"{slide_type}-experiments", experiment_name)
        gt_save_to = f"{slide_type}-experiments"
    os.makedirs(save_to, exist_ok=True)

    print("Encoding inputs...")
    pl.seed_everything(42)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False
    if checkpoint_dir:
        ckpt_path = os.path.join(checkpoint_dir, os.listdir(checkpoint_dir)[0])
        # logger.log(f"Checkpoint path: {ckpt_path}")
    else:
        ckpt_path = os.path.join("checkpoints", "lightning_logs", f"version_{vnum}", "checkpoints",
                                 f"epoch={epoch}-step={step}.ckpt")
    # if output_path != "":
    #     ckpt_path = os.path.join(output_path, ckpt_path)
    ae = AE().load_from_checkpoint(ckpt_path)
    ae.set_model_type(model_type)
    ae.eval()
    left_encodings, left_rec = ae.encode_decode(left_imgs)
    right_encodings, right_rec = ae.encode_decode(right_imgs)
    _, target_reconstructed = ae.encode_decode(target_imgs)  # reconstructed sharp
    synthetic_sharp_list = []
    for alpha in [alphas[0], alphas[0] + 1 / 15]:
        # print(alpha)
        interpolated_codes = interpolate_for_deblur(left_encodings, right_encodings, [alpha])
        synthetic_sharp = ae.decode(interpolated_codes)  # synthetic sharp
        synthetic_sharp_list.append(synthetic_sharp[0])
        # print(target_reconstructed.shape)
        # print(synthetic_sharp.shape)
    #     try:
    #         grid = make_grid(torch.stack([synthetic_sharp[0], target_reconstructed[0]], dim=0), nrow=2)
    #     except:
    #         try:
    #             grid = make_grid([synthetic_sharp[0], target_reconstructed[0]], nrow=2)
    #         except:
    #             grid = make_grid([synthetic_sharp, target_reconstructed], nrow=2)
    #     os.makedirs(os.path.join(save_to, "deblur_finetune_alpha"), exist_ok=True)
    #     save_image(grid, os.path.join(save_to, "deblur_finetune_alpha", f"img{image_idx}-{alpha}.png"))
    # try:
    #     grid = make_grid(torch.stack(synthetic_sharp_list, dim=0), nrow=5)
    # except:
    #     grid = make_grid(synthetic_sharp_list, nrow=5)
    # save_image(grid, os.path.join(save_to, "deblur_finetune_alpha", f"img{image_idx}.png"))
    # grid with left_rec, right_rec, sharp_fixed, sharp_adjusted, target_rec, target_original
    grid = make_grid([left_imgs[0], right_imgs[0], synthetic_sharp_list[0], synthetic_sharp_list[1],
                      target_reconstructed[0], target_imgs[0]], nrow=6)
    save_image(grid, os.path.join(f"{experiment_name}_deblur_by_models_img{image_idx}.png"))
    ae.train()
    return synthetic_sharp[0]


def get_gt_transition(img_idx, slide_type="w1"):
    images, _, _ = ds.test_images(slide_type)
    to_viz = [images[img_idx * TRANSITION_LENGTH:img_idx * TRANSITION_LENGTH + TRANSITION_LENGTH]]
    for idx, image in enumerate(to_viz[0]):
        grid = make_grid(image, nrow=1)
        save_image(grid, f"{slide_type}-gt-{idx}.png")
    grid = make_grid(torch.cat(to_viz, dim=0), nrow=TRANSITION_LENGTH)
    save_image(grid, f"{slide_type}-gt-transition.png")


def compute_metrics(experiment_name, slide_type, logger, model_type, checkpoint_dir=None, vnum=None, epoch=None,
                    step=None, output_path="", metrics=True, input_path=""):
    logger.info(f"Computing evaluation metrics for experiment {experiment_name}.")
    assert slide_type in ["w1", "w2"]
    assert model_type in ["baseline", "indirect", "direct"]
    # dir_path = os.path.dirname(os.path.abspath(__file__))
    images_gt, _, _ = ds.test_images(slide_type=slide_type)
    if not output_path == "":
        save_to = os.path.join(output_path, f"{slide_type}-experiments", experiment_name)
        gt_save_to = os.path.join(output_path, f"{slide_type}-experiments")
    else:
        save_to = os.path.join(f"{slide_type}-experiments", experiment_name)
        gt_save_to = f"{slide_type}-experiments"
    os.makedirs(save_to, exist_ok=True)
    with torch.no_grad():
        try:
            latent_codes = torch.load(os.path.join(save_to, f"latent_codes_{experiment_name}.pt"))
            reconstructed_images = torch.load(os.path.join(save_to, f"reconstructed_blur_{experiment_name}.pt"))
            generated_images = torch.load(os.path.join(save_to, f"synthesised_blur_{experiment_name}.pt"))
            sharp_generated = torch.load(os.path.join(save_to, f"generated_sharp_{experiment_name}.pt"))
            sharp_reconstructed = torch.load(os.path.join(save_to, f"reconstructed_sharp_{experiment_name}.pt"))
            sharp_ground_truth = torch.load(os.path.join(gt_save_to, f"deblurred_gt_{slide_type}.pt"))
            interpolated_codes = generate_interpolations_test(latent_codes)
        except FileNotFoundError:
            print("Encoding inputs...")
            pl.seed_everything(42)
            torch.backends.cudnn.determinstic = True
            torch.backends.cudnn.benchmark = False
            if checkpoint_dir:
                ckpt_path = os.path.join(checkpoint_dir, os.listdir(checkpoint_dir)[0])
                logger.log(f"Checkpoint path: {ckpt_path}")
            else:
                ckpt_path = os.path.join("checkpoints", "lightning_logs", f"version_{vnum}", "checkpoints",
                                         f"epoch={epoch}-step={step}.ckpt")
            # if output_path != "":
            #     ckpt_path = os.path.join(output_path, ckpt_path)
            ae = AE().load_from_checkpoint(ckpt_path)
            ae.set_model_type(model_type)
            ae.eval()
            print("Performing blur synthesis...")
            latent_codes, interpolated_codes, generated_images, reconstructed_images = perform_blurring(ae, images_gt,
                                                                                                        experiment_name,
                                                                                                        slide_type,
                                                                                                        save_to)
            print("Performing deblurring...")
            sharp_generated, sharp_reconstructed, sharp_ground_truth, extrapolated_codes = perform_deblurring(ae,
                                                                                                              experiment_name,
                                                                                                              slide_type,
                                                                                                              save_to,
                                                                                                              gt_save_to)
            ae.train()

        ############# METRICS ###########
        ffl_gen_vs_orig = FFL().forward(generated_images, images_gt)
        ffl_gen_vs_rec = FFL().forward(generated_images, reconstructed_images)
        logger.info(f"FFL (grd_interp_b) {ffl_gen_vs_orig}")
        logger.info(f"FFL (b_interp_b) {ffl_gen_vs_rec}")
        if metrics:
            lds = compute_lds(latent_codes)
            logger.info(f"LDS = {lds}")
            apd = compute_apd(latent_codes, interpolated_codes)
            logger.info(f"APD (original vs. interpolated latent codes) = {apd}")
            psnr_gen_vs_rec = eval_psnr(generated_images, reconstructed_images)
            psnr_gen_vs_orig = eval_psnr(generated_images, images_gt)
            logger.info(f"PSNR (grd_interp-b) {psnr_gen_vs_orig}")
            logger.info(f"PSNR (b_interp-b) {psnr_gen_vs_rec}")
            logger.info(f"Deblur PSNR (grd_extr-d): {eval_psnr(sharp_generated, sharp_ground_truth)}")
            logger.info(f"Deblur PSNR (d_extr-d): {eval_psnr(sharp_generated, sharp_reconstructed)}")
            # MSE
            mse_gen_vs_rec = eval_mse(generated_images, reconstructed_images)
            mse_gen_vs_orig = eval_mse(generated_images, images_gt)
            logger.info(f"MSE (grd_interp-b) {mse_gen_vs_orig}")
            logger.info(f"MSE (b_interp-b) {mse_gen_vs_rec}")
            logger.info(f"Deblur MSE (grd_extr-d): {eval_mse(sharp_generated, sharp_ground_truth)}")
            logger.info(f"Deblur MSE (d_extr-d): {eval_mse(sharp_generated, sharp_reconstructed)}")

        return latent_codes, interpolated_codes, reconstructed_images, generated_images, extrapolated_codes


def test(experiment_name=None, slide_type=None, model_type=None, vnum=None, epoch=None, step=None, output_path="",
         logger=None, input_path=""):
    if slide_type is not None:
        assert slide_type in ["w1", "w2"]
    pl.seed_everything(42)
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False
    if not output_path == "":
        save_to = os.path.join(output_path, f"{slide_type}-experiments", experiment_name)
        gt_save_to = os.path.join(output_path, f"{slide_type}-experiments")
    else:
        save_to = os.path.join(f"{slide_type}-experiments", experiment_name)
        gt_save_to = f"{slide_type}-experiments"

    os.makedirs(save_to, exist_ok=True)
    with torch.no_grad():
        try:
            latent_codes = torch.load(os.path.join(save_to, f"latent_codes_{experiment_name}.pt"), map_location="cpu")
            reconstructed_blur = torch.load(os.path.join(save_to, f"reconstructed_blur_{experiment_name}.pt"),
                                            map_location="cpu")
            generated_blur = torch.load(os.path.join(save_to, f"synthesised_blur_{experiment_name}.pt"),
                                        map_location="cpu")
            sharp_generated = torch.load(os.path.join(save_to, f"generated_sharp_{experiment_name}.pt"),
                                         map_location="cpu")
            sharp_reconstructed = torch.load(os.path.join(save_to, f"reconstructed_sharp_{experiment_name}.pt"),
                                             map_location="cpu")
            sharp_ground_truth = torch.load(os.path.join(gt_save_to, f"deblurred_gt_{slide_type}.pt")
                                            , map_location="cpu")
            interpolated_codes = generate_interpolations_test(latent_codes)
        except FileNotFoundError:
            print("File not found")
            latent_codes, interpolated_codes, reconstructed_blur, generated_blur, extrapolated_codes = \
                compute_metrics(experiment_name=experiment_name, slide_type=slide_type, model_type=model_type,
                                vnum=vnum, epoch=epoch, step=step, metrics=False, logger=logger,
                                output_path=output_path, input_path=input_path)

    create_img_folder(reconstructed_blur, experiment_name, slide_type, save_to=save_to)
    create_img_folder(generated_blur, experiment_name, slide_type, True, save_to=save_to)
    to_visualize = torch.cat([latent_codes, interpolated_codes], dim=0)
    # intr_extr_codes = torch.cat([extrapolated_codes, interpolated_codes], dim=0)
    ######### LATENT SPACE VISUALIZATIONS ################
    process_encodings(to_visualize, save_to)
    # process_encodings(intr_extr_codes, save_to)


if __name__ == "__main__":
    # First the compute_metrics() method must be called, which will generate the files with tensors corresponding to
    # images reconstructed by the AE and their latent representations and will compute and log certain eval. metrics.
    # Then, the test() method can be called to perform further assessments through visualizations

    parser = argparse.ArgumentParser()
    # Input and output path must be specified only if the input/output paths are not in the same directory as the
    # project.
    parser.add_argument("--output-path", required=False, help="The path where to write the output files.")
    parser.add_argument("--input-path", required=False, help="The path from where to read the input images.")
    parser.add_argument("-experiment", required=True, help="The name of the experiment.")
    parser.add_argument("--model-type", required=False, help="The model type.", choices=["baseline", "indirect",
                                                                                         "direct"], default="baseline")
    parser.add_argument("--slide-type", required=False, choices=["w1", "w2"], default="w1",
                        help="With which type of slides (w1 or w1) will the model be tested?")
    # vnum, epoch and step for checkpoint file
    parser.add_argument("-vnum", required=True, default=None)
    parser.add_argument("-epoch", required=True, default=None)
    parser.add_argument("-step", required=True, default=None)

    args = parser.parse_args()

    output_path = args.output_path
    input_path = args.input_path
    vnum, epoch, step = args.vnum, args.epoch, args.step
    slide_type = args.slide_type
    exp_name = args.experiment
    model_type = args.model_type

    # logger = init_logger(slide_type, exp_name)
    # compute_metrics(exp_name, slide_type, logger, model_type,  vnum=vnum, epoch=epoch, step=step,
    # output_path=output_path)
    # test(exp_name, slide_type, model_type, vnum, epoch, step, output_path=output_path, logger=logger)

    # Instead of using arguments we can uncomment the row(s) for the experiment we want to test

    # logger_bw1 = init_logger("w1", f"w1-baseline")
    # logger_dw1 = init_logger("w1", f"w1-direct")
    # logger_iw1 = init_logger("w1", f"w1-indirect")
    # logger_bw2 = init_logger("w2", f"w2-baseline")
    # logger_dw2 = init_logger("w2", f"w2-direct")
    # logger_iw2 = init_logger("w2", f"w2-indirect")

    # logger_w2 = init_logger("w2", f"w2")
    # logger_w1 = init_logger("w1", f"w1")

    # W1 set
    # compute_metrics(experiment_name="w1-baseline", slide_type="w1", model_type="baseline", vnum=2275817, epoch=39,
    #                 step=9440, output_path=output_path, logger=logger_bw1, input_path=input_path)
    # test(experiment_name="w1-baseline", slide_type="w1", model_type="baseline", vnum=2275817, epoch=39,
    #                 step=9440, output_path=output_path, logger=logger_w1, input_path=input_path)

    # compute_metrics(experiment_name="w1-indirect", slide_type="w1", model_type="indirect", vnum=2276747, epoch=39,
    #                 step=9440, output_path=output_path, logger=logger_iw1, input_path=input_path)
    # test(experiment_name="w1-indirect", slide_type="w1", model_type="indirect", vnum=2276747, epoch=39,
    #                 step=9440, output_path=output_path, logger=logger_w1, input_path=input_path)

    # compute_metrics(experiment_name="w1-direct", slide_type="w1", model_type="direct", vnum=2277109, epoch=39,
    #                 step=9440, output_path=output_path, logger=logger_dw1, input_path=input_path)
    # test(experiment_name="w1-direct", slide_type="w1", model_type="direct", vnum=2277109, epoch=39,
    #                 step=9440, output_path=output_path, logger=logger_w1, input_path=input_path)

    # W2 set
    # compute_metrics(experiment_name="w2-baseline", slide_type="w2", model_type="baseline", vnum=2312466, epoch=39,
    #                 step=9440, output_path=output_path, logger=logger_bw2, input_path=input_path)
    # test(experiment_name="w2-baseline", slide_type="w2", model_type="baseline", vnum=2312466, epoch=39,
    #                 step=9440, output_path=output_path, logger=logger_w2, input_path=input_path)

    # compute_metrics(experiment_name="w2-indirect", slide_type="w2", model_type="indirect", vnum=2416042, epoch=39,
    #                 step=9440, output_path=output_path, logger=logger_iw2, input_path=input_path)
    # test(experiment_name="w2-indirect", slide_type="w2", model_type="indirect", vnum=2416042, epoch=39,
    #                 step=9440, output_path=output_path, logger=logger_w2, input_path=input_path)

    # compute_metrics(experiment_name="w2-direct", slide_type="w2", model_type="direct", vnum=2419642, epoch=39,
    #                 step=9440, output_path=output_path, logger=logger_dw2, input_path=input_path)
    # test(experiment_name="w2-direct", slide_type="w2", model_type="direct", vnum=2419642, epoch=39,
    #                 step=9440, output_path=output_path, logger=logger_w2, input_path=input_path)


    # Based on individual images (using indices) we can generate various visualizations
    # !! Keep in mind: before visualizations can be generated, `compute_metrics()` must have been called with the 3 args
    # corresponding to the checkpoint file (vnum, epoch, step). However, some of the functions which generate
    # visualizations still need as input these 3 arguments (see below: `deblur_with_alpha_range()`)

    # Functions used to generate figures from the paper:

    # visualize_2d_projections() - generates Fig. 3

    # grid_comparison_reconstructions() - generates Fig. 4(a). The `interpolated` parameter controls weather we
    # generate comparisons between real blurry images and reconstructed blurry images or real blurry images and
    # synthetic blurry images. A reconstructed blurry images uses a non-interpolated latent code, a synthetic blurry
    # image comes from an interpolated latent code.

    # deblur_fixed_alpha() - generated Fig. 4(c). Different alpha values will generate grids with input pairs having
    # various blur levels

    # deblur_visualisations_vary_alpha() - generates Fig. 5

    # Uncomment lines below accordingly. Replaces indices to visualize different cell slides.

    # for image_idx in [14, 476]:
    #     # get a transition from z0 to z16 for an image based on its index
    #     get_gt_transition(image_idx)
    #     deblur_fixed_alpha("w1", 0.125, image_idx, output_path=output_path)
    #     deblur_fixed_alpha("w1", 0.5, image_idx, output_path=output_path)
    #     deblur_fixed_alpha("w1", 0.625, image_idx, output_path=output_path)
    #     deblur_visualisations_vary_alpha_transposed("w1-baseline", "w1", image_idx, output_path=output_path)
    #     deblur_visualisations_vary_alpha_transposed("w1-indirect", "w1", image_idx, output_path=output_path)
    #     deblur_visualisations_vary_alpha_transposed("w1-direct", "w1", image_idx, output_path=output_path)
    #
    #     synthetic_sharp = []
    #     synthetic_sharp.append(deblur_with_alpha_range(image_idx, experiment_name="w1-baseline", slide_type=slide_type,
    #                                                    model_type="baseline", vnum=2275817, epoch=epoch, step=step,
    #                                                    output_path=output_path))
    #     synthetic_sharp.append(deblur_with_alpha_range(image_idx, experiment_name="w1-indirect", slide_type=slide_type,
    #                                                    model_type="indirect",
    #                                                    vnum=2276747, epoch=epoch, step=step, output_path=output_path))
    #     synthetic_sharp.append(deblur_with_alpha_range(image_idx, experiment_name="w1-direct", slide_type=slide_type,
    #                                                    model_type="direct",
    #                                                    vnum=2277109, epoch=epoch, step=step, output_path=output_path))
    #     try:
    #         grid = make_grid(torch.stack(synthetic_sharp, dim=0), nrow=1)
    #     except:
    #         grid = make_grid(synthetic_sharp, nrow=1)
    #     save_image(grid, os.path.join(output_path, "w1-experiments", f"img{image_idx}_bid.png"))
    #
    #     visualize_2d_projections(image_idx, slide_type, output_path)
    #     grid_comparison_reconstructions(image_idx, slide_type, False, output_path)
    #     grid_comparison_reconstructions(image_idx, slide_type, True, output_path)
