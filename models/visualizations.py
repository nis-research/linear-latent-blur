import traceback
from itertools import chain
from torch import Tensor
from torchvision.utils import make_grid, save_image
from dataset import datasetModules as ds
from models.config import TRANSITION_LENGTH
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from models.utils import project_encodings_to_2d


def _read_latent_codes(experiment_name, slide_type, output_path=""):
    return torch.load(os.path.join(output_path, f"{slide_type}-experiments",
                                   experiment_name, f"latent_codes_{experiment_name}.pt"),
                      map_location="cpu")


def read_from_file(filename):
    return torch.load(filename, map_location="cpu")


def visualize_2d_projections(img_idx, slide_type, output_path=""):
    """
    ** Used to generate Fig. 3 - 2D latent representations of a cell slide in the paper. **
    Creates a plot of the 2D-projected latent representations of different blur levels (z-stack 0 through z-stack 16)
    corresponding to one cell slide.

    :param img_idx: the index of the image to be used
    :param slide_type: the type of slides to which the image belongs
    """
    enc_baseline = project_encodings_to_2d(_read_latent_codes(f"{slide_type}-baseline", slide_type, output_path))
    enc_indirect = project_encodings_to_2d(_read_latent_codes(f"{slide_type}-indirect", slide_type, output_path))
    enc_direct = project_encodings_to_2d(_read_latent_codes(f"{slide_type}-direct", slide_type, output_path))
    markers = ["o", "v", "s"]
    colors = ["blue", "red", "green", "purple", "yellow", "pink", "violet", "orange", "brown", "grey", "black"]
    encodings = [enc_baseline, enc_indirect, enc_direct]
    models = ["baseline", "indirect", "direct"]
    fig, ax = plt.subplots()
    for idx, enc in enumerate(encodings):
        enc = enc[img_idx * TRANSITION_LENGTH:img_idx * TRANSITION_LENGTH + TRANSITION_LENGTH]
        ax.scatter(enc[:, 0], enc[:, 1], c=colors[idx], alpha=0.5, label=models[idx], marker=markers[idx])
        # mark the linear path between latent codes of z0 and z16 with a dashed line
        ax.plot([enc[0, 0], enc[-1, 0]], [enc[0, 1], enc[-1, 1]], color=colors[idx], marker=markers[idx],
                 linestyle='dashed', alpha=0.2)
        # annotate z0, a8 and z16
        for zstack, zstack_lbl in zip([0, 4, 8], ["z0", "z8", "z16"]):
            ax.annotate(zstack_lbl, (enc[zstack, 0], enc[zstack, 1]), fontsize="x-small")

    fig.legend()
    if output_path:
        save_to = os.path.join(output_path, f"{slide_type}-experiments", "2D_projections_comparison")
        os.makedirs(save_to, exist_ok=True)
    else:
        save_to = os.path.join(f"{slide_type}-experiments")
    fig.savefig(os.path.join(save_to, f"{img_idx}.png"))
    # fig.close()


def create_transition_directories(experiment_name: str, images: Tensor, slide_type):
    dir_path = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(dir_path, f"{slide_type}-experiments", experiment_name, f"transition-{experiment_name}-0"),
                exist_ok=True)
    os.makedirs(os.path.join(dir_path, f"{slide_type}-experiments", experiment_name, f"transition-{experiment_name}-1"),
                exist_ok=True)
    os.makedirs(os.path.join(dir_path, f"{slide_type}-experiments", experiment_name, f"transition-{experiment_name}-2"),
                exist_ok=True)
    for zs, z_stack in enumerate(images):
        for idx, image in enumerate(z_stack):
            # image = denormalize_image(image, experiment_name)
            save_image(image, os.path.join(f"{slide_type}-experiments", experiment_name,
                                           f"transition-{experiment_name}-{idx}", f"original_z{zs * 2}.tif"))


def grid_comparison_reconstructions(img_idx, slide_type, interpolated=False, output_path=""):
    """
    ** Used to generate Fig. 4(a) in the paper. **
    For a given image, generate a grid showing a transition from z-stack 0 to z-stack 16, with 4 rows: real images on
    the first row, reconstructions with the baseline, indirectly and directly regularized models on the following rows.
    The reconstructions can come from latent codes obtained by encoding the real images or from latent codes obtained
    from linear interpolation between other latent codes. This is controlled with the `interpolated` argument.

    :param img_idx: the idx of the image to be used for the grid.
    :param slide_type: the type of slide to which the image belongs.
    :param interpolated: if the grid should show reconstructions from original or linearly interpolated latent
     representations.
    """
    file_extension = "png"  # "pdf", "eps"
    models = [f"{slide_type}-baseline", f"{slide_type}-indirect", f"{slide_type}-direct"]
    images, _, _ = ds.test_images(slide_type)
    # select only the images (blur levels) associated with the cell slide with index `img_idx`
    to_viz = [images[img_idx * TRANSITION_LENGTH:img_idx * TRANSITION_LENGTH + TRANSITION_LENGTH]]
    save_to = os.path.join(output_path, f"{slide_type}-experiments", "gt_vs_synthetic_blur")
    # directory for saving a 1-row grid with the real blur levels of the cell slide
    os.makedirs(os.path.join(save_to, "gt_blur"), exist_ok=True)
    # directory for saving several 1-row grids with the synthetic blur levels of the cell slide, using the 3 models
    # one 1-row grid per model is generated
    os.makedirs(os.path.join(save_to, "synthetic_blur"), exist_ok=True)
    # save a grid of the transition from sharp to blurry with the real images
    grid = make_grid(to_viz, nrow=TRANSITION_LENGTH)
    save_image(grid, os.path.join(save_to, "gt_blur", f"gt_img{img_idx}.{file_extension}"))
    for model in models:
        path_to = os.path.join(output_path, f"{slide_type}-experiments", model)
        if not interpolated:
            reconstr = read_from_file(os.path.join(path_to, f"reconstructed_blur_{model}.pt"))
        else:
            reconstr = read_from_file(os.path.join(path_to, f"synthesised_blur_{model}.pt"))
        to_viz.append(reconstr[img_idx * TRANSITION_LENGTH:img_idx * TRANSITION_LENGTH + TRANSITION_LENGTH])
        grid = make_grid(reconstr[img_idx * TRANSITION_LENGTH:img_idx * TRANSITION_LENGTH + TRANSITION_LENGTH],
                         nrow=TRANSITION_LENGTH)
        if not interpolated:
            save_image(grid, os.path.join(save_to, "gt_blur", f"{model}_img{img_idx}.{file_extension}"))
        else:
            save_image(grid, os.path.join(save_to, "synthetic_blur", f"{model}_img{img_idx}.{file_extension}"))
    # save one 4-row grid as well from the 1-row grids generated above
    grid = make_grid(torch.cat(to_viz, dim=0), nrow=TRANSITION_LENGTH)
    if not interpolated:
        save_image(grid, os.path.join(save_to, f"{slide_type}_img{img_idx}_gt_blur.{file_extension}"))
    else:
        save_image(grid, os.path.join(save_to, f"{slide_type}_img{img_idx}_synthetic_blur.{file_extension}"))


def deblur_fixed_alpha(slide_type, alpha, img_idx, output_path=""):
    """
    ** Used to generate Fig. 4(c) in the paper. **
    Generates a 3-row grid to visualize how the 3 models synthesize a sharp image starting from the same input pair
    and using the same interpolation parameter alpha. Each row contains: a pair of input images, the synthetic sharp
    image, the sharp image as reconstructed by the model from its associated (non-interpolated) latent code and the real
    sharp image.
    :param slide_type: what type of slide the image is (w1 or w2).
    :param alpha: the interpolation parameter used for the deblurring.
    :param img_idx: the index of the zstack-0 version of the cell slide to be deblurred.
    """
    # alpha_range = [*range(1 / (TRANSITION_LENGTH - 1), 1, 1 / (TRANSITION_LENGTH - 1))]
    alpha_range = list(map(lambda val: val / (TRANSITION_LENGTH-1), [*range(1, (TRANSITION_LENGTH-2) + 1)]))
    # there are 154 test images for w1 slides and 153 for w2 slides
    img_range = [*range(img_idx, 1078, 154)] if slide_type == "w1" else [*range(img_idx, 1071, 153)]
    models = [f"{slide_type}-baseline", f"{slide_type}-indirect", f"{slide_type}-direct"]
    to_viz = []
    alpha_dict = dict(zip(alpha_range, img_range))
    i = alpha_dict[alpha]
    sharp_gt = read_from_file(os.path.join(output_path, f"{slide_type}-experiments", f"deblurred_gt_{slide_type}.pt"))
    # save a 1-row grid per model type
    for model in models:
        path_to = os.path.join(output_path, f"{slide_type}-experiments", model)
        try:
            generated_sharp = read_from_file(os.path.join(path_to, f"generated_sharp_{model}.pt"))
            reconstructed_sharp = read_from_file(os.path.join(path_to, f"reconstructed_sharp_{model}.pt"))
            left_rec = read_from_file(os.path.join(path_to, f"left_rec_by_{model}.pt"))
            right_rec = read_from_file(os.path.join(path_to, f"right_rec_by_{model}.pt"))
            to_viz.extend([left_rec[i], right_rec[i], generated_sharp[i], reconstructed_sharp[i], sharp_gt[i]])
            model_grid = make_grid([left_rec[i], right_rec[i], generated_sharp[i], reconstructed_sharp[i], sharp_gt[i]],
                             nrow=5)
            save_image(model_grid, os.path.join(output_path, f"{slide_type}-experiments",
                                              f"deblur-model-comparison-{model}-alpha-{alpha}-img{img_idx}.png"))
        except FileNotFoundError as e:
            print(e)
    # save a 3-row grid using the 1-row grids generated above
    img_grid = make_grid(torch.stack(to_viz, dim=0), nrow=5)
    save_image(img_grid, os.path.join(output_path, f"{slide_type}-experiments",
                                      f"deblur-model-comparison-{slide_type}-alpha-{alpha}-img{img_idx}.png"))


def deblur_visualisations_vary_alpha(exp_name, slide_type, img_idx, output_path=""):
    """
    ** Generates Fig. 5 in the paper. **
    Generates a 5-row grid to show how using different alpha parameters and input pairs affects the synthetic sharp
    image, using a specified model (based on the experiment name).
    The rows in the grid correspond to: input pair (first 2 rows), synthetic sharp image, reconstructed sharp image and
    real sharp image. The columns correspond to different input pairs, where one image is fixed at z-stack 0 and the
    other varies from z-stack 2 to z-stack 14 (from blurry to sharper).

    :param expe_name: the name of the experiment (model)
    :param slide_type: the type of slide to which the image belongs
    :param img_idx: the idx of the cell slide to be used for the grid
    """
    try:
        sharp_gt = read_from_file(
            os.path.join(output_path, f"{slide_type}-experiments", f"deblurred_gt_{slide_type}.pt"))
        generated_sharp = read_from_file(
            os.path.join(output_path, f"{slide_type}-experiments", exp_name, f"generated_sharp_{exp_name}.pt"))
        reconstructed_sharp = read_from_file(
            os.path.join(output_path, f"{slide_type}-experiments", exp_name, f"reconstructed_sharp_{exp_name}.pt"))
        left_rec = read_from_file(os.path.join(output_path, f"{slide_type}-experiments", exp_name,
                                               f"left_rec_by_{exp_name}.pt"))
        right_rec = read_from_file(os.path.join(output_path, f"{slide_type}-experiments", exp_name,
                                                f"right_rec_by_{exp_name}.pt"))
        images = [left_rec, right_rec, generated_sharp, reconstructed_sharp, sharp_gt]
        to_viz = []
        img_range = [*range(img_idx, 1078, 154)] if slide_type == "w1" else [*range(img_idx, 1071, 153)]
        for img_tensors in images:
            for i in img_range:
                to_viz.append(img_tensors[i])
        grid = make_grid(torch.stack(to_viz, dim=0), nrow=TRANSITION_LENGTH - 2)
        os.makedirs(os.path.join(output_path, f"{slide_type}-experiments", "deblurring_examples", exp_name), exist_ok=True)
        save_image(grid, os.path.join(output_path, f"{slide_type}-experiments", "deblurring_examples", exp_name,
                                      f"img{img_idx}.png"))
    except FileNotFoundError as e:
        print(f"File not found for {exp_name}. Error: {traceback.print_exc()}")


def deblur_visualisations_vary_alpha_transposed(exp_name, slide_type, img_idx, output_path=""):
    """
    Generates the same plot as `deblur_visualisations_vary_alpha()`, bur transposed.
    Generates a grid to visualize how using different alpha parameters and input pairs affects the synthetic sharp
    image, using a specified model (based on the experiment name).

    :param expe_name: the name of the experiment (model)
    :param slide_type: the type of slide to which the image belongs
    :param img_idx: the idx of the image to be used for the grid
    """
    try:
        sharp_gt = read_from_file(
            os.path.join(output_path, f"{slide_type}-experiments", f"deblurred_gt_{slide_type}.pt"))
        generated_sharp = read_from_file(
            os.path.join(output_path, f"{slide_type}-experiments", exp_name, f"generated_sharp_{exp_name}.pt"))
        reconstructed_sharp = read_from_file(
            os.path.join(output_path, f"{slide_type}-experiments", exp_name, f"reconstructed_sharp_{exp_name}.pt"))
        left_rec = read_from_file(os.path.join(output_path, f"{slide_type}-experiments", exp_name,
                                               f"left_rec_by_{exp_name}.pt"))
        right_rec = read_from_file(os.path.join(output_path, f"{slide_type}-experiments", exp_name,
                                                f"right_rec_by_{exp_name}.pt"))
        images = [left_rec, right_rec, generated_sharp, reconstructed_sharp, sharp_gt]
        img_range = [*range(img_idx, 1078, 154)] if slide_type == "w1" else [*range(img_idx, 1071, 153)]
        for i in img_range:
            to_viz = []
            for img_tensors in images:
                to_viz.append(img_tensors[i])
            grid = make_grid(torch.stack(to_viz, dim=0), nrow=len(to_viz))
            os.makedirs(os.path.join(output_path, f"{slide_type}-experiments", "deblurring_examples_transposed",
                                     exp_name), exist_ok=True)
            save_image(grid, os.path.join(output_path, f"{slide_type}-experiments", "deblurring_examples_transposed",
                                          exp_name, f"img{img_idx}-{i}.png"))
    except FileNotFoundError as e:
        print(f"File not found for {exp_name}. Error: {traceback.print_exc()}")


def process_encodings(encodings, save_to):
    """
    Given some latent representations, project them to a 2D space and generate plots of them. A plot for each image is
    generated, as well as a plot of all the images together.
    :param encodings: the latent representations to be plotted after dimensionality reduction. Formed of sets of
    `TRANSITION_LENGTH` images, where each set is a transition from z0 to z16 of one image.
    :param save_to: the location where the plots are saved
    :return:
    """
    # apply PCA to project the latent representations to a 2D space
    encodings_reduced = project_encodings_to_2d(encodings)
    min_x, max_x = np.min(encodings_reduced[:, 0]), np.max(encodings_reduced[:, 0])
    min_y, max_y = np.min(encodings_reduced[:, 1]), np.max(encodings_reduced[:, 1])
    # Generate 2D plots of the latent representations, where each plot displays the encodings of 1 image
    # (its `TRANSITION_LENGTH` transitions from z0 to z16)
    generate_plots(encodings_reduced, TRANSITION_LENGTH, save_to, [min_x, max_x, min_y, max_y])
    # generate a 2D plot of all the latent representations
    # generate_plots(encodings_reduced, None, save_to, None)


def generate_plots(encodings, images_per_plot, save_to, plot_axes_limits=None, slide_type=None):
    """
    Generates a plot / multiple plots given some latent representations
    :param encodings: the latent codes to be used for plotting
    :param images_per_plot: how many images to be in each plot (TRANSITION_LENGTH for plots displaying a transition
    from z0 to z16 of one image, None for a plot with all the latent codes provided).
    :param save_to: where the plots will be saved
    :param plot_axes_limits: if given, the min and max values for the axes of the plots
    :param slide_type: the type of slide to which the image representations belong
    """
    img_cnt = encodings.shape[0] // 2  # separate the original and interpolated representations when counting
    save_to_dir = os.path.join(save_to, "pca_latent_codes")
    os.makedirs(save_to_dir, exist_ok=True)
    enc_orig = encodings[:img_cnt]
    enc_interp = encodings[img_cnt:]
    if slide_type is not None:
        scatter_name = slide_type
    else:
        scatter_name = ""
    if images_per_plot is None:
        # create a plot with all the given latent codes
        save_to = os.path.join(save_to_dir, scatter_name + f"original_latent_codes.png")
        colors = list(
            chain.from_iterable([[i] * TRANSITION_LENGTH for i in range(img_cnt // 9)]))  # unique color for each image
        plot_embeddings(enc_orig, save_to, colors, plot_axes_limits)
        save_to = os.path.join(save_to_dir, scatter_name + f"interpolated_latent_codes.png")
        plot_embeddings(enc_interp, save_to, colors, plot_axes_limits)
        return
    for idx, i in enumerate([*range(0, len(enc_orig), images_per_plot)]):
        # for each set of `images_per_plot` images create a plot
        last_img_idx = i + images_per_plot
        enc_to_plot = np.append(enc_orig[i:last_img_idx], enc_interp[i:last_img_idx], axis=0)
        scatter_name = f"{idx}"
        save_to = os.path.join(save_to_dir, f"{scatter_name}.png")
        colors = list(chain.from_iterable([[i] * TRANSITION_LENGTH for i in range(2)]))
        plot_embeddings(enc_to_plot, save_to, colors, plot_axes_limits)
        # print the original latent representations, but give each z-stack a different color
        # scatter_name += "-stacks"
        # save_to = os.path.join(save_to_dir, f"{scatter_name}.png")
        # plot_embeddings(enc_orig[i:last_img_idx], save_to, None, plot_axes_limits)


def plot_embeddings(embeddings, save_to, colors, plot_axes_limits):
    """
    Creates a scatter plot from the given 2D array.
    :param embeddings: the 2D array to be plotted.
    :param save_to: the location where the plot should be saved.
    :param colors: the colors used to differentiate between the plotted points.
    :param plot_axes_limits: if provided, used to set the min and max values for the plot axes
    """
    plt.figure()
    if colors is None:
        # each point will have a new color (used to plot points based on z-stack level)
        colors = [*range(TRANSITION_LENGTH)]
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=colors, cmap="plasma")
    if plot_axes_limits is not None:
        plt.xlim(plot_axes_limits[0], plot_axes_limits[1])
        plt.ylim(plot_axes_limits[2], plot_axes_limits[3])
    plt.savefig(save_to)
    plt.close()


if __name__ == "__main__":
    for image_idx in range(100):
        deblur_visualisations_vary_alpha("w1-baseline", "w1", image_idx)
        deblur_visualisations_vary_alpha("w1-indirect", "w1", image_idx)
