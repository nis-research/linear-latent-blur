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


def _read_latent_codes(experiment_name, slide_type):
    return torch.load(os.path.join(f"{slide_type}-experiments", experiment_name, f"latent_codes_{experiment_name}-"
                                                                                 f"{slide_type}.pt"), map_location="cpu")


def read_from_file(filename):
    return torch.load(filename, map_location="cpu")


def visualize_2d_projections(img_idx, slide_type):
    """
    Creates a plot of the 2D projected latent representations of z0 to z16 corresponding to the chosen image.
    *Used to generate image displayed in the paper.*
    :param img_idx: the index of the image to be used
    :param slide_type: the type of slides to which the image belongs
    """
    enc_baseline = project_encodings_to_2d(_read_latent_codes("baseline-64-1024", slide_type))
    enc_weak = project_encodings_to_2d(_read_latent_codes("weak-64-1024", slide_type))
    enc_strong = project_encodings_to_2d(_read_latent_codes("strong-64-1024", slide_type))
    markers = ["o", "v", "s"]
    colors = ["blue", "red", "green", "purple", "yellow", "pink", "violet", "orange", "brown", "grey", "black"]
    encodings = [enc_baseline, enc_weak, enc_strong]
    models = ["baseline", "weak", "strong"]
    for idx, enc in enumerate(encodings):
        enc = enc[img_idx*TRANSITION_LENGTH:img_idx*TRANSITION_LENGTH+TRANSITION_LENGTH]
        plt.scatter(enc[:, 0], enc[:, 1], c=colors[idx], alpha=0.5, label=models[idx], marker=markers[idx])
        # mark the linear path between latent codes of z0 and z16 with a dashed line
        plt.plot([enc[0, 0], enc[-1,0]], [enc[0,1], enc[-1,1]], color=colors[idx], marker=markers[idx],
                 linestyle='dashed', alpha=0.2)
    plt.legend()
    save_to = os.path.join(f"{slide_type}-experiments")
    plt.savefig(os.path.join(save_to, f"2D-projections-{slide_type}-{img_idx}.png"))
    plt.close()


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


def grid_comparison_reconstructions(img_idx, slide_type, interpolated=False):
    """
    For a given image, generate a grid showing a transition from z0 to z16, with 4 rows: original images,
    reconstructions with the baseline, weak and strong regularization.
    :param img_idx: the idx of the image to be used for the grid
    :param slide_type: the type of slide to which the image belongs
    :param interpolated: if the grid should show reconstructions from original or interpolated latent representations
    """
    file_extension = "png"  # "pdf", "eps"
    models = [f"baseline-64-1024-{slide_type}", f"weak-64-1024-{slide_type}", f"strong-64-1024-{slide_type}"]
    images, _, _ = ds.test_images(slide_type)
    to_viz = [images[img_idx * TRANSITION_LENGTH:img_idx * TRANSITION_LENGTH + TRANSITION_LENGTH]]
    for model in models:
        if not interpolated:
            reconstr = read_from_file(os.path.join(f"{slide_type}-experiments", model, f"reconstructed_blur_{model}.pt"))
        else:
            reconstr = read_from_file(os.path.join(f"{slide_type}-experiments", model, f"generated_blur_{model}.pt"))
        to_viz.append(reconstr[img_idx * TRANSITION_LENGTH:img_idx * TRANSITION_LENGTH + TRANSITION_LENGTH])
    grid = make_grid(torch.cat(to_viz, dim=0), nrow=TRANSITION_LENGTH)
    if not interpolated:
        save_image(grid, os.path.join(f"{slide_type}-experiments", f"Comparison reconstructions for {slide_type} "
                                                                   f"img{img_idx}.{file_extension}"))
    else:
        save_image(grid, os.path.join(f"{slide_type}-experiments", f"Comparison interpolated reconstructions for "
                                                                   f"{slide_type} img{img_idx}.{file_extension}"))


def deblur_fixed_alpha(slide_type, alpha, img_idx):
    """
    Generates a grid to visualize how the chosen models deblur the same image, using the same alpha parameter.
    :param slide_type: what type of slide is the image
    :param alpha: the interpolation parameter used for the deblurring
    :param img_idx: the index of the z0 version of the image to be deblurred
    """
    alpha_range = [*range(1/(TRANSITION_LENGTH-1), 1, 1/(TRANSITION_LENGTH-1))]
    # there are 154 test images for w1 slides and 153 for w2 slides
    img_range = [*range(img_idx, 1078, 154)] if slide_type == "w1" else [*range(img_idx, 1071, 153)]
    models = [f"baseline-64-1024-{slide_type}", f"weak-64-1024-{slide_type}", f"strong-64-1024-{slide_type}"]
    to_viz = []
    alpha_dict = dict(zip(alpha_range, img_range))
    i = alpha_dict[alpha]
    sharp_gt = read_from_file(os.path.join(f"{slide_type}-experiments", f"deblurred_gt_{slide_type}.pt"))
    for model in models:
        try:
            generated_sharp = read_from_file(os.path.join(f"{slide_type}-experiments", model, f"generated_sharp_{model}.pt"))
            reconstructed_sharp = read_from_file(os.path.join(f"{slide_type}-experiments", model, f"reconstructed_sharp_{model}.pt"))
            left_rec = read_from_file(os.path.join(f"{slide_type}-experiments", model, f"left_rec_by_{model}.pt"))
            right_rec = read_from_file(os.path.join(f"{slide_type}-experiments", model, f"right_rec_by_{model}.pt"))
            to_viz.extend([left_rec[i], right_rec[i], generated_sharp[i], reconstructed_sharp[i], sharp_gt[i]])
        except FileNotFoundError:
            print("File not found")
    img_grid = make_grid(to_viz, nrow=5)
    save_image(img_grid, os.path.join(f"{slide_type}-experiments", f"deblur-model-comparison-{slide_type}-alpha-{alpha}-img{img_idx}.png"))


def deblur_visualisations_vary_alpha(exp_name, slide_type, img_idx):
    """
    Generates a grid to visualize how the same target image is deblurred using different alpha parameters,
    by the chosen model.
    :param expe_name: the name of the experiment (model)
    :param slide_type: the type of slide to which the image belongs
    :param img_idx: the idx of the image to be used for the grid
    """
    try:
        sharp_gt = read_from_file(os.path.join(f"{slide_type}-experiments", f"deblurred_gt_{slide_type}.pt"))
        generated_sharp = read_from_file(
            os.path.join(f"{slide_type}-experiments", exp_name, f"generated_sharp_{exp_name}.pt"))
        reconstructed_sharp = read_from_file(
            os.path.join(f"{slide_type}-experiments", exp_name, f"reconstructed_sharp_{exp_name}.pt"))
        left_rec = read_from_file(os.path.join(f"{slide_type}-experiments", exp_name, f"left_rec_by_{exp_name}.pt"))
        right_rec = read_from_file(os.path.join(f"{slide_type}-experiments", exp_name, f"right_rec_by_{exp_name}.pt"))
        images = [left_rec, right_rec, generated_sharp, reconstructed_sharp, sharp_gt]
        to_viz = []
        # TODO: probably the line below needs to be updated with new values
        img_range = [*range(img_idx, 1078, 154)] if slide_type == "w1" else [*range(img_idx, 1071, 153)]
        for img_tensors in images:
            for i in img_range:
                to_viz.append(img_tensors[i])
        grid = make_grid(torch.stack(to_viz, dim=0), nrow=TRANSITION_LENGTH-2)
        os.makedirs(os.path.join("deblurring_examples", exp_name), exist_ok=True)
        save_image(grid, os.path.join("deblurring_examples", exp_name, f"img{img_idx}.png"))
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
    encodings_reduced = project_encodings_to_2d(encodings)  # apply PCA to project the latent representations to a 2D space
    min_x, max_x = np.min(encodings_reduced[:, 0]), np.max(encodings_reduced[:, 0])
    min_y, max_y = np.min(encodings_reduced[:, 1]), np.max(encodings_reduced[:, 1])
    # Generate 2D plots of the latent representations, where each plot displays the encodings of 1 image
    # (its `TRANSITION_LENGTH` transitions from z0 to z16)
    generate_plots(encodings_reduced, TRANSITION_LENGTH, save_to, [min_x, max_x, min_y, max_y])
    # generate a 2D plot of all the latent representations
    generate_plots(encodings_reduced, None, save_to, None)


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
        colors = list(chain.from_iterable([[i] * TRANSITION_LENGTH for i in range(img_cnt//9)]))  # unique color for each image
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
        deblur_visualisations_vary_alpha("w1-indirect-reg", "w1", image_idx)
