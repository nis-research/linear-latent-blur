import logging
import torch
import os
import torchvision.utils
from torchvision.utils import save_image
from sklearn.decomposition import PCA
from config import TRANSITION_LENGTH, STD_W1, MEAN_W1, MEAN_W2, STD_W2, NORMALIZE_IMAGES


def init_logger(experiment_name):
    test_logger = logging.getLogger(f"{experiment_name}_logger")
    test_logger.setLevel(logging.DEBUG)
    os.makedirs("logs", exist_ok=True)
    f_handler = logging.FileHandler(os.path.join("logs", f"{experiment_name}.log"))
    f_handler.setLevel(logging.DEBUG)
    test_logger.addHandler(f_handler)
    return test_logger


def merge_batch_crops(images):
    bs, nrcrops, c, h, w = images.size()
    return images.view(-1, c, h, w)


def prepare_batch(batch, pretrain=False):
    """
    Prepares a batch of images, by stacking the 10 crops obtained from each image.
    """
    if pretrain:
        return merge_batch_crops(batch)
    left, right, target = batch
    return merge_batch_crops(left), merge_batch_crops(right), merge_batch_crops(target)


def generate_interpolations_test(encodings, steps=TRANSITION_LENGTH - 1):
    """
    Generates interpolated image representations by linearly traversing the latent space between two given latent codes.
    Generates `steps`-1 representations.
    :param encodings: the original encodings of images. Each set of `TRANSITION_LENGTH` consecutive images belong to the
    same transition.
    :param steps: parameter chosen such that the interpolation step is 1/steps.
    :return: returns a stack of tensors where each consecutive `TRANSITION_LENGTH` tensors are latent representations
    belonging to the same transition from z0 to z16.
    """
    interpolations = []
    for idx, i in enumerate([*range(0, len(encodings), TRANSITION_LENGTH)]):
        enc_left, enc_right = encodings[i], encodings[i + TRANSITION_LENGTH - 1]
        interpolations.append(enc_left)
        alpha = 1 / steps
        while alpha < 1:
            # if alpha == 1/8:  # fixme: delete this later
            #     alpha = 1/10
            interp = (1 - alpha) * enc_left + alpha * enc_right
            interpolations.append(interp)
            # if alpha == 1/10:  # fixme: delete this later
            #     alpha = 1/8
            alpha += 1 / steps
        interpolations.append(enc_right)
    interpolations = torch.stack(interpolations, dim=0)
    return interpolations


def interpolate_for_deblur(left_encodings, right_encodings, alphas):
    """
    Performs a linear interpolation operation between two given latent representations, based on the interpolation for
    deblurring formula.
    :param left_encodings: The encodings with lower z-stack level (higher blur level)
    :param right_encodings: The encodings with higher z-stack level (lower blur level)
    :param alphas: a list with the interpolation parameters to be used for each pair of latent codes
    :return: a tensor with interpolated latent representations corresponding to sharp images
    """
    interpolations = []
    print(alphas)
    for i in range(len(alphas)):
        # interpolations.append(
        #     1 / (1 - alphas[i]) * right_encodings[i] - ((alphas[i]) / (1 - alphas[i])) * left_encodings[i])
        inverted_alpha = 1-alphas[i]
        # if inverted_alpha == 1/8:  # fixme: delete this later
        #     inverted_alpha = 1/10
        interpolations.append(
            (1 / inverted_alpha) * right_encodings[i] - ((1-inverted_alpha) / inverted_alpha) * left_encodings[i])
    return torch.stack(interpolations, dim=0)


def denormalize_image(image, experiment_name):
    """
    Reverses the normalization of an image, using the data set precomputed statistics.
    :param image: the image to be denormalized.
    :param experiment_name: the experiment_name.
    :return: the denormalized image.
    """
    low = float(image.min())
    high = float(image.max())
    image.clamp_(min=low, max=high)
    return image.sub_(low).div_(max(high - low, 1e-5))
    # return image * STD_W1 + MEAN_W1 if "w1" in experiment_name else image * STD_W2 + MEAN_W2


def create_img_folder(reconstructions, exp_name, slide_type, interpolated=False, save_to=None):
    """
    Creates a folder and saves to its location the given images (`reconstructions`).
    :param reconstructions: the images to be saved in the created folder.
    :param exp_name: the name of the experiment.
    :param interpolated: if the saved images are reconstructions of interpolated latent representations.
    :return:
    """
    if interpolated:
        dir_name = f"{exp_name}-interpolated"
    else:
        dir_name = f"{exp_name}-original"
    if save_to:
        os.makedirs(os.path.join(save_to, f"{slide_type}-experiments", exp_name, dir_name), exist_ok=True)
    else:
        os.makedirs(os.path.join(save_to, f"{slide_type}-experiments", exp_name, dir_name), exist_ok=True)
    for idx, reconstr in enumerate(reconstructions):
        reconstr = reconstr if not NORMALIZE_IMAGES else denormalize_image(reconstr, exp_name)
        save_image(reconstr, os.path.join(save_to, f"{slide_type}-experiments", exp_name, dir_name, f"{idx}.png"))


def orig_and_interp(experiment_name, slide_type):
    """
    For a given experiment, returns the reconstructed original and generated images based on the test set.
    The reconstructions are read from a file, where the decoded images were saved as tensors.
    :param experiment_name: the name of the experiment
    :param slide_type: the type of slides in this experiment
    :return: the original and generated images decoded by the model
    """
    reconstructed_blur = torch.load(os.path.join(f"{slide_type}-experiments", experiment_name,
                                                 f"reconstructed_blur_{experiment_name}.pt"),
                                    map_location="cpu")
    generated_blur = torch.load(os.path.join(f"{slide_type}-experiments", experiment_name,
                                             f"synthesised_blur_{experiment_name}.pt"), map_location="cpu")
    return reconstructed_blur, generated_blur


# def visualizations_2d_embeddings(slide_type):
#     """
#     !!! old version, not used for paper
#     Creates visualizations of 2D projections of the latent representations for a set of chosen images (defined in the
#     `imgs` list).
#     :param slide_type: the type of slides (w1 or w2)
#     """
#     source_dir1 = os.path.join(f"{slide_type}-experiments", f"baseline-64-1024-{slide_type}", "pca_latent_embeddings")
#     source_dir3 = os.path.join(f"{slide_type}-experiments", f"strong-64-1024-{slide_type}", "pca_latent_embeddings")
#     source_dir2 = os.path.join(f"{slide_type}-experiments", f"weak-64-1024-{slide_type}", "pca_latent_embeddings")
#     imgs = [6, 14, 22, 52] if slide_type == "w1" else [9, 12, 16, 36]
#     to_viz_baseline, to_viz_weak, to_viz_strong = [], [], []
#     for img in imgs:
#         baseline_img = transforms.ToTensor()(Image.open(os.path.join(source_dir1, f"{img}.png")))
#         weak_img = transforms.ToTensor()(Image.open(os.path.join(source_dir2, f"{img}.png")))
#         strong_img = transforms.ToTensor()(Image.open(os.path.join(source_dir3, f"{img}.png")))
#         to_viz_baseline.append(baseline_img)
#         to_viz_weak.append(weak_img)
#         to_viz_strong.append(strong_img)
#     to_viz_baseline.extend(to_viz_weak)
#     to_viz_baseline.extend(to_viz_strong)
#     img_grid = make_grid(to_viz_baseline, nrow=len(imgs))
#     save_image(img_grid, f"2d-embeddings-comparison-{slide_type}.png")


def project_encodings_to_2d(encodings):
    """
    Applies PCA to the provided latent representations to project them to a 2D space.
    :param encodings: the data to be projected to a 2D space.
    :return: the data after dimensionality reduction is applied to it.
    """
    if len(encodings.shape) != 2:
        encodings = encodings.reshape(encodings.shape[0],
                                      encodings.shape[1] * encodings.shape[2] * encodings.shape[3]).numpy()
    else:
        encodings = encodings.numpy()
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(encodings)
