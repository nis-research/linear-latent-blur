from torchmetrics import PeakSignalNoiseRatio as PSNR
import torch.nn.functional as F
import torch
import numpy as np
from models.config import TRANSITION_LENGTH


def eval_psnr(y_hat, target):
    psnr = PSNR()
    return psnr(y_hat, target)


def compute_lds(encodings, interpolations):
    """
    Computes the LDS score, as defined in the paper.
    :param encodings: the latent codes of the original images.
    :param interpolations: the latent codes of the original images obtained through linear interpolation.
    :return: the LDS score for the set of given latent codes.
    """
    if len(encodings.shape) != 2:
        encodings = encodings.reshape(encodings.shape[0], encodings.shape[1] * encodings.shape[2] * encodings.shape[3])
        interpolations = interpolations.reshape(interpolations.shape[0],
                                                interpolations.shape[1] * interpolations.shape[2] *
                                                interpolations.shape[3])
    img_cnt = len(encodings) // TRANSITION_LENGTH
    cos_sim = torch.nn.CosineSimilarity(dim=1)
    result = 0
    for i in range(0, len(interpolations), TRANSITION_LENGTH):
        # iterate over each set of TRANSITION_LENGTH latent representations corresponding to one image and compute the
        # average pairwise cosine similarity between an interpolated latent code and its corresponding original latent code
        oe = encodings[i + 1:i + TRANSITION_LENGTH - 1]
        ie = interpolations[i + 1:i + TRANSITION_LENGTH - 1]
        result += torch.sum(cos_sim(oe, ie)) / (TRANSITION_LENGTH-2)  # we only consider the intermediate points, so we divide by 7 for each
        # set of TRANSITION_LENGTH images
    return result / img_cnt


def compute_apd(encodings, interpolations):
    """
    Computed the APD score for a set of latent representations, as defined in the paper.
    :param encodings: the latent codes of the original images.
    :param interpolations: the latent codes of the original images obtained through linear interpolation.
    :return: the APD score for the given set of latent codes.
    """
    if len(encodings.shape) != 2:
        encodings = encodings.reshape(encodings.shape[0], encodings.shape[1] * encodings.shape[2] * encodings.shape[3])
        interpolations = interpolations.reshape(interpolations.shape[0],
                                                interpolations.shape[1] * interpolations.shape[2] *
                                                interpolations.shape[3])
    img_cnt = len(encodings) // TRANSITION_LENGTH
    result = 0
    for i in range(0, len(interpolations), TRANSITION_LENGTH):
        img_avg_dist = 0
        for j in range(i, i + TRANSITION_LENGTH - 1):
            # compute the euclidian distance between pairs of neighbouring latent representations
            dist_orig = F.mse_loss(encodings[j], encodings[j + 1])
            dist_interp = F.mse_loss(interpolations[j], interpolations[j + 1])
            # compute the absolute difference between these distances
            img_avg_dist += np.abs(dist_orig - dist_interp)
        # average the result over the 8 pairs of latent codes corresponding to one transition from z0 to z16
        result += img_avg_dist / (TRANSITION_LENGTH-1)
    return result / img_cnt
