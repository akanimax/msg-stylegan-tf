""" Demo script for running Random-latent space interpolation on the trained MSG-StyleGAN OR
    Show the effect of stochastic noise on a fixed image (for all multiple scales)"""

import argparse
import os
import pickle
from math import sqrt
from pathlib import Path

import dnnlib.tflib as tflib
import imageio
import numpy as np
import torch
import torchvision as tv
from tqdm import tqdm
from training.misc import dumb_upsample_nn


def parse_arguments():
    parser = argparse.ArgumentParser("MSG-StyleGAN Multiscale image_generator")
    parser.add_argument(
        "--pickle_file",
        type=str,
        required=True,
        action="store",
        help="pickle file containing the trained styleGAN model",
    )

    parser.add_argument(
        "--output_path",
        action="store",
        type=str,
        default=None,
        required=True,
        help="Path to directory for saving the files",
    )

    parser.add_argument(
        "--random_state",
        action="store",
        type=int,
        default=3,
        help="random_state (seed) for the script to run",
    )

    parser.add_argument(
        "--num_cols",
        action="store",
        type=int,
        default=None,
        help="number of rows in the required grid",
    )

    parser.add_argument(
        "--num_samples",
        action="store",
        type=int,
        default=100,
        help="Number of samples to be generated",
    )

    parser.add_argument(
        "--truncation_psi",
        action="store",
        type=float,
        default=0.6,
        help="value of truncation psi for image generation",
    )

    parser.add_argument(
        "--alternate_reso",
        action="store",
        type=bool,
        default=False,
        help="Whether to visualize only alternate resolutions in the multi-scale generation",
    )

    parser.add_argument(
        "--only_noise",
        action="store",
        type=bool,
        default=False,
        help="to visualize the same point with only different realizations of noise",
    )

    args = parser.parse_args()
    return args


def get_image(gen, point, num_cols, trunc_psi=0.6, alternate_reso=False):
    """
    obtain an All-resolution grid of images from the given point
    :param gen: the generator object
    :param point: random latent point for generation
    :param num_cols: depth of network from where the images are to be generated
    :param trunc_psi: value of the truncation psi used for generating the images
    :param alternate_reso: whether to visualize only alternate resolutions in the multi-scale generation
    :return: img => generated image
    """
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    point = np.expand_dims(point, axis=0)
    images = gen.run(
        point,
        None,
        truncation_psi=trunc_psi,
        randomize_noise=True,
        output_transform=fmt,
    )
    highest_res_log_2 = int(np.log2(images[-1].shape[1]))
    images = [
        dumb_upsample_nn(
            np.transpose(image, (0, 3, 1, 2)),
            int(2 ** (highest_res_log_2 - int(np.log2(image.shape[1])))),
        )
        for image in images
    ]

    if alternate_reso:
        new_images = []
        for num, image in enumerate(reversed(images)):
            if num % 2 == 0:
                new_images.append(image)
        images = list(reversed(new_images))

    if num_cols is None:
        num_cols = len(images)  # default behaviour is to create a horizontal sequence
    if num_cols == 1:
        # for a vertical tower, reverse the order of the images
        images = list(reversed(images))
    images = np.concatenate(images, 0)  # concatenate the images

    multi_scale_image_grid = tv.utils.make_grid(
        torch.tensor(images), nrow=num_cols
    ).numpy()
    multi_scale_image_grid = np.transpose(multi_scale_image_grid, (1, 2, 0))

    return multi_scale_image_grid


def main(args):
    # Initialize TensorFlow.
    tflib.init_tf()

    # Load pre-trained network.
    with open(args.pickle_file, "rb") as f:
        _, _, Gs = pickle.load(f)
        # _  = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
        # _  = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
        # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.

    # Print network details.
    print("\n\nLoaded the Generator as:")
    Gs.print_layers()

    # Pick latent vector.
    latent_size = Gs.input_shape[1]
    rnd = np.random.RandomState(args.random_state)

    # create the random latent_points for the interpolation
    total_samples = args.num_samples
    all_latents = rnd.randn(total_samples, latent_size)
    all_latents = (
        all_latents / np.linalg.norm(all_latents, axis=-1, keepdims=True)
    ) * sqrt(latent_size)

    # animation mechanism
    start_point = np.expand_dims(all_latents[0], axis=0)
    points = all_latents[1:]

    # all points are start_point, if we have only noise realization
    if args.only_noise:
        points = np.array([np.squeeze(start_point) for _ in points])

    # make sure that the output path exists
    output_path = Path(args.output_path)
    output_path.mkdir(exist_ok=True)

    print("Generating the requested number of samples ... ")
    for count, point in tqdm(enumerate(points, 1)):
        image = get_image(
            Gs, point, args.num_cols, args.truncation_psi, args.alternate_reso
        )
        imageio.imwrite(os.path.join(output_path, str(count) + ".png"), image)

    print(f"Requested images have been generated at: {output_path}")


if __name__ == "__main__":
    main(parse_arguments())
