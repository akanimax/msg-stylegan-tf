""" Demo script for running Random-latent space interpolation on the trained MSG-StyleGAN OR
    Show the effect of stochastic noise on a fixed image """

import argparse
import os
import pickle
from math import sqrt
from pathlib import Path

import dnnlib.tflib as tflib
import imageio
import numpy as np
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser("MSG-StyleGAN image_generator")
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
        default=33,
        help="random_state (seed) for the script to run",
    )

    parser.add_argument(
        "--out_depth",
        action="store",
        type=int,
        default=None,
        help="output depth of the generated images",
    )

    parser.add_argument(
        "--num_samples",
        action="store",
        type=int,
        default=100,
        help="Number of samples to be generated",
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


def get_image(gen, point, out_depth):
    """
    obtain an All-resolution grid of images from the given point
    :param gen: the generator object
    :param point: random latent point for generation
    :param out_depth: depth of network from where the images are to be generated
    :return: img => generated image
    """
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    point = np.expand_dims(point, axis=0)
    images = gen.run(
        point, None, truncation_psi=0.6, randomize_noise=True, output_transform=fmt
    )
    if out_depth is None or out_depth >= len(images):
        out_depth = -1
    return np.squeeze(images[out_depth])


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
        image = get_image(Gs, point, args.out_depth)
        imageio.imwrite(os.path.join(output_path, str(count) + ".png"), image)

    print(f"Requested images have been generated at: {output_path}")


if __name__ == "__main__":
    main(parse_arguments())
