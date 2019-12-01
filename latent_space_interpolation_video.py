""" Demo script for running Random-latent space interpolation on the trained StyleGAN OR
    Show the effect of stochastic noise on a fixed image """

import argparse
import pickle
from math import sqrt

import cv2
import dnnlib.tflib as tflib
import numpy as np
from scipy.misc import imresize
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser("StyleGAN image_generator")
    parser.add_argument(
        "--pickle_file",
        type=str,
        required=True,
        action="store",
        help="pickle file containing the trained styleGAN model",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        required=False,
        default="latent_space_exploration.mpeg",
        action="store",
        help="output video file",
    )

    parser.add_argument(
        "--random_state",
        action="store",
        type=int,
        default=5,
        help="random_state (seed) for the script to run",
    )

    parser.add_argument(
        "--num_points",
        action="store",
        type=int,
        default=12,
        help="Number of samples to be seen",
    )

    parser.add_argument(
        "--transition_points",
        action="store",
        type=int,
        default=60,
        help="Number of transition samples for interpolation. Can also be considered as fps",
    )

    parser.add_argument(
        "--generation_depths",
        action="append",
        default=None,
        nargs="+",
        required=False,
        help="Resolutions used for generating the interpolation",
    )

    parser.add_argument(
        "--resize",
        action="store",
        default=None,
        nargs=2,
        required=False,
        help="Resolutions used for generating the interpolation",
    )

    parser.add_argument(
        "--num_cols",
        action="store",
        type=int,
        default=None,
        help="number of cols in the generated video (used only if generation_depths are provided)",
    )

    parser.add_argument(
        "--smoothing",
        action="store",
        type=float,
        default=1.0,
        help="amount of transitional smoothing",
    )

    parser.add_argument(
        "--only_noise",
        action="store",
        type=bool,
        default=False,
        help="to visualize the same point with only different realizations of noise",
    )

    parser.add_argument(
        "--truncation_psi",
        action="store",
        type=float,
        default=0.6,
        help="value of truncation_psi used for generating the video",
    )

    args = parser.parse_args()
    return args


def get_image(
    point,
    generator,
    truncation_psi=0.7,
    resize=None,
    generation_depths=None,
    num_cols=None,
    randomize_noise=False,
):
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    point = np.expand_dims(point, axis=0)
    gen_images = generator.run(
        point,
        None,
        truncation_psi=truncation_psi,
        randomize_noise=randomize_noise,
        output_transform=fmt,
    )
    if generation_depths is None:
        img = np.squeeze(gen_images[-1], axis=0)
        img = imresize(img, resize, interp="bicubic") if resize is not None else img
    else:
        import torch as th
        from torchvision.utils import make_grid

        assert all(
            i < len(gen_images) for i in generation_depths
        ), "Requested depth cannot be produced"
        imgs = [np.squeeze(gen_images[i], axis=0) for i in generation_depths]
        imgs = th.stack([th.tensor(gen_images[i]) for i in generation_depths], dim=0)
        n_cols = num_cols if num_cols is not None else int(np.ceil(sqrt(len(imgs))))
        img = make_grid(imgs, nrow=num_cols, padding=0).cpu().numpy()

    return np.squeeze(img)


def main(args):
    # Initialize TensorFlow.
    tflib.init_tf()

    with open(args.pickle_file, "rb") as f:
        _, _, Gs = pickle.load(f)

    # Print network details.
    print("\n\nLoaded the Generator as:")
    Gs.print_layers()

    # Pick latent vector.
    latent_size = Gs.input_shape[1]
    rnd = np.random.RandomState(args.random_state)

    # create the random latent_points for the interpolation
    total_frames = args.num_points * args.transition_points
    all_latents = rnd.randn(total_frames, latent_size)
    all_latents = gaussian_filter(
        all_latents, [args.smoothing * args.transition_points, 0], mode="wrap"
    )
    all_latents = (
        all_latents / np.linalg.norm(all_latents, axis=-1, keepdims=True)
    ) * sqrt(latent_size)

    # handling the latent points
    start_point = all_latents[0]
    points = all_latents[:]

    # if we have only noise realization, then all points are start_point
    if args.only_noise:
        points = np.array([start_point for _ in points])

    # handle the dynamic inputs
    resize, generation_depths = args.resize, args.generation_depths
    if resize is not None:
        resize = [int(val) for val in resize]
    if generation_depths is not None:
        generation_depths = [int(val) for val in generation_depths]

    # make the video:
    sample_image_for_shape = get_image(
        start_point,
        Gs,
        truncation_psi=args.truncation_psi,
        resize=resize,
        generation_depths=generation_depths,
        num_cols=args.num_cols,
        randomize_noise=args.only_noise,
    )
    height, width, _ = sample_image_for_shape.shape

    video = cv2.VideoWriter(
        args.output_file, 0, args.transition_points, (width, height)
    )

    for point in tqdm(all_latents):
        image = get_image(
            point,
            Gs,
            truncation_psi=args.truncation_psi,
            resize=resize,
            generation_depths=generation_depths,
            num_cols=args.num_cols,
            randomize_noise=args.only_noise,
        )
        video.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    cv2.destroyAllWindows()
    video.release()

    print(f"Video created at: {args.output_file}")


if __name__ == "__main__":
    main(parse_arguments())
