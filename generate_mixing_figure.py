""" Script for generating mixing diagram """
import argparse
import os
import pickle

import config
import dnnlib
from generate_figures import draw_style_mixing_figure


def generate_and_save_figure(pickle_file_path, output_file, truncation_psi=1.0):
    # load the generator model using the pickle_file
    print("Loading the weights file from:", pickle_file_path)
    with open(pickle_file_path, "rb") as filer:
        _, _, Gs = pickle.load(filer)

    print("Generating and saving the figure at:", output_file)
    draw_style_mixing_figure(
        output_file,
        Gs,
        # using the original values for the rest
        w=1024,
        h=1024,
        src_seeds=[3, 12, 42, 30, 33],
        dst_seeds=[3333, 121, 28, 24, 35, 534],
        style_ranges=[range(0, 4)] * 3 + [range(4, 8)] * 2 + [range(8, 18)],
        truncation_psi=truncation_psi,
    )

    print("Figure has been generated! Please check:", os.path.abspath(output_file))


def parse_arguments():
    """ default Argument parser """
    parser = argparse.ArgumentParser("Generate the Style-Mixing diagram")

    default_output_path = os.path.join(config.result_dir, "style_mixing_figure.png")

    parser.add_argument(
        "--pickle_file_path",
        action="store",
        type=str,
        help="pretrained weights pickle file",
        required=True,
    )

    parser.add_argument(
        "--output_file_path",
        action="store",
        type=str,
        default=default_output_path,
        help="pretrained weights pickle file",
        required=False,
    )

    parser.add_argument(
        "--truncation_psi",
        action="store",
        type=float,
        default=1.0,  # by default do not truncate
        help="pretrained weights pickle file",
        required=False,
    )

    args = parser.parse_args()

    return args


def main(args):
    # initialize tensorflow
    dnnlib.tflib.init_tf()
    generate_and_save_figure(
        args.pickle_file_path, args.output_file_path, truncation_psi=args.truncation_psi
    )


if __name__ == "__main__":
    main(parse_arguments())
