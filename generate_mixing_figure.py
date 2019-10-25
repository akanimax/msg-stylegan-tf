""" Script for generating mixing diagram """
import pickle
import dnnlib
import os
import argparse
from generate_figures import draw_style_mixing_figure


def generate_and_save_figure(pickle_file_path, output_file):
    # load the generator model using the pickle_file
    print("Loading the weights file from:", pickle_file_path)
    with open(pickle_file_path, "rb") as filer:
        _, _, Gs = pickle.load(filer)

    print("Generating and saving the figure at:", output_file)
    draw_style_mixing_figure(output_file, Gs,
                             # using the original values for the rest
                             w=1024, h=1024,
                             src_seeds=[639,701,687,615,2268],
                             dst_seeds=[888,829,1898,1733,1614,845],
                             style_ranges=[range(0,4)]*3+[range(4,8)]*2+[range(8,18)])

    print("Figure has been generated! Please check:", os.path.abspath(output_file))

def parse_arguments():
    """ default Argument parser """
    parser = argparse.ArgumentParser("Generate the Style-Mixing diagram")

    parser.add_argument("--pickle_file_path", action="store", type=str,
                        help="pretrained weights pickle file", required=True)

    parser.add_argument("--output_file_path", action="store", type=str, default="style_mixing_figure.png",
                        help="pretrained weights pickle file", required=False)

    args = parser.parse_args()

    return args

def main(args):
    # initialize tensorflow
    dnnlib.tflib.init_tf()
    generate_and_save_figure(args.pickle_file_path,
                             args.output_file_path)

if __name__ == '__main__':
    main(parse_arguments())