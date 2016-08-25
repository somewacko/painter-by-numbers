#!/usr/bin/env python3
"""
painter.py

Trains a distance metric model with a triplet loss function using ResNet50 to
learn whether or not two paintings are by the same painter.

Uses data from Kaggle's "Painter by Numbers" challenge:

    https://www.kaggle.com/c/painter-by-numbers

"""

import argparse
import os
import sys
import types
import warnings

from PIL import Image
from tqdm import tqdm

warnings.simplefilter('ignore', Image.DecompressionBombWarning)
sys.path.append('deep-learning-models')


# ---- Available commands

def preprocess():
    """
    Command to preprocess a dataset. Resizes images such that their smallest
    size is fixed to some value.

    Needed because some of these images are huuuge and processing/loading them
    online becomes a bottleneck.
    """

    parser = argparse.ArgumentParser(
            description = "Preprocesses a dataset.")
    parser.add_argument('data', type=str, help=
            "Directory where data lives.")
    parser.add_argument('output', type=str, help=
            "Directory to write files to.")
    parser.add_argument('-s', '--size', type=int, default=256, help=
            "Size to constrain the smallest side to.")

    args = parser.parse_args(sys.argv[2:])

    all_files = [x for x in os.listdir(args.data)
                 if os.path.isfile( os.path.join(args.data, x) )]

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    print("Processing '{}' -> '{}' with base size {}".format(
        args.data, args.output, args.size))

    for image_filename in tqdm(all_files):
        img = Image.open( os.path.join(args.data, image_filename) )

        w, h = img.size
        if w < h:
            new_w = args.size
            new_h = int(h * (args.size/float(w)))
        else:
            new_w = int(w * (args.size/float(h)))
            new_h = args.size

        try:
            img.resize((new_w, new_h))\
               .convert('RGB')\
               .save( os.path.join(args.output, image_filename) )
        except Exception as e:
            print("Unable to process {}".format(image_filename))
            print(e)


def train():
    """
    Command to train the model.
    """

    parser = argparse.ArgumentParser(
            description = "Trains a model to tell whether two paintings are "
                          "by the same painter.")
    parser.add_argument('data', type=str, help=
            "Directory where training data lives.")

    parser.add_argument('-m', '--model', type=str, default='model.h5', help=
            "Path to save the model as.")
    parser.add_argument('-b', '--batch-size', type=int, default=32, help=
            "Batch size to use while training.")
    parser.add_argument('-e', '--num-epochs', type=int, default=1000, help=
            "Max number of epochs to train for.")
    parser.add_argument('-p', '--patience', type=int, default=5, help=
            "The number of epochs that must occur without validation loss "
            "improving before stopping training early.")

    args = parser.parse_args(sys.argv[2:])

    import painter

    painter.train(args.data, args.model,
        batch_size       = args.batch_size,
        num_epochs       = args.num_epochs,
        patience         = args.patience,
    )


def test():
    """
    Command to evaluate the model on test data.
    """

    parser = argparse.ArgumentParser(
            description = "Evaluates a model on test data.")
    parser.add_argument('data', type=str, help=
            "Directory where test data lives.")
    parser.add_argument('-o', '--output', type=str, default='results.csv', help=
            "Output .csv file to write to.")

    args = parser.parse_args(sys.argv[2:])

    import painter

    raise RuntimeError("Not implemented yet!")


# ---- Command-line invocation

if __name__ == '__main__':

    # Use all functions defined in this file as possible commands to run
    cmd_fns   = [x for x in locals().values() if isinstance(x, types.FunctionType)]
    cmd_names = sorted([fn.__name__ for fn in cmd_fns])
    cmd_dict  = {fn.__name__: fn for fn in cmd_fns}

    parser = argparse.ArgumentParser(
            description = "Generate faces using a deconvolution network.",
            usage       = "fg <command> [<args>]"
    )
    parser.add_argument('command', type=str, help=
            "Command to run. Available commands: {}.".format(cmd_names))

    args = parser.parse_args([sys.argv[1]])

    cmd = None
    try:
        cmd = cmd_dict[args.command]
    except KeyError:
        sys.stderr.write('\033[91m')
        sys.stderr.write("\nInvalid command {}!\n\n".format(args.command))
        sys.stderr.write('\033[0m')
        sys.stderr.flush()

        parser.print_help()

    if cmd is not None:
        cmd()


