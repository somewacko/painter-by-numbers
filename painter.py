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

sys.path.append('deep-learning-models')


# ---- Available commands

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

    raise RuntimeError("Not implemented yet!")


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


