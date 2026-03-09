"arguments for training neural solver for optimal transport"

import argparse
import sys


def str2bool(bool_str):
    if isinstance(bool_str, bool):
        return bool_str
    if bool_str.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif bool_str.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def _get_args():
    if "ipykernel_launcher" in sys.argv[0]:
        sys.argv = [""]  # add if ruwnning in interactivate mode in vscode

    parser = argparse.ArgumentParser(
        description="Input for the (partially-) input-convex neural networks."
    )

    ##General settings
    parser.add_argument(
        "-device", "--device", type=str, default="cuda", help="Which device to run on"
    )

    parser.add_argument(
        "-d",
        "--output_dir",
        type=str,
        # default="/home/users/a/algren/work/ftag-otcalib/tb/gridsearch/",
        default=None,
        help="Choose the base output directory",
    )

    parser.add_argument(
        "-covx_d",
        "--convex-dimensions",
        type=int,
        help="Which activation function should be used to enforce convexity?",
        default=3,
    )

    parser.add_argument(
        "-noncovx_d",
        "--nonconvex-dimensions",
        type=int,
        help="Which activation function should be used to enforce convexity?",
        default=1,
    )

    ##network settings
    parser.add_argument(
        "-cns",
        "--nodes-per-layer-convex",
        type=int,
        default=128,
        help="Choose number of nodes per layer in the convex network",
    )

    parser.add_argument(
        "-non-cns",
        "--nodes-per-layer-non-convex",
        type=int,
        default=8,
        help="Choose number of nodes per layer in the non convex network",
    )

    parser.add_argument(
        "-ls",
        "--number-layers-sets",
        type=int,
        default=6,
        help="Choose number of layers in the PICNN",
    )

    parser.add_argument(
        "-act",
        "--convex-layer-activation",
        type=str,
        help="Choose activation function used for convex network",
        # default="trainable_softplus",
        default="softplus_zeroed",
    )

    parser.add_argument(
        "-non-act",
        "--non-convex-layer-activation",
        type=str,
        help="Choose activation function for non convex network",
        default="softplus_zeroed",
    )

    parser.add_argument(
        "-lnorm",
        "--normalization",
        type=str,
        help="Choose to use 'batch' norm or none ()",
        default="",
    )

    parser.add_argument(
        "-act_params",
        "--activation-params",
        type=str,
        help="Choose activation function for non convex network",
        default={},
    )

    parser.add_argument(
        "--act_enforce_cvx",
        choices=["relu", "softplus"],
        type=str,
        help="""Activation function for force conditional connections to be positive
                should be >=0 at all x""",
        default="softplus",
    )
    parser.add_argument(
        "--act_weight_zz",
        choices=["relu", "softplus"],
        type=str,
        help="""Activation function for force the weight_zz to be positive.
                Should be >=0 at all x""",
        default="softplus",
    )
    parser.add_argument(
        "--first_act_sym",
        choices=["sym_softplus", "sym_leakyrelu", "sym_leaky_relu", "no"],
        type=str,
        help="Activation function for the first layer can be x^2 like",
        # default="sym_softplus",
        default="no",
    )

    ##Training settingss
    parser.add_argument(
        "-f_lr",
        "--f-learning-rate",
        type=float,
        help="Learning rate for training for f network.",
        # default=1e-4,
        default=1e-4,
    )

    parser.add_argument(
        "-g_lr",
        "--g-learning-rate",
        type=float,
        help="Learning rate for training for f network.",
        default=1e-4,
        # default=5e-4,
    )

    parser.add_argument(
        "-opt",
        "--optimizer",
        type=str,
        # choices=["Adam", "Nadam", "SGD"],
        help="Choose which optimizer to use in training.",
        default="adamw",
        # default="adam",
    )

    parser.add_argument(
        "-lr-sch",
        "--learning-rate-scheduler",
        help="Choose which optimiser to use in training. not working at the moment",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
    )

    parser.add_argument(
        "-ct",
        "--correction-trainable",
        type=str2bool,
        nargs="?",
        const=True,
        help="Trainable correction in output",
        default="yes",
    )

    parser.add_argument(
        "-gf", "--g-per-f", type=int, help="Number of training iter over g", default=4
    )

    parser.add_argument(
        "-fg", "--f-per-g", type=int, help="Number of training iter over f", default=1
    )

    parser.add_argument(
        "-grad_clip",
        "--gradient-clipping",
        type=str,
        help="Gradient clipping",
        default="5,5",
    )

    parser.add_argument(
        "-grad_norm",
        "--gradient-penalty",
        type=str,
        help="Gradient penalities",
        default="0,0",
    )

    parser.add_argument(
        "-be", "--burn-epochs", type=int, help="burn epochs", default=-1
    )

    parser.add_argument(
        "-es",
        "--epoch-size",
        type=int,
        help="Number of batches in an epoch",
        default=64,
    )

    parser.add_argument(
        "-ne",
        "--nepochs",
        type=int,
        help="For how many epochs to train.",
        default=5000,
    )

    parser.add_argument(
        "-rr",
        "--reverse-ratio",
        type=float,
        help="ratio of switching roles between discriminator and generator",
        default=1,
    )

    parser.add_argument(
        "-b", "--batchsize", type=int, help="Size of batches in training", default=512
    )

    parser.add_argument(
        "--pretrain",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="To pretrain the model",
    )

    parser.add_argument(
        "-n",
        "--outputname",
        type=str,
        default=None,
        help="Set the output name directory",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        dest="verbose",
        help="Increase verbosity of training output",
    )

    return parser
