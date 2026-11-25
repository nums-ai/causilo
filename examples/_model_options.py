import argparse


def model_options(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--model-version", choices=("1.0", "1.5"), default=argparse.SUPPRESS,
        help="Defaults to Causilo 1.5. Select 1.0 to use the public checkpoints.",
    )
    return vars(parser.parse_args())
