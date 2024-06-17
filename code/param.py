import argparse


def parameter_parser():

    parser = argparse.ArgumentParser(description="Run.")

    parser.add_argument('--seed', type=int, default=0, help='Random seed.')

    parser.add_argument("--dataset-path",
                        nargs="?",
                        default="./dataset",
                        help="Training dataset.")

    parser.add_argument("--epoch",
                        type=int,
                        default=200,
                        help="Number of training epochs. Default is 400.")

    parser.add_argument("--cnn",
                        type=int,
                        default=2,
                        help="Number Default is 2.")

    parser.add_argument("--out-channels",
                        type=int,
                        default=16,
                        help="out-channels of cnn. Default is 256.")

    parser.add_argument("--miRNA-number",
                        type=int,
                        default=788,
                        help="miRNA number. Default is 788.")

    parser.add_argument("--fmi",
                        type=int,
                        default=64,
                        help="miRNA feature dimensions. Default is 128.")

    parser.add_argument("--disease-number",
                        type=int,
                        default=374,
                        help="disease number. Default is 88.")

    parser.add_argument("--fdis",
                        type=int,
                        default=64,
                        help="disease feature dimensions. Default is 128.")

    return parser.parse_args()