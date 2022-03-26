""""Parameter parsing."""
import argparse
import torch


def parameter_parser():
    """
    A method to parse up command line parameters.
    By default it trains on the cora dataset.
    The default hyperparameters give a good quality representation without grid search.
    """
    parser = argparse.ArgumentParser(description="HIN Transformers.")

    parser.add_argument("--dataset",
                        nargs="?",
                        default="DBLP",
                        help="Network dataset")

    parser.add_argument("--embedding_output_path",
                        nargs="?",
                        default="./embeddings",
                        help="Embedding output path.")

    parser.add_argument("--number_walks",
                        type=int,
                        default=80,
                        help="Number of random walks per source node. Default is 10.")

    parser.add_argument("--window_size",
                        type=int,
                        default=5,
                        help="Skip-gram window size. Default is 5.")

    parser.add_argument("--negative_samples",
                        type=int,
                        default=5,
                        help="Negative sample number. Default is 5.")

    parser.add_argument("--walk_length",
                        type=int,
                        default=40,
                        help="Truncated random walk length. Default is 40.")

    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Random seed for PyTorch. Default is 42.")

    parser.add_argument("--learning_rate",
                        type=float,
                        default=0.025,  # 0.2
                        help="Learning rate. Default is 0.025.")

    parser.add_argument("--dimension",
                        type=int,
                        default=128,
                        help="Embedding dimensions. Default is 128.")

    parser.add_argument("--iterations",
                        type=int,
                        default=5,
                        help="Embedding dimensions. Default is 10.")

    parser.add_argument('--workers',
                        type=int,
                        default=10,
                        help='Number of parallel workers. Default is 3.')

    parser.add_argument('--walks_per_batch',
                        type=int,
                        default=22)

    parser.add_argument('--batch_size',
                        type=int,
                        default=10000)

    parser.add_argument('--train',
                        type=int,
                        default=1)  # 0: test, 1: train, 2 :validate

    parser.add_argument("--device",
                        default=torch.device("cpu"))

    return parser.parse_args()

