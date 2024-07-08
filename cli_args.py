from dataclasses import dataclass
from typing import List
import argparse
import subprocess
import os


@dataclass
class TrainArguments:
    epochs: int
    number_train_workers: int
    number_validate_workers: int
    number_devices: int
    batch_size: int
    kmer_size: int
    stride: int
    # window_size: int
    embedding_dimensions: int
    # learning_rate: float
    initial_lr: float
    peak_lr: float
    lr_gamma: float
    # number_train_files_per_epoch: int
    # number_validate_files_per_epoch: int
    tags: List[str]
    vocab_artifact_uri: str
    embedding_artifact_uri: str
    artifact_directory: str

    # keep this key last
    command: str


def get_arguments() -> TrainArguments:
    parser = argparse.ArgumentParser(description="Train dna2vec model.")

    parser.add_argument("--epochs", type=int, default=1, help="The number of epochs to train")
    parser.add_argument("--number_train_workers", type=int, default=1,
                        help="The number of worker processes to provide training data")
    parser.add_argument("--number_validate_workers", type=int, default=1,
                        help="The number of worker processes to provide validation data")
    parser.add_argument("--number_devices", type=int, default=1,
                        help="The number of devices on which to train the model")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="The size of each batch for training and validation")
    parser.add_argument("--kmer_size", type=int, default=6,
                        help="The size of each kmer")
    parser.add_argument("--stride", type=int, default=3,
                        help="The stride between kmers")
    # parser.add_argument("--window_size", type=int, default=7,
    #                     help="The size of the window for training embedding")
    parser.add_argument("--embedding_dimensions", type=int, default=64,
                        help="The number of dimensions for each embedding")
    parser.add_argument("--initial_lr", type=float, default=0.0001,
                        help="Initial learning rate for optimizer")
    parser.add_argument("--peak_lr", type=float, default=0.01, help="Peak learning rate for optimizer")
    parser.add_argument("--lr_gamma", type=float, default=0.5,
                        help="The learning rate gamma for the scheduler")
    # parser.add_argument("--number_train_files_per_epoch", type=int, default=1,
    #                     help="The number of fasta files to train per device per epoch")
    # parser.add_argument("--number_validate_files_per_epoch", type=int, default=1,
    #                     help="The number of fasta files to validate per device per epoch")
    parser.add_argument("--tags", action="append", help="Additional key:value tags to capture with the training run")

    # vocabulary arguments
    parser.add_argument("--vocab_artifact_uri", type=str,
                        default="mlflow-artifacts:/2/5b1b448e36b74d74a5f043c6605ab538/artifacts/6mer-s1-202406010300.pickle",
                        help="The uri for vocabulary artifacts")
    parser.add_argument("--embedding_artifact_uri", type=str,
                        default="mlflow-artifacts:/3/c258d034b88c42a5b402baeadb4102c9/artifacts/scripted_embedding/data/model.pth",
                        help="The uri for embedding artifact")
    parser.add_argument("--artifact_directory", type=str, default="./artifacts",
                        help="Directory for downloading artifacts")

    args = parser.parse_args()
    train_arguments = TrainArguments(**vars(args), command=str(
        subprocess.run(["ps", "-p", f"{os.getpid()}", "-o", "args", "--no-headers"], capture_output=True,
                       text=True).stdout))
    return train_arguments
