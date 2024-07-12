import os
import mlflow
from tqdm import tqdm
import mlflow.pytorch as mlflow_pytorch
import pickle
from torch.utils.data import DataLoader
import torch
from torchinfo import summary
import torch.distributed as dist
import torch.nn as nn
import torchmetrics
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from fasta_utils.tokenizers import KmerTokenizer
# from fasta_utils.vocab import Vocab
from fasta_pytorch_utils.data.FastaDataset import FastaDataset
from models.GeneRecognition import GeneRecognitionLSTM, ModelType, SplitLSTMModel

from dataclasses import dataclass
from typing import List
import argparse
import subprocess
import os

@dataclass
class TrainArguments:
    number_test_workers: int
    number_devices: int
    batch_size: int
    kmer_size: int
    stride: int
    input_file: str
    model_artifact_uri: str
    artifact_directory: str
    classification_threshold: float


def get_arguments() -> TrainArguments:
    parser = argparse.ArgumentParser(description="Train dna2vec model.")

    parser.add_argument("--number_test_workers", type=int, default=1,
                        help="The number of worker processes to provide test data")
    parser.add_argument("--number_devices", type=int, default=1,
                        help="The number of devices on which to train the model")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="The size of each batch for training and validation")
    parser.add_argument("--kmer_size", type=int, default=6,
                        help="The size of each kmer")
    parser.add_argument("--stride", type=int, default=3,
                        help="The stride between kmers")
    parser.add_argument("--input_file", type=str, default=None, help="File containing list of genes fasta files to process", required=True)

    # vocabulary arguments
    parser.add_argument("--model_artifact_uri", type=str,
                        default="mlflow-artifacts:/4/5b15ec022e864e04a5fdb5a167750440/artifacts/model/data/model.pth",
                        help="The uri for model artifact")
    parser.add_argument("--artifact_directory", type=str, default="./artifacts",
                        help="Directory for downloading artifacts")
    parser.add_argument("--classification_threshold", type=float, default=0.5, help="Classification probability threshold of sequence being a gene")

    return parser.parse_args()


mlflow.set_tracking_uri("http://localhost:8080")

data_directory = "./data"
test_directory = os.path.join(data_directory, "test_data")
fasta_file_extension = ".genes.fa"




def load_model(model_uri):
    model = torch.load(model_uri)
    return model


def load_vocabulary(vocabulary_uri):
    with open(vocabulary_uri, "rb") as file:
        return pickle.load(file)

def read_genes_files(genes_files):
    with open(genes_files) as file:
        for line in file:
            if line and not line.startswith("#"):
                yield line.rstrip()



class CreateDnaSequenceCollateFunction:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def __call__(self, batch):
        """
        Collect then pad the sequences into a tensor
        :param batch: the batch
        :return: padded sequences, the original targets, and corresponding original lengths of each sequence
        """
        batch = [(item[0], item[1], index) for index, item in enumerate(batch)]
        # sort the batch by length of sequence ascending
        batch = sorted(batch, key=lambda item: len(item[0]), reverse=True)
        # unzip the sequences from the corresponding targets
        [sequences, targets, indices] = zip(*batch)

        # make the targets a 2-dimensional batch of size 1, so we can easily support multiple targets later
        # by easily refactoring the dataset and dataloader
        # targets = torch.stack([torch.tensor([target]) for target in targets], dim=0)
        targets = torch.unsqueeze(torch.tensor(targets, dtype=torch.float32), dim=1)
        # gather the original lengths of the sequence before padding
        lengths = [len(sequence) for sequence in sequences]
        """
        The sequences should have already been loaded for cpu manipulation, so we should pad them before
        moving them to the gpu because it is more efficient to pad on the cpu
        """
        sequences = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(sequence, dtype=torch.long) for sequence in sequences], batch_first=True,
            padding_value=self.vocabulary["pad"])
        return sequences, targets, lengths, indices


def demo_basic(rank, world_size, train_arguments: TrainArguments):
    # genes_files = []
    if (not train_arguments.input_file):
        raise Exception("Input file required")
    genes_files = list(read_genes_files(train_arguments.input_file))
    test_fasta_files = genes_files
    # test_fasta_files = [f"{test_directory}/{file}" for file in os.listdir(test_directory) if
    #                     file.endswith(fasta_file_extension)]


    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = rank
    data_type = torch.float32

    model_name = os.path.basename(train_arguments.model_artifact_uri)
    model_path = os.path.join(train_arguments.artifact_directory, model_name)
    model = load_model(model_path)
    model.to(device)
    vocabulary = model.vocabulary

    setup(rank, world_size)

    tokenizer = KmerTokenizer(train_arguments.kmer_size, train_arguments.stride)
    test_dataset = FastaDataset(test_fasta_files, tokenizer=tokenizer, vocabulary=vocabulary,
                                 dtype=data_type)

    test_dataloader = DataLoader(test_dataset, batch_size=train_arguments.batch_size,
                                     num_workers=train_arguments.number_test_workers,
                                     collate_fn=CreateDnaSequenceCollateFunction(vocabulary))


    ddp_model = DDP(model, device_ids=[rank])

    criterion = nn.BCEWithLogitsLoss().to(device=device)

    precision = torchmetrics.Precision("binary", num_classes=2, threshold=train_arguments.classification_threshold).to(device=device)
    recall = torchmetrics.Recall("binary", num_classes=2, threshold=train_arguments.classification_threshold).to(device=device)
    f1score = torchmetrics.F1Score("binary", num_classes=2, threshold=train_arguments.classification_threshold).to(device=device)
    validate_loss_mean = torchmetrics.aggregation.MeanMetric().to(device)
    validate_accuracy_mean = torchmetrics.aggregation.MeanMetric().to(device)

    with tqdm(test_dataloader, unit="batch") as test_batch:
        test_batch.set_description(f"Test")
        ddp_model.module.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_batch):
                genes, labels, lengths, indices = batch
                # data loader provides tensors for CPU
                # convert genes and labels to tensors on desired device
                genes = genes.to(device)
                labels = labels.to(device)

                # lengths should remain on cpu as all processing what needs lengths must be done on cpu
                with torch.autocast(device_type=genes.device.type, dtype=torch.float16):
                    output = ddp_model.module(genes, lengths)
                    loss = criterion(output,
                                     labels)

                probabilities = torch.sigmoid(output)
                precision.update(probabilities, labels)
                recall.update(probabilities, labels)
                f1score.update(probabilities, labels)
                binary_predictions = (probabilities > train_arguments.classification_threshold).float()
                correct = (binary_predictions == labels).float().sum().item()

                loss = validate_loss_mean(loss.item()).item()
                accuracy = validate_accuracy_mean(correct / output.size(0)).item()
                test_batch.set_postfix(batch_loss=loss, batch_accuracy=accuracy)

        for key, value in ({
            f"accuracy": validate_accuracy_mean.compute().item(),
            f"precision": precision.compute().item(),
            f"recall": recall.compute().item(),
            f"f1_score": f1score.compute().item()
        }).items():
            print(f"{key}: {value}")
    precision.reset()
    recall.reset()
    validate_loss_mean.reset()
    validate_accuracy_mean.reset()
    f1score.reset()

    cleanup()


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def main():
    train_arguments = get_arguments()
    # Download artifacts needed by the script
    mlflow.artifacts.download_artifacts(artifact_uri=train_arguments.model_artifact_uri,
                                        dst_path=train_arguments.artifact_directory)
    os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "0"
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    torch.multiprocessing.spawn(demo_basic,
                                args=(world_size, train_arguments),
                                nprocs=world_size)


if __name__ == "__main__":
    main()
