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
    embedding_dimensions: int
    number_genomes: int
    vocab_artifact_uri: str
    embedding_artifact_uri: str
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
    parser.add_argument("--embedding_dimensions", type=int, default=64,
                        help="The number of dimensions for each embedding")
    parser.add_argument("--number_genomes", type=int, default=1, help="Number of genomes to use per epoch")

    # vocabulary arguments
    parser.add_argument("--vocab_artifact_uri", type=str,
                        default="mlflow-artifacts:/2/5b1b448e36b74d74a5f043c6605ab538/artifacts/6mer-s1-202406010300.pickle",
                        help="The uri for vocabulary artifacts")
    parser.add_argument("--embedding_artifact_uri", type=str,
                        default="mlflow-artifacts:/3/c258d034b88c42a5b402baeadb4102c9/artifacts/scripted_embedding/data/model.pth",
                        help="The uri for embedding artifact")
    parser.add_argument("--model_artifact_uri", type=str,
                        default="mlflow-artifacts:/4/4e26a66645684c55b2a77413e0ab468a/artifacts/model/data/model.pth",
                        help="The uri for model artifact")
    parser.add_argument("--artifact_directory", type=str, default="./artifacts",
                        help="Directory for downloading artifacts")
    parser.add_argument("--classification_threshold", type=float, default=0.5, help="Classification probability threshold of sequence being a gene")

    return parser.parse_args()


mlflow.set_tracking_uri("http://localhost:8080")

data_directory = "./data"
test_directory = os.path.join(data_directory, "test_data")
fasta_file_extension = ".genes.fa"
test_fasta_files = [f"{test_directory}/{file}" for file in os.listdir(test_directory) if
                    file.endswith(fasta_file_extension)]



def load_model(model_uri):
    model = torch.load(model_uri)
    return model


def load_vocabulary(vocabulary_uri):
    with open(vocabulary_uri, "rb") as file:
        return pickle.load(file)


def save_model_summary(model, input_size, artifact_path):
    file_name = os.path.join(artifact_path, "model_summary.txt")
    with open(file_name, "w") as file:
        summary_string = str(summary(model, input_size, dtypes=[torch.long]))
        file.write(summary_string)
    mlflow.log_artifact(file_name, artifact_path="artifacts")


class CreateDnaSequenceCollateFunction:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def __call__(self, batch):
        """
        Collect then pad the sequences into a tensor
        :param batch: the batch
        :return: padded sequences, the original targets, and corresponding original lengths of each sequence
        """
        # sort the batch by length of sequence ascending
        batch = sorted(batch, key=lambda item: len(item[0]), reverse=True)
        # unzip the sequences from the corresponding targets
        [sequences, targets] = zip(*batch)

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
        # worker_info = torch.utils.data.get_worker_info()
        # print(f"worker {worker_info.id} providing a batch")
        return sequences, targets, lengths


def demo_basic(rank, world_size, train_arguments: TrainArguments):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = rank
    data_type = torch.float32

    vocabulary_name = os.path.basename(train_arguments.vocab_artifact_uri)
    vocabulary_path = os.path.join(train_arguments.artifact_directory, vocabulary_name)
    vocabulary = load_vocabulary(vocabulary_path)

    embedding_name = os.path.basename(train_arguments.embedding_artifact_uri)
    embedding_path = os.path.join(train_arguments.artifact_directory, embedding_name)
    embedding = load_model(embedding_path)
    embedding.to(device)

    setup(rank, world_size)

    tokenizer = KmerTokenizer(train_arguments.kmer_size, train_arguments.stride)
    test_dataset = FastaDataset(test_fasta_files[0:train_arguments.number_genomes], tokenizer=tokenizer, vocabulary=vocabulary,
                                 dtype=data_type)
    print(len(test_fasta_files[0:train_arguments.number_genomes]))

    test_dataloader = DataLoader(test_dataset, batch_size=train_arguments.batch_size,
                                     num_workers=train_arguments.number_test_workers,
                                     collate_fn=CreateDnaSequenceCollateFunction(vocabulary))

    model_name = os.path.basename(train_arguments.model_artifact_uri)
    model_path = os.path.join(train_arguments.artifact_directory, model_name)
    model = load_model(model_path)
    model.to(device)
    # model = GeneRecognitionLSTM(vocabulary, train_arguments.embedding_dimensions, bidirectional=False)
    # model.to(device)

    # model.embedding.load_state_dict({"weight": embedding.weight})
    # model.embedding.weight.requires_grad = False

    ddp_model = DDP(model, device_ids=[rank])

    criterion = nn.BCEWithLogitsLoss().to(device=device)

    # experiment = mlflow.get_experiment_by_name("Gene Recognition")
    # with mlflow.start_run(experiment_id=experiment.experiment_id):
    # mlflow.log_params(train_arguments.__dict__ | {
    #
    #     # training hyper parameters
    #     "optimizer": type(optimizer).__name__,
    #     "optimizer_detailed": str(optimizer),
    #     "lr_scheduler": type(lr_scheduler).__name__,
    #     "loss_function": type(criterion).__name__,
    #     # "window_size": train_arguments.window_size,
    #
    # })

    # additional_tags = {}
    # if train_arguments.tags is not None and len(train_arguments.tags) > 0:
    #     for tag in train_arguments.tags:
    #         keyValue = tag.split(":")
    #         if len(keyValue) > 1:
    #             additional_tags[keyValue[0]] = keyValue[1]
    # mlflow.set_tags(additional_tags)
    # if rank == 0:
    #     save_model_summary(model, (train_arguments.batch_size, train_arguments.window_size - 1),
    #                        train_arguments.artifact_directory)
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
                genes, labels, lengths = batch
                batch_count = batch_idx
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
                # if batch_idx % 100 == 0:
                binary_predictions = (probabilities > train_arguments.classification_threshold).float()
                correct = (binary_predictions == labels).float().sum().item()

                loss = validate_loss_mean(loss.item()).item()
                accuracy = validate_accuracy_mean(correct / output.size(0)).item()

                # mlflow.log_metrics({
                #     f"loss_validate_{epoch + 1}": loss,
                #     f"accuracy_validate_{epoch + 1}": accuracy
                # }, step=batch_idx)
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
    if rank == 0:
        mlflow_pytorch.log_model(model, "model")

        # TODO: figure out how to trace network containing other python code
        # scripted_model = torch.jit.script(model)
        # mlflow_pytorch.log_model(scripted_model, "scripted_model")

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

    mlflow.artifacts.download_artifacts(artifact_uri=train_arguments.vocab_artifact_uri,
                                        dst_path=train_arguments.artifact_directory)
    mlflow.artifacts.download_artifacts(artifact_uri=train_arguments.embedding_artifact_uri,
                                        dst_path=train_arguments.artifact_directory)
    mlflow.artifacts.download_artifacts(artifact_uri=train_arguments.model_artifact_uri,
                                        dst_path=train_arguments.artifact_directory)
    os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "0"
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    torch.multiprocessing.spawn(demo_basic,
                                args=(world_size, train_arguments),
                                nprocs=world_size)


if __name__ == "__main__":
    main()
