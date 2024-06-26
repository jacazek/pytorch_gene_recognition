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

from cli_args import get_arguments, TrainArguments
mlflow.set_tracking_uri("http://localhost:8080")


data_directory = "./data"
train_directory = os.path.join(data_directory, "train_data")
test_directory = os.path.join(data_directory, "test_data")
validate_directory = os.path.join(data_directory, "validate_data")
fasta_file_extension = ".genes.fa"
train_fasta_files = [(file, f"{train_directory}/{file}") for file in os.listdir(train_directory) if
               file.endswith(fasta_file_extension)]
test_fasta_files = [(file, f"{test_directory}/{file}") for file in os.listdir(test_directory) if
               file.endswith(fasta_file_extension)]
validate_fasta_files = [(file, f"{validate_directory}/{file}") for file in os.listdir(validate_directory) if
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
        sequences = torch.nn.utils.rnn.pad_sequence([torch.tensor(sequence, dtype=torch.long) for sequence in sequences], batch_first=True, padding_value=self.vocabulary["pad"])
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
    # embedding.weight.requires_grad = False


    print(f"Running basic DDP example on rank {rank}.")
    print(f"torch enabled {torch.cuda.is_available()}")
    setup(rank, world_size)



    tokenizer = KmerTokenizer(train_arguments.kmer_size, train_arguments.stride)
    train_dataset = FastaDataset(train_fasta_files[0:3], tokenizer=tokenizer, vocabulary=vocabulary,
                                 dtype=data_type)
    validate_dataset = FastaDataset(validate_fasta_files[0:3], tokenizer=tokenizer, vocabulary=vocabulary,
                                dtype=data_type)

    train_dataloader = DataLoader(train_dataset, batch_size=train_arguments.batch_size, num_workers=train_arguments.number_train_workers,
                                  collate_fn=CreateDnaSequenceCollateFunction(vocabulary), prefetch_factor=5)
    validate_dataloader = DataLoader(validate_dataset, batch_size=train_arguments.batch_size, num_workers=train_arguments.number_validate_workers,
                                 collate_fn=CreateDnaSequenceCollateFunction(vocabulary), prefetch_factor=5)

    model = GeneRecognitionLSTM(vocabulary, train_arguments.embedding_dimensions)
    model.to(device)

    model.embedding.load_state_dict({"weight": embedding.weight})
    model.embedding.weight.requires_grad = False

    ddp_model = DDP(model, device_ids=[rank])

    criterion = nn.BCEWithLogitsLoss().to(device=device)
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=train_arguments.learning_rate, fused=True)
    # optimizer = torch.optim.SGD(model.parameters(), lr=train_arguments.learning_rate, fused=True)
    scaler = torch.cuda.amp.GradScaler()
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)

    experiment = mlflow.get_experiment_by_name("Gene Recognition")
    with mlflow.start_run(experiment_id=experiment.experiment_id):
        mlflow.log_params({
            "epochs": train_arguments.epochs,
            "number_devices": train_arguments.number_devices,
            "number_train_workers": train_arguments.number_train_workers,
            "number_validate_workers": train_arguments.number_validate_workers,

            # data processing parameters
            "vocabulary": train_arguments.vocab_artifact_uri,
            "kmer_size": train_arguments.kmer_size,
            "stride": train_arguments.stride,
            "batch_size": train_arguments.batch_size,
            "embedding_dimensions": train_arguments.embedding_dimensions,

            # training hyper parameters
            "optimizer": type(optimizer).__name__,
            "optimizer_detailed": str(optimizer),
            "lr_initial": train_arguments.learning_rate,
            "lr_scheduler": type(lr_scheduler).__name__,
            "lr_gamma": lr_scheduler.gamma,
            "loss_function": type(criterion).__name__,
            # "window_size": train_arguments.window_size,

        })

        additional_tags = {}
        if train_arguments.tags is not None and len(train_arguments.tags) > 0:
            for tag in train_arguments.tags:
                keyValue = tag.split(":")
                if len(keyValue) > 1:
                    additional_tags[keyValue[0]] = keyValue[1]
        mlflow.set_tags({
                            "command": train_arguments.command
                        } | additional_tags)
        # if rank == 0:
        #     save_model_summary(model, (train_arguments.batch_size, train_arguments.window_size - 1),
        #                        train_arguments.artifact_directory)

        for epoch in range(train_arguments.epochs):
            with tqdm(train_dataloader, unit="batch") as train_batch:
                train_batch.set_description(f"Epoch {epoch} train")
                ddp_model.train()
                train_loss_mean = torchmetrics.aggregation.MeanMetric().to(device)
                train_accuracy_mean = torchmetrics.aggregation.MeanMetric().to(device)

                for batch_idx, batch in enumerate(train_batch):
                    genes, labels, lengths = batch
                    genes = genes.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad(set_to_none=True)

                    # lengths should remain on cpu as all processing what needs lengths must be done on cpu
                    with torch.autocast(device_type=genes.device.type, dtype=torch.float16):
                        output = ddp_model(genes, lengths)
                        loss = criterion(output,
                                         labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    probabilities = torch.sigmoid(output)

                    # if batch_idx % 100 == 0:
                    binary_predictions = (probabilities > 0.5).float()
                    correct = (binary_predictions == labels).float().sum().item()

                    loss = train_loss_mean(loss.item()).item()
                    accuracy = train_accuracy_mean(correct / output.size(0)).item()

                    mlflow.log_metrics({
                        f"loss_train_{epoch + 1}": loss,
                        f"accuracy_train_{epoch + 1}": accuracy
                    }, step=batch_idx)
                    train_batch.set_postfix(batch_loss=loss, batch_accuracy=accuracy)
            lr_scheduler.step()

            with tqdm(validate_dataloader, unit="batch") as test_batch:
                test_batch.set_description(f"Epoch {epoch} test")
                ddp_model.module.eval()
                validate_loss_mean = torchmetrics.aggregation.MeanMetric().to(device)
                validate_accuracy_mean = torchmetrics.aggregation.MeanMetric().to(device)
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

                        # if batch_idx % 100 == 0:
                        binary_predictions = (probabilities > 0.5).float()
                        correct = (binary_predictions == labels).float().sum().item()

                        loss = validate_loss_mean(loss.item()).item()
                        accuracy = validate_accuracy_mean(correct / output.size(0)).item()

                        mlflow.log_metrics({
                            f"loss_validate_{epoch + 1}": loss,
                            f"accuracy_validate_{epoch + 1}": accuracy
                        }, step=batch_idx)
                        train_batch.set_postfix(batch_loss=loss, batch_accuracy=accuracy)

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
    os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "0"
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    torch.multiprocessing.spawn(demo_basic,
                                args=(world_size,train_arguments),
                                nprocs=world_size)


if __name__ == "__main__":
    main()