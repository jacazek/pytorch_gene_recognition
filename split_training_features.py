import math
import sys
import os
import random
import fasta_utils
import reactivex
import subprocess
from reactivex import operators
from reactivex.scheduler import ThreadPoolScheduler, immediatescheduler
import multiprocessing
import gzip
from sys import stderr

fasta_file_directory = "./fasta_files"
fasta_file_extension = ".genes.gff3"
fasta_files = [(file, f"{fasta_file_directory}/{file}") for file in os.listdir(fasta_file_directory) if
               file.endswith(fasta_file_extension)]
# source_feature_file = "./uncompressed-data/combined.gff3"
data_output_directory = "./data"
training_set_percent = .60
test_set_percent = .20


def debug(function):
    '''
    Tap into the stream so you can print information
    :param function: the funtion you want to do logging or whatever
    :return: a function what will invoke the provided debug function and resume processing of the stream
    '''

    def tapped(item):
        print(f"DEBUG:\t{function(item)}", file=sys.stderr)
        return item

    return operators.map(tapped)


def print_debug(content):
    print(f"DEBUG:\t{content}", file=sys.stderr)


def write_dataset(dataset):
    (file_info, type, data) = dataset
    directory = f"{data_output_directory}/{type}"
    if (not os.path.exists(directory)):
        os.makedirs(directory)
    file_name = f"{directory}/{file_info[0]}"
    with open(file_name, "w") as file:
        for item in data:
            file.write(f"{item}\n")
    print(f"{file_name}\n")


def extract_start_index(feature):
    split_feature = feature.split("\t")
    return split_feature[0], int(split_feature[3])


def sort_dataset(dataset):
    (name, data) = dataset

    return name, sorted(data, key=extract_start_index)


def decompress(file_info):
    name, file = file_info
    root_name = name.split("_")[0]

    # with gzip.open(f"./zea_mays/{root_name}.fa.gz", "rt") as zipped:
    print(f"unzipping {name}", file=stderr)
    with open(f"./zea_mays/{root_name}.fa", "w") as output:
        # output.writelines(zipped.readlines())
        subprocess.call(["gunzip", "-c", f"./zea_mays/{root_name}.fa.gz"], stdout=output)
    return file_info


def get_entries(file):
    gene_entries = [gene_entry for gene_entry in fasta_utils.read_feature_file(file[1])]
    gene_entries = sorted(gene_entries, key=extract_start_index)
    # randomly shuffle the genes so test/data/validation sets are spread across the genome
    # we may need a more deterministic method of spreading the genes across the genome for
    # reproducible results
    random.shuffle(gene_entries)

    total_entries = len(gene_entries)

    # this will not work well for very small data
    # but then again, we shouldn't be training with data that small
    train_end_index = math.floor(total_entries * training_set_percent)
    test_end_index = math.floor(total_entries * (test_set_percent + training_set_percent))

    return file, gene_entries, [("train_data", 0, train_end_index), ("test_data", train_end_index, test_end_index),
                                ("validation_data", test_end_index, total_entries)]


optimal_thread_count = multiprocessing.cpu_count()
pool_scheduler = ThreadPoolScheduler(optimal_thread_count)

reactivex.from_list(fasta_files).pipe(
    operators.flat_map(lambda file_info: reactivex.from_list([file_info], scheduler=pool_scheduler).pipe(
        operators.map(decompress)
    )),
    operators.map(get_entries),

    operators.flat_map(lambda input: reactivex.from_iterable(input[2], scheduler=pool_scheduler).pipe(

        operators.map(lambda data_spec: (data_spec[0], input[1][data_spec[1]:data_spec[2]])),
        operators.map(sort_dataset),
        operators.map(lambda dataset: (input[0], dataset[0], dataset[1]))
        # operators.subscribe_on(pool_scheduler)
    )),
    operators.map(write_dataset)

    # operators.merge(max_concurrent=10)
).run()
# ).subscribe(on_next=lambda input: print(f"{input[0][0]}, {input[1]}"), on_error=lambda err: print(f"{err}"))
# ).subscribe(on_next=write_dataset, on_error=lambda err: print(f"{err}"))

#
# reactivex.from_list(data).pipe(
#     debug(lambda item: f"{item[0]} start: {item[1]}, end: {item[2]}"),
#     ,
#     debug(lambda item: f"{item[0]} length: {len(item[1])}"),
#     operators.map(sort_dataset),
# ).subscribe(on_next=write_dataset)
