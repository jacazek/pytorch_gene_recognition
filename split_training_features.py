import math
import sys
import os
import random
import fasta_utils
import reactivex
from reactivex import operators


source_feature_file = "./uncompressed-data/combined.gff3"
data_output_directory = "./data"
training_set_percent = .50
test_set_percent = .25


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
    (type, data) = dataset
    directory = f"{data_output_directory}/{type}"
    if (not os.path.exists(directory)):
        os.makedirs(directory)
    file_name = f"{directory}/genes.gff3"
    with open(file_name, "w") as file:
        for item in data:
            file.write(f"{item}\n")
    print(f"{file_name}\n")


def extract_start_index(feature):
    split_feature = feature.split("\t")
    return int(split_feature[0]), int(split_feature[3])


def sort_dataset(dataset):
    (name, data) = dataset
    return name, sorted(data, key=extract_start_index)


gene_entries = [gene_entry for gene_entry in fasta_utils.read_feature_file(source_feature_file)]
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

data = [("train_data", 0, train_end_index), ("test_data", train_end_index, test_end_index),
        ("validation_data", test_end_index, total_entries)]
reactivex.from_list(data).pipe(
    debug(lambda item: f"{item[0]} start: {item[1]}, end: {item[2]}"),
    operators.map(lambda data_spec: (data_spec[0], gene_entries[data_spec[1]:data_spec[2]])),
    debug(lambda item: f"{item[0]} length: {len(item[1])}"),
    operators.map(sort_dataset),
).subscribe(on_next=write_dataset)
