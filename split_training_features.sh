#!/bin/bash

for featurefile in `rocm5.7-pytorch split_training_features.py`; do
  base_directory=$(dirname $featurefile)
  file_name=$(basename $featurefile)
  file_name="${file_name%.*}"
  output_fasta="$base_directory/$file_name.fa"
  bedtools getfasta -s -name -fi ./uncompressed-data/Zea_mays.Zm-B73-REFERENCE-NAM-5.0.dna.toplevel.fa -bed "$featurefile" -fo "$output_fasta"
  samtools samtools faidx "$output_fasta"
done;