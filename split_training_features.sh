#!/bin/bash

stuff() {
  featurefile="$1"
  #  echo $featurefile
  base_directory=$(dirname "$featurefile")
  file_name=$(basename "$featurefile")
  file_name="${file_name%.*}"
  source_fasta_file="./zea_mays/${file_name%_Zm*}.fa"
  output_fasta="$base_directory/$file_name.fa"
  echo "${source_fasta_file}"
  echo "${output_fasta}"
#  if [ ! -f "$source_fasta_file" ]; then
#    echo "extracting ${source_fasta_file}.gz"
#
#    gunzip -c "$source_fasta_file.gz" > "$source_fasta_file"
#    echo "done extracting ${source_fasta_file}.gz"
#  fi
  bedtools getfasta -s -name -fi "$source_fasta_file" -bed "$featurefile" -fo "$output_fasta"
#  rm "$source_fasta_file"
  samtools samtools faidx "$output_fasta"
}

for featurefile in $(rocm5.7-pytorch split_training_features.py); do
  stuff "$featurefile" &
done;

wait