#!/bin/bash

# Source directory containing .gz files
source_directory="./zea_mays"

# Destination directory for uncompressed files
destination_directory="./fasta_files"

# Check if the destination directory exists, create if not
if [ ! -d "$destination_directory" ]; then
  mkdir -p "$destination_directory"
fi

# Function to unzip and move files in the background
extract() {
  gz_file="$1"
  base_name=$(basename "$gz_file" .gz)
  extension="${base_name##*.}"
  filename="${base_name%.*}"
  genome_name=${filename%_Zm*}
  gff3_file="$destination_directory/$filename.genes.gff3"
  temp="$destination_directory/$filename.temp"
  gunzip -c "$gz_file" | grep -v  "#" | grep -i "\sgene\s" > "$gff3_file"
  bedtools shuffle -chrom -i "$gff3_file" -g <(cat "$source_directory/$genome_name.fa.gz.fai") | sed s/gene/random/g > "$temp"
  cat "$temp" >> "$gff3_file"
  rm "$temp"
  bedtools getfasta -fi <(gunzip -c "$source_directory/$genome_name.fa.gz") -bed "$gff3_file" -s > "$destination_directory/$genome_name.genes.fa"


#  samtools samtools faidx "$gff3_file"
}

# Loop through each .gz file in the source directory and run in parallel
for gz_file in "$source_directory"/*[!TE].gff3.gz; do
  #  &&
  (extract "$gz_file") &
done

# Wait for all background processes to finish
wait



echo "Unzipping complete. Uncompressed files are in $destination_directory."