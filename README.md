# pytorch_gene_recognition


## Example scripts
Extract gene records from the annotation file
```bash
cat ./uncompressed-data/Zea_mays.Zm-B73-REFERENCE-NAM-5.0.58.chr.gff3 | grep -v  "#" | grep -i "\sgene\s" > ./uncompressed-data/genes.gff3
```

Get only the chromosomes from the fasta index
```bash
head -n 10 ./uncompressed-data/Zea_mays.Zm-B73-REFERENCE-NAM-5.0.dna.toplevel.fa.fai > ./uncompressed-data/chromosome.fa.fai
```

Generate random sequences from gene file and append to the genes file
```bash
bedtools shuffle -i ./uncompressed-data/genes.gff3 -g ./uncompressed-data/chromosome.fa.fai | sed s/gene/random/g >> ./uncompressed-data/non-genes.gff3 && \
cat ./uncompressed-data/genes.gff3 ./uncompressed-data/non-genes.gff3 > ./uncompressed-data/combined.gff3
```

Extract the set of sequences from the genome into a separate fasta file
```bash
bedtools getfasta -fi ./uncompressed-data/Zea_mays.Zm-B73-REFERENCE-NAM-5.0.dna.toplevel.fa -bed ./uncompressed-data/combined.gff3 -s > ./uncompressed-data/combined.fa
```

Build the vocabulary from the combined fasta file
```bash
rocm5.7-pytorch build_vocabulary.py
```


Splitting the training data
```bash
./split_training_features.sh
```


## Other scripts

single bedtools
```sh
 bedtools getfasta -fi Zea_mays.Zm-B73-REFERENCE-NAM-5.0.dna.toplevel.fa -bed genes.gff3 > output.txt
```

parallel bedtools out to file.  May not need to be parallelized
```sh
split --number=l/10 genes.gff3
parallel bedtools getfasta -fi Zea_mays.Zm-B73-REFERENCE-NAM-5.0.dna_rm.toplevel.fa -bed ::: $(ls xa*) > output.txt
```




## TODO:
- [x] Generate gene data
- [x] Simple model to process batches of gene data
- [ ] Create embedding using an embedding model out of band of training the gene model
  - Word2vec is a common example
  - others?
- [ ] And pre-analysis check for longest sequence in training and calculate if it will exceed GPU memory
- [ ] Move skipping the comments of a gff3 file to the logic that processes those files
- [ ] Create custom data loader what encapsulates the custom collate function
  - We might need to keep the collate function. We may just need to move the data rep out of the collate
- [ ] Create a standard data set so we can randomly select data from the set
  - [x] or make the iterable dataset work with multiple workers?
  - [ ] make the multiprocess iterable dataset robust to different fasta files
- [x] Add vocabulary summary debug output to the build vocab module
- [ ] Instrument the model with asynchronous performance profiler
  - Need to understand where is the bottleneck in the model. Training should be faster.
  - Using smaller batches tends to slow things down (more updates)
  - Nvidia GPU very fast, but not enough memory
  - Using half precision is still just as slow (only 10% decrease in time, 3:40 vs 4:00 for 1 epoch of 17,383 input sequences at batch size 32)
  - Looks like there might be a performance issue in MIOpen library that affects RNN models (might support using transformer instead)
  - ROCM 5.7 seems to perform a little better
  - Training embedding is very fast on Instinct GPU
  - Don't forget to set cpu scaling governer to performance
- [x] Move the model to its own file
  - The model is becoming more complex and cannot live within the main application
  - All dependencies should be injected
- [x] Create a fast file reader class to be  used by various data loaders
- [x] Create a vocabulary class to encapsulate all the vocabulary related functionality
- [x] Ability to save the vocabulary to disk
- [x] Figure out how to prefetch/prepare data before requested by the training loop
  - The pytorch dataloader allows specifying 1 or more workers and how many batches to prefetch
- [x] Generate non-gene data
  - Bed tools can be used to generate random samples from the genome of similar length as genes in provided feature file
- [ ] Handle trailing bases when tokenizer does not evenly divide the entire sequence
- [ ] Establish a process for capturing trained weights of the model for later inference
  - This will need to include the embedding
  - Embeddings and models need to be saved together
  - The token -> id -> vector mapping MUST match for each uniquely trained model
  - If any of the tokenizer parameters are changed, a new embedding and model must be trained
- [ ] Consider NLP transformer model. Optimizations should have already been made around those models
- [x] For the LSTM, consider packing and unpacking the input batch to avoid processing padding tokens
  - Packing and unpacking the sequences are now performed as part of the training on the GPU
- [ ] Clear workflow diagram/algorithm documented for reproducibility
- [ ] Extract parameters for file inputs/outputs and split amounts to arguments
- [ ] Develop a better parallel data loading method.  Currently can only do 1 worker per chromosome.
  - Should consider doing multiple workers per chromosome where possible.
  - Could also just load more chromosomes from more than one genome
- [ ] Save the tokenizer used with the embedding so embeddings and tokens fed into model match

## Helpful links
1. https://pytorch.org/docs/stable/notes/hip.html
2. https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
3. https://pytorch.org/docs/stable/profiler.html
4. https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html
5. https://www.youtube.com/watch?v=-SNdvNdnEl8 - performance profiling


## Trials

The model seems to overfit when embedding layer and hidden layer are greater than 32.
Consider training embedding separately to see if the LSTM can learn better features.
Need to fix the tokenizer before I can train the embedding.
Then need to create embedding checkpoints during training. Probably use name format of
`embed-[dimension]-[vocab]-[accuracy]-[date-time]-checkpoint.pt`.

We can save the embedding doing
```python
torch.save(model.embedding.stat_dict(), "name-of-checkpoint.pt")
```

We can then reload the embedding by:
```python
model.embedding.load_state_dict(torch.load("name-of-checkpoint.pt"))
```

