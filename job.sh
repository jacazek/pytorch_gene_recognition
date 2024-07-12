#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
# export MASTER_ADDR=localhost
# export MASTER_PORT=12355
#export WORLD_SIZE=2
export TQDM_DISABLE=1
#echo "Sleeping 10 seconds"
#sleep 10
# echo "Moving to script directory"
# cd /home/jacob/PycharmProjects/pytorch_dna2vec
echo "Running scripts"

#command="rocm-python"
command="nvidia-python"


srun $command main.py --batch_size=32 --epochs=10 --number_genomes=5 --peak_lr=0.003 --initial_lr=0.0007 --warmup_steps=3 --lr_gamma=0.5 --classification_threshold=0.8
#srun rocm-python main.py --batch_size=32 --epochs=15 --number_genomes=4
# python-rocm main.py --learning_rate=0.0001 --batch_size=96 --embedding_artifact_uri=mlflow-artifacts:/3/02e1fd194fdc44cab4435191c94b27c2/artifacts/scripted_embedding/data/model.pth  --embedding_dimensions=128
