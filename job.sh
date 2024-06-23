#!/bin/bash
#SBATCH --exclusive
export MASTER_ADDR=localhost
export MASTER_PORT=12345
export WORLD_SIZE=1
export TQDM_DISABLE=1
#echo "Sleeping 10 seconds"
#sleep 10
# echo "Moving to script directory"
# cd /home/jacob/PycharmProjects/pytorch_dna2vec
echo "Running scripts"
srun rocm-python main.py --learning_rate=0.001 --batch_size=32 --epochs=6
# python-rocm main.py --learning_rate=0.0001 --batch_size=96 --embedding_artifact_uri=mlflow-artifacts:/3/02e1fd194fdc44cab4435191c94b27c2/artifacts/scripted_embedding/data/model.pth  --embedding_dimensions=128