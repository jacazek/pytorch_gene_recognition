#!/bin/bash
#SBATCH --exclusive
export MASTER_ADDR=localhost
export MASTER_PORT=12345
export WORLD_SIZE=2
export TQDM_DISABLE=1
echo "Sleeping 10 seconds"
sleep 10
# echo "Moving to script directory"
# cd /home/jacob/PycharmProjects/pytorch_dna2vec
echo "Running scripts"
srun rocm-python main.py
#srun rocm-python src/train.py --number_train_workers 3 --number_devices 1 --number_validate_workers 7 --learning_rate 0.0001 --epochs 3 --window_size 7 --batch_size 20480 --kmer_size 6 --stride 3 --lr_gamma 0.5 --tag genus:zea --vocab_artifact_uri=mlflow-artifacts:/2/5b1b448e36b74d74a5f043c6605ab538/artifacts/6mer-s1-202406010300.pickle --embedding_dimensions 128
# srun rocm-python src/train.py --number_train_workers 3 --number_devices $WORLD_SIZE --number_validate_workers 7 --learning_rate 0.0001 --epochs 1 --window_size 15 --batch_size 20480 --kmer_size 7 --stride 3 --lr_gamma 0.5 --tag genus:zea --vocab_artifact_uri=mlflow-artifacts:/2/913f8eee3e484dafb856287c9d5cee35/artifacts/7mer-s1-202406011143.pickle --embedding_dimensions 256
# srun rocm-python src/train.py --number_train_workers 3 --number_devices 2 --number_validate_workers 7 --learning_rate 0.0003 --epochs 3 --window_size 7 --batch_size 20480 --kmer_size 7 --stride 3 --lr_gamma 0.5 --tag genus:zea --vocab_artifact_uri=mlflow-artifacts:/2/913f8eee3e484dafb856287c9d5cee35/artifacts/7mer-s1-202406011143.pickle --embedding_dimensions 256
# srun rocm-python src/train.py --number_train_workers 3 --number_devices 2 --number_validate_workers 7 --learning_rate 0.0003 --epochs 3 --window_size 15 --batch_size 20480 --kmer_size 7 --stride 3 --lr_gamma 0.5 --tag genus:zea --vocab_artifact_uri=mlflow-artifacts:/2/913f8eee3e484dafb856287c9d5cee35/artifacts/7mer-s1-202406011143.pickle --embedding_dimensions 256
echo "Done"