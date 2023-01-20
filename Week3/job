#!/bin/bash
#SBATCH -n 4 # Number of cores
#SBATCH --mem 2000 # 2GB solicitados.
#SBATCH -p mhigh,mhigh # or mlow Partition to submit to master low prioriy queue
#SBATCH --gres gpu:1 # Para pedir Pascales MAX 8
#SBATCH -o %x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e %x_%u_%j.err # File to which STDERR will be written

eval "$(conda shell.bash hook)"
conda activate m3
python mlp_MIT_8_scene.py --experiment_name initial_baseline
python mlp_MIT_8_scene.py --LEARNING_RATE 0.1 --experiment_name lr_1
python mlp_MIT_8_scene.py --LEARNING_RATE 0.01 --experiment_name lr_2
python mlp_MIT_8_scene.py --LEARNING_RATE 0.001 --experiment_name lr_3
python mlp_MIT_8_scene.py --LEARNING_RATE 0.0001 --experiment_name lr_4
python mlp_MIT_8_scene.py --LEARNING_RATE 0.00001 --experiment_name lr_5
python mlp_MIT_8_scene.py --LEARNING_RATE 0.000001 --experiment_name lr_6

python mlp_MIT_8_scene.py --IMG_SIZE 8 --experiment_name Isize_1
python mlp_MIT_8_scene.py --IMG_SIZE 16 --experiment_name Isize_2
python mlp_MIT_8_scene.py --IMG_SIZE 32 --experiment_name Isize_3
python mlp_MIT_8_scene.py --IMG_SIZE 64 --experiment_name Isize_4
python mlp_MIT_8_scene.py --IMG_SIZE 128 --experiment_name Isize_5

python mlp_MIT_8_scene.py --BATCH_SIZE 1 --experiment_name bsize_1
python mlp_MIT_8_scene.py --BATCH_SIZE 8 --experiment_name bsize_2
python mlp_MIT_8_scene.py --BATCH_SIZE 16 --experiment_name bsize_3
python mlp_MIT_8_scene.py --BATCH_SIZE 32 --experiment_name bsize_4
python mlp_MIT_8_scene.py --BATCH_SIZE 64 --experiment_name bsize_5
python mlp_MIT_8_scene.py --BATCH_SIZE 128 --experiment_name bsize_6
python mlp_MIT_8_scene.py --BATCH_SIZE 256 --experiment_name bsize_7
python mlp_MIT_8_scene.py --BATCH_SIZE 512 --experiment_name bsize_8