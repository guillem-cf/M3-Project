#!/bin/bash
#SBATCH -n 4 # Number of cores
#SBATCH --mem 2000 # 2GB solicitados.
#SBATCH -p mhigh,mhigh # or mlow Partition to submit to master low prioriy queue
#SBATCH --gres gpu:1 # Para pedir Pascales MAX 8
#SBATCH -o %x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e %x_%u_%j.err # File to which STDERR will be written

eval "$(conda shell.bash hook)"
conda activate m3
python 5.hyperparameters.py --experiment_name task5_cut1_pool --EPOCHS 300 --DATASET_DIR /ghome/group07/M3-Project-new/Week4/MIT_small_train_1
# python 1.train.py --experiment_name task2_2_cut1_pool_D1024 --EPOCHS 300 --REMOVE_BLOCK 1 --DATASET_DIR /ghome/group07/M3-Project-new/Week4/MIT_small_train_2
# python 1.train.py --experiment_name task2_3_cut1_pool_D1024 --EPOCHS 300 --REMOVE_BLOCK 1 --DATASET_DIR /ghome/group07/M3-Project-new/Week4/MIT_small_train_3
# python 1.train.py --experiment_name task2_4_cut1_pool_D1024 --EPOCHS 300 --REMOVE_BLOCK 1 --DATASET_DIR /ghome/group07/M3-Project-new/Week4/MIT_small_train_4

# python 3.dataugmentation.py --experiment_name task3_cut1_pool_D1024_all --MODEL_NAME best_task_2_1_cut1_pool_D1024_model_checkpoint.h5 --EPOCHS 300 --DATASET_DIR /ghome/group07/M3-Project-new/Week4/MIT_small_train_1
# python 1.2.all_model.py --experiment_name task2_1_cut1_pool_D1024_all --MODEL_NAME best_task_2_2_cut1_pool_D1024_model_checkpoint.h5 --EPOCHS 300 --DATASET_DIR /ghome/group07/M3-Project-new/Week4/MIT_small_train_2
# python 1.2.all_model.py --experiment_name task2_1_cut1_pool_D1024_all --MODEL_NAME best_task_2_3_cut1_pool_D1024_model_checkpoint.h5 --EPOCHS 300 --DATASET_DIR /ghome/group07/M3-Project-new/Week4/MIT_small_train_3
# python 1.2.all_model.py --experiment_name task2_1_cut1_pool_D1024_all --MODEL_NAME best_task_2_4_cut1_pool_D1024_model_checkpoint.h5 --EPOCHS 300 --DATASET_DIR /ghome/group07/M3-Project-new/Week4/MIT_small_train_4



