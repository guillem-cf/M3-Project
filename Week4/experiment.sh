#!/bin/bash
#SBATCH -n 4 # Number of cores
#SBATCH --mem 2000 # 2GB solicitados.
#SBATCH -p mhigh,mhigh # or mlow Partition to submit to master low prioriy queue
#SBATCH --gres gpu:1 # Para pedir Pascales MAX 8
#SBATCH -o %x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e %x_%u_%j.err # File to which STDERR will be written

eval "$(conda shell.bash hook)"
conda activate m3
#python w4code_example_backup_profe.py
python 1.baseline.py --experiment_name baseline_1 --MODEL_START 1
python 1.baseline.py --experiment_name baseline_2 --MODEL_START 2
python 1.baseline.py --experiment_name model_11 --MODEL_START 1 --MODEL_HID 1024
python 1.baseline.py --experiment_name model_12 --MODEL_START 2 --MODEL_HID 1024
python 1.baseline.py --experiment_name model_21 --MODEL_START 1 --MODEL_HID 512
python 1.baseline.py --experiment_name model_22 --MODEL_START 2 --MODEL_HID 512
