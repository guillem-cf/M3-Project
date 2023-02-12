#!/bin/bash
#SBATCH -n 4 # Number of cores
#SBATCH --mem 2000 # 2GB solicitados.
#SBATCH -p mhigh,mhigh # or mlow Partition to submit to master low prioriy queue
#SBATCH --gres gpu:1 # Para pedir Pascales MAX 8
#SBATCH -o %x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e %x_%u_%j.err # File to which STDERR will be written

eval "$(conda shell.bash hook)"
conda activate m3


# python main.py --experiment_name medium_extended --MODEL medium_extended --IMG_WIDTH 64 --IMG_HEIGHT 64   # job 8071   accuracy 0.5939

# python main.py --experiment_name medium_2blocks --MODEL medium_256input_2blocks --IMG_WIDTH 64 --IMG_HEIGHT 64  # job 8073

# python main.py --experiment_name medium_extended_HF_Z --MODEL medium_extended --IMG_WIDTH 64 --IMG_HEIGHT 64 --horizontal_flip True --zoom_range 0.2 

# python main.py --experiment_name medium_extended_HF_Z_R --MODEL medium_extended --IMG_WIDTH 64 --IMG_HEIGHT 64 --horizontal_flip True --zoom_range 0.2 --rotation 10  # job 8074


python main.py --config config/best_model.yaml




