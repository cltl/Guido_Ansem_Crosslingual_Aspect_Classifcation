#!/bin/bash
#SBATCH -N 1
#SBATCH -t 00:10:00
#SBATCH -p gpu_short
#SBATCH --gpus=1

module load 2020
module load Python/3.8.2-GCCcore-9.3.0

cp $HOME/English_restaurant_reviews_split_on_sentences.txt "$TMPDIR"

python main.py --batch_size 4 \
                --smoothing 0 \
                --lr 5e-5 \
                --epochs 2 \
                --data_file 'English_restaurant_reviews_split_on_sentences.txt'\
                --save_folder './'

cp -r "$TMPDIR"/train $HOME
cp -r "$TMPDIR"/val $HOME