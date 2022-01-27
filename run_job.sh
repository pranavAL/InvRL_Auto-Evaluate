#!/bin/bash
#SBATCH --time=00:60:00
#SBATCH --account=def-ebrahimi
#SBATCH --mem=4000M
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=pranav2109@hotmail.com
#SBATCH --mail-type=ALL
source ~/myenv/bin/activate
python train_LSTM.py -sq 32
deactivate
