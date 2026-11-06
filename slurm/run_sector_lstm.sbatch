#!/bin/bash
#SBATCH --job-name=sector_lstm
#SBATCH --output=slurm/logs/sector_lstm_%j.out
#SBATCH --time=23:59:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:nvd:1

module purge
module load python/3.11

export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Sector LSTM -- the main specification of the paper.
python -u -m train \
    --model_architecture heterogeneity \
    --input_subdir lag_60_sector \
    --model_type lstm_sector \
    --experiment_type heterogeneity \
    --num_layers 3 \
    --hidden_size 25 \
    --learning_rate 0.001 \
    --batch_size 64 \
    --num_lags 60 \
    --embedding_dim 2 \
    --dropout 0.1 \
    --job_id $SLURM_JOB_ID
