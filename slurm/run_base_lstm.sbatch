#!/bin/bash
#SBATCH --job-name=base_lstm
#SBATCH --output=slurm/logs/base_lstm_%j.out
#SBATCH --time=23:59:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:nvd:1

module purge
module load python/3.11

export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Base LSTM, returns only.  Table 6 also reports 1, 2 and 5 layers.
python -u -m lstm_sector.train \
    --model_architecture base \
    --input_subdir lag_60 \
    --model_type lstm \
    --experiment_type complexity \
    --num_layers 3 \
    --hidden_size 25 \
    --learning_rate 0.001 \
    --batch_size 256 \
    --num_lags 60 \
    --dropout 0.1 \
    --job_id $SLURM_JOB_ID
