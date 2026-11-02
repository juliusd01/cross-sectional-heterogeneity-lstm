#!/bin/bash
#SBATCH --job-name=covariates_lstm
#SBATCH --output=slurm/logs/covariates_lstm_%j.out
#SBATCH --time=23:59:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:nvd:1

module purge
module load python/3.11

export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Covariate LSTM.  Uses the dual-dropout trainer: without input dropout the
# validation loss diverges (section 2.4.3).
# Add --var_left_out vix to reproduce the "without VIX" row of Table 6.
python -u -m lstm_sector.train_dropout \
    --model_architecture covariates \
    --input_subdir lag_60_cov \
    --model_type lstm_cov \
    --experiment_type covariates \
    --num_layers 1 \
    --hidden_size 25 \
    --learning_rate 0.001 \
    --batch_size 256 \
    --num_lags 60 \
    --recurrent_dropout 0.1 \
    --input_dropout 0.1 \
    --job_id $SLURM_JOB_ID
