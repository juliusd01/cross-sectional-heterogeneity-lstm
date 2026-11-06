#!/bin/bash
#SBATCH --job-name=tabular
#SBATCH --output=slurm/logs/tabular_%j.out
#SBATCH --time=23:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G

module purge
module load python/3.11

export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Random forest and gradient boosting benchmarks.  Edit the `config` dict at
# the top of main() to switch between 'rf' and 'hgb' and to vary the size.
python -u -m train_tabular
