#!/bin/bash
#SBATCH --job-name=neo_streets
#SBATCH --partition=short
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --array=0-6
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# One array task per city. Submit from this folder:
#   mkdir -p logs && sbatch run_many_streets.sh
# Single city only, e.g. Cape Town:  sbatch --array=3 run_many_streets.sh
#
# Resources measured on 1 seed (3 methods): <= 18 min, <= 3.9 GB, ~1 core busy.
# Several seeds per city: runtime grows linearly, so also raise the time limit, e.g. 5 seeds:
#   sbatch --partition=standard --time=02:30:00 --export=ALL,INSTANCES=5 run_many_streets.sh

CITIES=("New York" "Zurich" "Beijing" "Cape Town" "New Delhi" "London" "Paris")
CITY="${CITIES[$SLURM_ARRAY_TASK_ID]}"
INSTANCES=${INSTANCES:-1}

# keep numpy/scipy from grabbing every core on the node
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

source /work/cewallar/miniconda3/etc/profile.d/conda.sh
conda activate neo
# ignore packages in ~/.local (e.g. coffea pins numpy<2)
export PYTHONNOUSERSITE=1
# result pickles are large -> keep them on /work, not in the (10 GB) home quota
export NEO_RESULTS_DIR=${NEO_RESULTS_DIR:-/work/cewallar/PhD_UZH/Reinforcement_Learning_NEO/many_streets_results}
mkdir -p "$NEO_RESULTS_DIR"

cd "$SLURM_SUBMIT_DIR"
echo "City: $CITY  seeds: $INSTANCES  node: $(hostname)  start: $(date)"
python -u Many_Streets_Eval.py "$CITY" --instances "$INSTANCES"
status=$?
echo "end: $(date)  exit code: $status"
exit $status
