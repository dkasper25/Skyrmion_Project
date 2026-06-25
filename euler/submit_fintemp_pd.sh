#!/bin/bash
#SBATCH --job-name=fintemp_sim      # Job name
#SBATCH --time=04:00:00             # Max wall-clock time (HH:MM:SS)
#SBATCH --ntasks=1                  # Number of tasks (typically 1 for a single python script)
#SBATCH --cpus-per-task=24          # Number of CPU cores per task
#SBATCH --mem-per-cpu=2G            # Memory per CPU core
#SBATCH --output=fintemp_%j.out     # Standard output log (%j = job ID)
#SBATCH --error=fintemp_%j.err      # Standard error log

# Note: You should run this from your $SCRATCH directory on Euler for better performance!
# Example: cd $SCRATCH/Skyrmion_Project

# Load the Euler software stack and Python
module load stack/2024-06
module load python/3.11.6

# Activate the virtual environment
source venv/bin/activate

# CRITICAL: Prevent JAX from spanning infinite threads within multiprocessing workers!
# Setting these at the shell level ensures they are present in the environment block
# when the C++ static initializers of jaxlib are loaded, and are inherited by spawned subprocesses.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export JAX_CPU_DEFAULT_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

# Execute the modified phase diagram generation script. 
# We run it from the euler subdirectory, it will save the .npz data and skip plotting.
python euler/fintemp_phase_diagram_euler.py --T 0.01 --nH 26 --nA 33 --steps 5000 --workers 24 --iso-scale --standard-a 0.2 --grid-round 8

