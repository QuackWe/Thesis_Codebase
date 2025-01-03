#!/bin/bash

#SBATCH --job-name=my_job
#SBATCH --output=my_job_output_%j.txt
#SBATCH --partition=tue.gpu.q         # Choose a partition that has GPUs
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=1                      # This is how to request a GPU

module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# Execute the script or command
pip install transformers scikit-learn
# python data_processor.py mortgages
python MAM_v2.py mortgages
python nap_finetuning.py mortgages
python outcome_finetuning.py mortgages