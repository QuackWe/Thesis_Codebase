#!/bin/bash

#SBATCH --job-name=my_job
#SBATCH --output=my_job_output_%j.txt
#SBATCH --partition=mcs.default.q	         # Choose a partition that has GPUs
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G

module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# Execute the script or command
pip install transformers scikit-learn
# python preprocess.py mortgages
# python train.py mortgages
python eval.py mortgages
# python test_prompts.py
# python test_model_forward.py
