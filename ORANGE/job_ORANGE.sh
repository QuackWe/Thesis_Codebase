#!/bin/bash

#SBATCH --job-name=my_job
#SBATCH --output=my_job_output_%j.txt
#SBATCH --partition=tue.gpu.q         # Choose a partition that has GPUs
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=1                      # This is how to request a GPU

module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0

# Execute the script or command
# pip install pandas plotly scikit-learn
# python ae.py mortgages 200 64
# python generate_image.py mortgages
# python nn.py mortgages 0.0001 64
python load_weights.py mortgages