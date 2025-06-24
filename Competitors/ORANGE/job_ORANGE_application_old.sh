#!/bin/bash

#SBATCH --job-name=ORANGE
#SBATCH --output=my_job_output_%j.txt
#SBATCH --partition=tue.gpu.q         # Choose a partition that has GPUs
#SBATCH --time=50:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=1                      # This is how to request a GPU

module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0

# Execute the script or command
# pip install pandas plotly scikit-learn

# Record job start time
JOB_START=$(date +%s.%N)

# Function to time individual script execution
run_with_timing() {
    script_name=$1
    echo "Starting $script_name at $(date)"
    start_time=$(date +%s.%N)
    python $script_name
    end_time=$(date +%s.%N)
    execution_time=$(echo "$end_time - $start_time" | bc)
    echo "$script_name completed in $execution_time seconds (application)"
}

python preprocess.py application
python ae.py application 200 64
python generate_image.py application
python nn.py application 0.0001 64
python load_weights.py application
python roc_auc_plotting.py application
python eval_pred.py application

# Calculate total job time
JOB_END=$(date +%s.%N)
JOB_DURATION=$(echo "$JOB_END - $JOB_START" | bc)
echo "Total job duration: $JOB_DURATION seconds (application)"

# Print Slurm job statistics summary
echo "Slurm job statistics:"
sacct -j $SLURM_JOB_ID --format=JobID,JobName,Elapsed,MaxRSS,State
