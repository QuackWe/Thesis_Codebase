#!/bin/bash

#SBATCH --job-name=BERT
#SBATCH --output=my_job_output_%j.txt
#SBATCH --partition=mcs.gpu.q         # Choose a partition that has GPUs
#SBATCH --time=200:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=1                      # This is how to request a GPU

module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# Execute the script or command
pip install transformers scikit-learn

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
    echo "$script_name completed in $execution_time seconds"
}

python data_processor.py mortgages
# python seq_len.py
python MAM_v2.py mortgages
python nap_finetuning.py mortgages
python outcome_finetuning.py mortgages

# Calculate total job time
JOB_END=$(date +%s.%N)
JOB_DURATION=$(echo "$JOB_END - $JOB_START" | bc)
echo "Total job duration: $JOB_DURATION seconds (mortgages)"

# Print Slurm job statistics summary
echo "Slurm job statistics:"
sacct -j $SLURM_JOB_ID --format=JobID,JobName,Elapsed,MaxRSS,State