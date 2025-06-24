#!/bin/bash

#SBATCH --job-name=CRTP-LSTM_application
#SBATCH --output=my_job_output_%j.txt
#SBATCH --partition=tue.gpu.q         # Choose a partition that has GPUs
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=1                      # This is how to request a GPU

module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0

# Execute the script or command
pip install pandas plotly scikit-learn

# Create base output directory structure
BASE_DIR="datasets/application"  # $1 is the log name (e.g., application)
RUN_DIR="${BASE_DIR}/results_application_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

# Record job start time
JOB_START=$(date +%s.%N)

# Function to time individual script execution
run_with_timing() {
    script_name=$1
    shift  # Remove the script name from the arguments
    echo "----------------------------------------"
    echo "Starting $script_name at $(date)"
    
    # Start GPU monitoring in background
    (
        while true; do
            nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv,nounits >> "${RUN_DIR}/gpu_stats_${script_name}.log"
            sleep 1
        done
    ) &
    MONITOR_PID=$!

    start_time=$(date +%s.%N)
    python $script_name --output_dir "$RUN_DIR" "$@"
    end_time=$(date +%s.%N)

    # Stop GPU monitoring
    kill $MONITOR_PID

    execution_time=$(echo "$end_time - $start_time" | bc)
    echo "$script_name completed in $execution_time seconds (application)"

    # Process and display GPU statistics
    echo "GPU Statistics for $script_name:"
    awk -F, 'BEGIN {max=0; sum=0; count=0} 
        NR>1 {sum+=$2; if($2>max)max=$2; count++} 
        END {printf "Average VRAM usage: %.2f MB\nPeak VRAM usage: %.2f MB\n", sum/count, max}' "${RUN_DIR}/gpu_stats_${script_name}.log"
    echo "----------------------------------------"
}

run_with_timing "preprocess.py --log application"
run_with_timing "train.py --log application"
run_with_timing "eval.py --log application"

# Calculate total job time
JOB_END=$(date +%s.%N)
JOB_DURATION=$(echo "$JOB_END - $JOB_START" | bc)
echo "Total job duration: $JOB_DURATION seconds (application)"

# Print Slurm job statistics summary
echo "Slurm job statistics:"
sacct -j $SLURM_JOB_ID --format=JobID,JobName,Elapsed,MaxRSS,State