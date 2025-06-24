#!/bin/bash

#SBATCH --job-name=BERT_bpic2017
#SBATCH --output=my_job_output_%j.txt
#SBATCH --partition=tue.gpu.q
#SBATCH --time=80:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=1

module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

pip install transformers scikit-learn

# Create base output directory structure
BASE_DIR="datasets/bpic2017"  # $1 is the log name (e.g., bpic2017)
RUN_DIR="${BASE_DIR}/results_bpic2017_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

JOB_START=$(date +%s.%N)

# Enhanced function to monitor GPU usage and timing
run_with_monitoring() {
    script_name=$1
    shift  # Shift arguments to pass additional parameters to the script
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

    # Run the script and time it
    start_time=$(date +%s.%N)
    python $script_name --output_dir "$RUN_DIR" "$@"
    end_time=$(date +%s.%N)
    
    # Stop GPU monitoring
    kill $MONITOR_PID

    # Calculate execution time
    execution_time=$(echo "$end_time - $start_time" | bc)
    echo "$script_name completed in $execution_time seconds"

    # Process and display GPU statistics
    echo "GPU Statistics for $script_name:"
    awk -F, 'BEGIN {max=0; sum=0; count=0} 
        NR>1 {sum+=$2; if($2>max)max=$2; count++} 
        END {printf "Average VRAM usage: %.2f MB\nPeak VRAM usage: %.2f MB\n", sum/count, max}' "${RUN_DIR}/gpu_stats_${script_name}.log"
    echo "----------------------------------------"
}

# Run each script with monitoring
run_with_monitoring "data_processor.py --log bpic2017"
run_with_monitoring "MAM_v2.py --log bpic2017"
run_with_monitoring "nap_finetuning.py --log bpic2017"
run_with_monitoring "outcome_finetuning.py --log bpic2017"

# Calculate total job time
JOB_END=$(date +%s.%N)
JOB_DURATION=$(echo "$JOB_END - $JOB_START" | bc)
echo "Total job duration: $JOB_DURATION seconds (bpic2017)"

# Print Slurm job statistics summary
echo "Slurm job statistics:"
sacct -j $SLURM_JOB_ID --format=JobID,JobName,Elapsed,MaxRSS,State

# Cleanup GPU monitoring logs
rm gpu_stats_*.log

