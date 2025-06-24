#!/bin/bash

#SBATCH --job-name=bpic2017_ModernBERT
#SBATCH --output=my_job_output_bpic2017_%j.txt
#SBATCH --partition=tue.gpu.q         # Choose a partition that has GPUs
#SBATCH --time=150:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=1                      # This is how to request a GPU

module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# Execute the script or command
pip install scikit-learn
pip install transformers==4.51.1
nvidia-smi
# pip install --upgrade git+https://github.com/huggingface/transformers.git

# Create base output directory structure
BASE_DIR="datasets/bpic2017"  # $1 is the log name (e.g., bpic2017)
RUN_DIR="${BASE_DIR}/results_bpic2017_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

# Record job start time
JOB_START=$(date +%s.%N)

# Modify the run_with_timing function to pass the output directory:
run_with_timing() {
    script_name=$1
    shift  # Shift arguments to pass any additional parameters to the script
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
    echo "$script_name completed in $execution_time seconds (bpic2017)"

    # Process and display GPU statistics
    echo "GPU Statistics for $script_name:"
    awk -F, 'BEGIN {max=0; sum=0; count=0} 
        NR>1 {sum+=$2; if($2>max)max=$2; count++} 
        END {printf "Average VRAM usage: %.2f MB\nPeak VRAM usage: %.2f MB\n", sum/count, max}' "${RUN_DIR}/gpu_stats_${script_name}.log"
    echo "----------------------------------------"
}

# Run mam_stage.py first with timing
run_with_timing "mam_stage.py --log bpic2017"

# Rest of the model runs
run_with_timing "model_finetune.py --log bpic2017 --use_prompts true --use_prompt_updates true --focal_gamma 2.0 --mam_flag true --use_activity_head true --use_outcome_head true"
# run_with_timing "model_finetune.py --log bpic2017 --use_prompts true --use_prompt_updates true --focal_gamma 2.0 --mam_flag true --use_activity_head true --use_outcome_head false"
# run_with_timing "model_finetune.py --log bpic2017 --use_prompts true --use_prompt_updates true --focal_gamma 2.0 --mam_flag true --use_activity_head false --use_outcome_head true"
# run_with_timing "model_finetune.py --log bpic2017 --use_prompts true --use_prompt_updates true --focal_gamma 2.0 --mam_flag false --use_activity_head true --use_outcome_head true"
# run_with_timing "model_finetune.py --log bpic2017 --use_prompts true --use_prompt_updates true --focal_gamma 2.0 --mam_flag false --use_activity_head true --use_outcome_head false"
# run_with_timing "model_finetune.py --log bpic2017 --use_prompts true --use_prompt_updates true --focal_gamma 2.0 --mam_flag false --use_activity_head false --use_outcome_head true"
# run_with_timing "model_finetune.py --log bpic2017 --use_prompts false --use_prompt_updates false --focal_gamma 2.0 --mam_flag true --use_activity_head true --use_outcome_head true"
# run_with_timing "model_finetune.py --log bpic2017 --use_prompts false --use_prompt_updates false --focal_gamma 2.0 --mam_flag true --use_activity_head true --use_outcome_head false"
# run_with_timing "model_finetune.py --log bpic2017 --use_prompts false --use_prompt_updates false --focal_gamma 2.0 --mam_flag true --use_activity_head false --use_outcome_head true"
# run_with_timing "model_finetune.py --log bpic2017 --use_prompts false --use_prompt_updates false --focal_gamma 2.0 --mam_flag false --use_activity_head true --use_outcome_head true"
# run_with_timing "model_finetune.py --log bpic2017 --use_prompts false --use_prompt_updates false --focal_gamma 2.0 --mam_flag false --use_activity_head true --use_outcome_head false"
# run_with_timing "model_finetune.py --log bpic2017 --use_prompts false --use_prompt_updates false --focal_gamma 2.0 --mam_flag false --use_activity_head false --use_outcome_head true"



# run_with_timing "model_finetune.py bpic2017 true true 2.0 true true false"
# run_with_timing "model_finetune.py bpic2017 true true 2.0 true false true"
# run_with_timing "model_finetune.py bpic2017 true true 2.0 false true true"
# run_with_timing "model_finetune.py bpic2017 true true 2.0 false true false"
# run_with_timing "model_finetune.py bpic2017 true true 2.0 false false true"

# run_with_timing "model_finetune.py bpic2017 false false 2.0 true true true"
# run_with_timing "model_finetune.py bpic2017 false false 2.0 true true false"
# run_with_timing "model_finetune.py bpic2017 false false 2.0 true false true"
# run_with_timing "model_finetune.py bpic2017 false false 2.0 false true true"
# run_with_timing "model_finetune.py bpic2017 false false 2.0 false true false"
# run_with_timing "model_finetune.py bpic2017 false false 2.0 false false true"

# python eval_only.py bpic2017
# python train.py bpic2017
# python eval.py bpic2017

# Calculate total job time
JOB_END=$(date +%s.%N)
JOB_DURATION=$(echo "$JOB_END - $JOB_START" | bc)
echo "Total job duration: $JOB_DURATION seconds (bpic2017)"

# Print Slurm job statistics summary
echo "Slurm job statistics:"
sacct -j $SLURM_JOB_ID --format=JobID,JobName,Elapsed,MaxRSS,State
