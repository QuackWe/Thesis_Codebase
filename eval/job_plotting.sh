#!/bin/bash

#SBATCH --job-name=eval_pipeline
#SBATCH --output=my_job_output_%j.txt
#SBATCH --partition=tue.default.q         # Choose a partition that has GPUs
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G


module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

nvcc --version

# Execute the script or command
# python plot_trace_lengths.py mortgages
# python plot_performance_metrics.py mortgages
# python plot_per_epoch.py                        #TODO: add acc and f1 here per epoch
# # python plot_bucket_performance.py mortgages
# python plot_prefix_performance.py mortgages

pip install matplotlib seaborn numpy scikit-learn tabulate
# python pref_len.py
python unified_eval_pipeline.py mortgages 
python unified_eval_pipeline.py application
python unified_eval_pipeline.py bpic2017 





























# python plot_outcome_comp.py mortgages
# python plot_nap_comp.py mortgages

# python unified_eval_pipeline.py bpic2017 BERT
# python unified_eval_pipeline.py bpic2017 CRTP-LSTM_without-time_good
# python unified_eval_pipeline.py bpic2017 ORANGE
# python unified_eval_pipeline.py bpic2017 ModernBERT
# python plot_outcome_comp.py bpic2017
# python plot_nap_comp.py bpic2017

# python unified_eval_pipeline.py application BERT
# python unified_eval_pipeline.py application CRTP-LSTM_without-time_good
# python unified_eval_pipeline.py application ORANGE
# python unified_eval_pipeline.py application ModernBERT
# python plot_outcome_comp.py application
# python plot_nap_comp.py application

# python plot_prefix_metrics.py application
# python plot_prefix_metrics.py bpic2017



