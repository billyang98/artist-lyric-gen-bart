#!/bin/bash
#-----------------------------------------------------------------
# Example SLURM job script to run serial applications on TACC's
# Stampede system.
#
# This script requests one core (out of 16) on one node. The job
# will have access to all the memory in the node.  Note that this
# job will be charged as if all 16 cores were requested.
#-----------------------------------------------------------------

#SBATCH -J artist-gen-bart             # Job name
#SBATCH -o /work/05347/billyang/artist-lyric-gen-bart/out_dir/%j.out       # Specify stdout output file (%j expands to jobId)
#SBATCH -p gtx                   # Queue name: gtx or v100 or p100 -- check availability using "sinfo" command
#SBATCH -N 1                     # Total number of nodes requested (16 cores/node)
#SBATCH -n 1                     # Total number of tasks
#SBATCH -t 15:00:00              # Run time (hh:mm:ss)

sbatch -A CS378-NLP-sp20      # Specify allocation to charge against

# Load any necessary modules (these are examples)
# Loading modules in the script ensures a consistent environment.
module load python3
module load cuda

# Launch jobs
source /home1/05347/billyang/nlp_artist_gen/bin/activate   # Activate your virtual env here.
cd /work/05347/billyang/artist-lyric-gen-bart     # Move to your working dir.


# Training a baseline model from scratch
python bart.py --training_file support_files/train_dataset_id.csv --model_name bart_id --max_length 102
