#!/bin/bash
#SBATCH --job-name=paramTLtests
#SBATCH -t 0:30:00
#SBATCH --gres=gpu:1
#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --mem=16G

# load modules and activate env
module purge
module load python/3.8.2
source ~/venvs/3.8.2/bin/activate

# Launch your script, giving it as arguments the database file and the study name
python target_traditional_training.py

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s