#!/bin/bash

#SBATCH --mail-user=ria_vinod@brown.edu
#SBATCH --mail-type=ALL

#SBATCH --output=/users/rvinod/data/rvinod/batch_jobs/output-%j.out
#SBATCH --error=/users/rvinod/data/rvinod/batch_jobs/output-%j.err


#SBATCH -p gpu-he 
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH -N 1


#SBATCH --cpus-per-task=1
 
# Request an hour of runtime:
#SBATCH --time=100:00:00

#SBATCH --mem=50G

#SBATCH -J subnet-plm-1


# Run a command
ml cuda/12.2.0-4lgnkrh
ml anaconda/2023.09-0-7nso27y
source /gpfs/runtime/opt/anaconda/2023.03-1/etc/profile.d/conda.sh
source subnet-plm/bin/activate
python code/main.py bert_output/