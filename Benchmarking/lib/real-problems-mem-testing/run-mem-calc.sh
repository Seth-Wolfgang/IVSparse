#!/bin/bash			
#SBATCH --job-name=memcalc_realprobs
#SBATCH -p bigmem 
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH -o output.out
#SBATCH -e output.err
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=FILLIN
#SBATCH --mem=0
#SBATCH --exclusive

source ~/.bashrc	
workon vcsc
srun python mem-calc-real-problems.py
