#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32000M
#SBATCH --time=20-00:00
#SBATCH --output=RF_civil_uncivil.out

module load python/3.6

module load python scipy-stack

module load StdEnv/2020
module load gcc/9.3.0
module load cuda/11.0

source /home/isafer/scratch/cluster/ENV/bin/activate

module load scipy-stack

pip install nltk
pip install scikit-learn
pip install pandas
pip install numpy
pip install imblearn
pip install xlrd

python RF_civil_uncivil.py