#!/bin/bash
#SBATCH -J embed-wands
#SBATCH -o logs/embed-%j.out
#SBATCH -e logs/embed-%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -A TG-NAIRR240491

set -euo pipefail
cd /home1/10655/ejokhan123/agentic-recs

source /work/10655/ejokhan123/ls6/miniconda3/bin/activate
conda activate rag
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export HF_HOME=$PWD/.hf_cache

echo "=== Node: $(hostname) ==="
echo "=== Start: $(date) ==="
nvidia-smi | head -20

python src/embeddings/embed_products.py

echo "=== End: $(date) ==="
