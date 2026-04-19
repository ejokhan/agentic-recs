#!/bin/bash
#SBATCH -J embed-wands
#SBATCH -o logs/embed-%j.out
#SBATCH -e logs/embed-%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -A TG-NAIRR240491

cd /home1/10655/ejokhan123/agentic-recs

source /work/10655/ejokhan123/ls6/miniconda3/bin/activate
conda activate rag

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export HF_HOME=/work/10655/ejokhan123/ls6/hydro-rag/.hf_cache

echo "=== Node: $(hostname) ==="
echo "=== Start: $(date) ==="
echo "=== Python: $(which python) ==="
echo "=== CUDA visible: ==="
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
echo "=== Model cache check: ==="
ls /work/10655/ejokhan123/ls6/hydro-rag/.hf_cache/models--BAAI--bge-large-en-v1.5/snapshots/*/model.safetensors
echo "=== Products check: ==="
ls -la data/processed/products.parquet
echo "=== Starting embedding script ==="

python src/embeddings/embed_products.py

echo "=== Exit code: $? ==="
echo "=== End: $(date) ==="
echo "=== Output files: ==="
ls -la data/embeddings/
