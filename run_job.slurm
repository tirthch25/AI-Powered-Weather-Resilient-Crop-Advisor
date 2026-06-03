#!/bin/bash
#─────────────────────────────────────────────────────────────────────────────
# SLURM Job Script — Indian Farmer Crop Recommendation System
# Target System : PARAM Shavak (C-DAC Supercomputer)
# OS            : BOSS Linux (Bharat Operating System Solutions)
#
# HOW TO USE:
#   1. Replace <YOUR_USERNAME> with your actual Param Shavak username
#   2. Check available partitions on login node:  sinfo
#   3. Check available modules:                   module avail
#   4. Submit with:  sbatch run_job.slurm
#   5. Check status: squeue -u <YOUR_USERNAME>
#   6. View output:  cat logs/output_<job-id>.log
#─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=crop_recommender
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.log
#SBATCH --partition=gpu              # Run: sinfo  to verify partition name on your system

# ── Change this to your actual Param Shavak username ─────────────────────────
HPC_USER="<YOUR_USERNAME>"
PROJECT_DIR="/scratch/${HPC_USER}/Indian-Farmer-Crop-Recommendation-System"
# ─────────────────────────────────────────────────────────────────────────────

echo "========================================"
echo "  Indian Farmer Crop Recommendation"
echo "  Running on PARAM Shavak (C-DAC)"
echo "  Date     : $(date)"
echo "  Node     : $(hostname)"
echo "  Job ID   : ${SLURM_JOB_ID}"
echo "========================================"

# ── Load Modules ──────────────────────────────────────────────────────────────
# NOTE: Run  module avail  on the login node to see exact module names
#       Common Param Shavak module names are listed below — uncomment what works

# Option A: If Anaconda/Miniconda is available as a module
module load anaconda3               # Try: module spider anaconda  to find exact name

# Option B: If CUDA is needed for GPU (PyTorch)
module load cuda                    # Try: module spider cuda  to find exact version

# Option C: If Python is available as a standalone module (no conda)
# module load python/3.10

echo ""
echo "Loaded modules:"
module list
echo ""

# ── Activate Conda Environment ────────────────────────────────────────────────
# If conda is not auto-initialized, source it manually:
if ! command -v conda &>/dev/null; then
    # Common Param Shavak conda locations — uncomment the correct one:
    # source /opt/ohpc/pub/apps/anaconda3/etc/profile.d/conda.sh
    # source /home/${HPC_USER}/miniconda3/etc/profile.d/conda.sh
    source /scratch/${HPC_USER}/miniconda3/etc/profile.d/conda.sh
fi

conda activate crop_env

# ── Move to Project Directory ─────────────────────────────────────────────────
cd "${PROJECT_DIR}/agri_crop_recommendation"

# Create logs directory if not present
mkdir -p logs

# ── Pre-flight Checks ─────────────────────────────────────────────────────────
echo "Working directory : $(pwd)"
echo "Python            : $(which python)"
echo "Python version    : $(python --version)"
echo ""

# Verify .env file exists
if [ ! -f ".env" ]; then
    echo "ERROR: .env file not found!"
    echo "Run:  cp .env.example .env  and fill in your GEMINI_API_KEY"
    exit 1
fi

# Print GPU info (if available)
if command -v nvidia-smi &>/dev/null; then
    echo "GPU Info:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo ""
fi

echo "Running: python main.py"
echo "─────────────────────────────────────────"

# ── Run the Application ───────────────────────────────────────────────────────
python main.py

EXIT_CODE=$?

echo ""
echo "========================================"
echo " Job Finished"
echo " Exit Code : ${EXIT_CODE}"
echo " Date      : $(date)"
echo "========================================"

exit ${EXIT_CODE}
