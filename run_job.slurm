#!/bin/bash
#─────────────────────────────────────────────────────────────────────────────
# SLURM Job Script — AI-Powered Weather-Resilient Crop Advisor
# Target System : PARAM Shavak (C-DAC Supercomputer)
# OS            : BOSS Linux (Bharat Operating System Solutions)
#
# HOW TO USE:
#   1. Replace <YOUR_USERNAME> below with your actual Param Shavak login name
#   2. Run  sinfo  on the login node to find the correct partition name
#   3. Run  module avail  on the login node to find module names
#   4. Submit with:  sbatch run_job.slurm
#   5. Check status: squeue -u <YOUR_USERNAME>
#   6. View output:  cat logs/output_<job-id>.log
#─────────────────────────────────────────────────────────────────────────────

#── SLURM Directives ──────────────────────────────────────────────────────────
#SBATCH --job-name=crop_advisor           # Name shown in squeue
#SBATCH --nodes=1                         # 1 compute node
#SBATCH --ntasks=1                        # 1 task (single process)
#SBATCH --cpus-per-task=8                 # 8 CPU cores (good for FastAPI + ML)
#SBATCH --gres=gpu:1                      # 1 GPU for PyTorch (remove if no GPU needed)
#SBATCH --mem=32G                         # 32 GB RAM
#SBATCH --time=04:00:00                   # Max 4 hours (HH:MM:SS)
#SBATCH --output=logs/output_%j.log       # stdout log (created in project dir)
#SBATCH --error=logs/error_%j.log         # stderr log
#SBATCH --partition=gpu                   # ← RUN sinfo TO VERIFY THIS NAME!
                                          #   Common: gpu, compute, batch, standard

#── ⚠️  CHANGE THIS ───────────────────────────────────────────────────────────
HPC_USER="<YOUR_USERNAME>"
PROJECT_ROOT="/scratch/${HPC_USER}/AI-Powered-Weather-Resilient-Crop-Advisor"
PROJECT_DIR="${PROJECT_ROOT}/agri_crop_recommendation"
#─────────────────────────────────────────────────────────────────────────────

echo "========================================================"
echo "  AI-Powered Weather-Resilient Crop Advisor"
echo "  Running on PARAM Shavak (C-DAC)"
echo "  Date     : $(date)"
echo "  Node     : $(hostname)"
echo "  Job ID   : ${SLURM_JOB_ID}"
echo "  Cores    : ${SLURM_CPUS_PER_TASK}"
echo "========================================================"

#── Load Required Modules ─────────────────────────────────────────────────────
# NOTE: Run  module avail  on login node to find exact names on your system.
#       The examples below are common on C-DAC / BOSS Linux clusters.

# Load Anaconda3 (try: module spider anaconda  to find the exact name)
module load anaconda3
# If the above fails, try one of:
# module load miniconda3
# module load python/3.11

# Load CUDA for GPU support (needed for PyTorch GPU acceleration)
# Remove this line if you are running CPU-only (no GPU requested above)
module load cuda
# If the above fails, try: module load cuda/12.0  or  module load cuda/11.8

echo ""
echo "Loaded modules:"
module list
echo ""

#── Activate Conda Environment ────────────────────────────────────────────────
# If conda is not initialized automatically in the batch environment,
# source it manually. Try one of these paths — comment out the ones that fail:

if ! command -v conda &>/dev/null; then
    # Common Param Shavak / C-DAC conda locations:
    if   [ -f "/opt/ohpc/pub/apps/anaconda3/etc/profile.d/conda.sh" ]; then
        source /opt/ohpc/pub/apps/anaconda3/etc/profile.d/conda.sh
    elif [ -f "/home/${HPC_USER}/miniconda3/etc/profile.d/conda.sh" ]; then
        source /home/${HPC_USER}/miniconda3/etc/profile.d/conda.sh
    elif [ -f "/scratch/${HPC_USER}/miniconda3/etc/profile.d/conda.sh" ]; then
        source /scratch/${HPC_USER}/miniconda3/etc/profile.d/conda.sh
    else
        echo "WARNING: Could not find conda. Using system Python."
    fi
fi

conda activate crop_env
echo "Python: $(which python)  ($(python --version))"

#── Go to Project Directory ───────────────────────────────────────────────────
cd "${PROJECT_DIR}" || { echo "ERROR: Project dir not found: ${PROJECT_DIR}"; exit 1; }

# Create logs directory
mkdir -p logs

#── Pre-flight Checks ─────────────────────────────────────────────────────────
echo ""
echo "Working directory : $(pwd)"
echo ""

# Verify .env exists
if [ ! -f ".env" ]; then
    echo "ERROR: .env file not found!"
    echo "Run:  cp .env.example .env  and fill in your GEMINI_API_KEY"
    exit 1
fi

# Verify GEMINI_API_KEY is set (needed for states dropdown + LLM features)
if grep -q "GEMINI_API_KEY=<" .env || ! grep -q "GEMINI_API_KEY=" .env; then
    echo "WARNING: GEMINI_API_KEY appears unset in .env"
    echo "         States/districts dropdown and LLM features will not work."
    echo "         Get a free key at: https://aistudio.google.com/app/apikey"
fi

# Verify LLM_PROVIDER is set to gemini (Ollama cannot run on HPC batch nodes)
if grep -q "LLM_PROVIDER=ollama" .env; then
    echo "WARNING: LLM_PROVIDER=ollama detected in .env"
    echo "         Ollama cannot run on HPC batch nodes."
    echo "         Change to: LLM_PROVIDER=gemini"
fi

# Print GPU info (if available)
if command -v nvidia-smi &>/dev/null; then
    echo "GPU Info:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo ""
fi

# Quick Python package check
echo "Checking key packages..."
python -c "
import fastapi, uvicorn, pandas, numpy, torch, sklearn, xgboost
print(f'  FastAPI   : {fastapi.__version__}')
print(f'  PyTorch   : {torch.__version__}  (GPU: {torch.cuda.is_available()})')
print(f'  Pandas    : {pandas.__version__}')
print(f'  XGBoost   : {xgboost.__version__}')
print(f'  Scikit-learn: {sklearn.__version__}')
print('All packages OK.')
" 2>&1 || { echo "ERROR: Package check failed. Run: pip install -r requirements.txt"; exit 1; }

echo ""
echo "Running application..."
echo "────────────────────────────────────────────────────"

#── Run the Application ───────────────────────────────────────────────────────
# Option A: Run the quick sanity test (default)
python main.py

# Option B: Run the full web server (uncomment this and comment out the line above)
# python run_website.py
# Note: To access the web UI, use SSH port-forwarding from your PC:
#   ssh -L 8000:localhost:8000 <YOUR_USERNAME>@<hpc-address>
#   Then open: http://localhost:8000

EXIT_CODE=$?

echo ""
echo "========================================================"
echo " Job Finished"
echo " Exit Code : ${EXIT_CODE}"
echo " Date      : $(date)"
echo "========================================================"

exit ${EXIT_CODE}
