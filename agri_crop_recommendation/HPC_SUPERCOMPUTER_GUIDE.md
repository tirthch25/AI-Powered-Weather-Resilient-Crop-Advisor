# 🖥️ Supercomputer / HPC Migration Guide
## Indian Farmer Crop Recommendation System

This file tells you **exactly what changes to make** to run this project on a
supercomputer or HPC cluster (e.g., PARAM Shakti, PARAM Siddhi, or any SLURM-based
cluster).

---

## 📋 TABLE OF CONTENTS

1. [Quick Checklist](#quick-checklist)
2. [Step 1 — Copy Your Project to HPC](#step-1--copy-your-project-to-hpc)
3. [Step 2 — Set Up Python Environment](#step-2--set-up-python-environment)
4. [Step 3 — Change the .env File](#step-3--change-the-env-file)
5. [Step 4 — Handle Ollama (LLM)](#step-4--handle-ollama-llm)
6. [Step 5 — Create a SLURM Job Script](#step-5--create-a-slurm-job-script)
7. [Step 6 — Run the Web Server (Optional)](#step-6--run-the-web-server-optional)
8. [Files You Must Change](#files-you-must-change)
9. [Files You Do NOT Need to Change](#files-you-do-not-need-to-change)
10. [Common Errors and Fixes](#common-errors-and-fixes)

---

## ✅ QUICK CHECKLIST

Use this list to track your progress:

- [ ] Copied project files to HPC using `scp` or `rsync`
- [ ] Created a Conda or venv environment on HPC
- [ ] Installed all dependencies from `requirements.txt`
- [ ] Updated `.env` file with correct settings for HPC
- [ ] Switched LLM from Ollama → Gemini API (recommended for HPC)
- [ ] Created and tested a SLURM job script (`run_job.slurm`)
- [ ] Submitted job with `sbatch run_job.slurm`
- [ ] Verified output in the `logs/` folder

---

## STEP 1 — Copy Your Project to HPC

On your **Windows PC**, open PowerShell or Command Prompt:

```powershell
# Replace <your-username> and <hpc-address> with your actual HPC login
scp -r "D:\CDAC\Indian-Farmer-Crop-Recommendation-System" <your-username>@<hpc-address>:/scratch/<your-username>/
```

Or use **WinSCP** (GUI tool) if you prefer drag and drop.

> **Where to put it:**  Use `/scratch/<your-username>/` — this is the fast
> working storage on most HPC clusters. Do NOT put it in your home directory
> (`~/`) as it has very low storage limits.

---

## STEP 2 — Set Up Python Environment

After logging into HPC via SSH:

### Option A: Using Conda (Recommended)
```bash
# Load Conda module (command may differ on your HPC — ask your HPC admin)
module load anaconda3/2023.09

# Create a new environment
conda create -n crop_env python=3.11 -y

# Activate it
conda activate crop_env

# Go to your project folder
cd /scratch/<your-username>/Indian-Farmer-Crop-Recommendation-System/agri_crop_recommendation

# Install all packages
pip install -r requirements.txt
```

### Option B: Using Python venv
```bash
module load python/3.11

python -m venv .venv
source .venv/bin/activate

cd /scratch/<your-username>/Indian-Farmer-Crop-Recommendation-System/agri_crop_recommendation
pip install -r requirements.txt
```

---

## STEP 3 — Change the .env File

### ⚠️ CHANGE REQUIRED — This is important!

Copy the example file and edit it:

```bash
cd /scratch/<your-username>/Indian-Farmer-Crop-Recommendation-System/agri_crop_recommendation
cp .env.example .env
nano .env   # or use: vi .env
```

**Change these values in `.env`:**

```dotenv
# BEFORE (your current .env on Windows)
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434
GEMINI_API_KEY=your_gemini_api_key_here

# ─────────────────────────────────────────────────────────────────
# AFTER (what to put in .env on HPC)
# ─────────────────────────────────────────────────────────────────
LLM_PROVIDER=gemini
GEMINI_API_KEY=<paste your actual Gemini API key here>

# Remove or comment out the Ollama lines — Ollama won't run on HPC nodes
# OLLAMA_MODEL=llama3.2
# OLLAMA_BASE_URL=http://localhost:11434
```

> Get your Gemini API key free at: https://aistudio.google.com/app/apikey

---

## STEP 4 — Handle Ollama (LLM)

### Why Ollama won't work on HPC:
Ollama runs as a **background server** (`ollama serve`) on your local machine.
HPC compute nodes are **batch-only** — you cannot start a persistent background
service the same way.

### ✅ Solution: Switch to Gemini API (Easiest)

You already have Gemini support built in! Just do what Step 3 says:
set `LLM_PROVIDER=gemini` and add your API key. The code in
`src/services/llm_chat.py` will automatically use Gemini — no code changes needed.

### Alternative: Run Ollama on HPC (Advanced)
If your HPC has internet access and allows it, you can try:
```bash
# Download and run Ollama (ask your HPC admin if this is allowed)
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &        # start in background
ollama pull llama3.2  # pull the model
```
Then keep `LLM_PROVIDER=ollama` in your `.env`.

---

## STEP 5 — Create a SLURM Job Script

### 📄 Create this new file: `run_job.slurm`

Save this file inside your project root:
`/scratch/<your-username>/Indian-Farmer-Crop-Recommendation-System/run_job.slurm`

```bash
#!/bin/bash
#─────────────────────────────────────────────────
# SLURM Job Configuration
#─────────────────────────────────────────────────
#SBATCH --job-name=crop_recommender       # Name of your job
#SBATCH --nodes=1                         # Use 1 node
#SBATCH --ntasks=1                        # 1 task
#SBATCH --cpus-per-task=8                 # 8 CPU cores (good for scikit-learn)
#SBATCH --gres=gpu:1                      # 1 GPU (for PyTorch — remove if no GPU needed)
#SBATCH --mem=32G                         # 32 GB RAM
#SBATCH --time=02:00:00                   # Max run time: 2 hours (HH:MM:SS)
#SBATCH --output=logs/output_%j.log       # Save stdout to logs/
#SBATCH --error=logs/error_%j.log         # Save stderr to logs/
#SBATCH --partition=gpu                   # Partition/queue name (ask your HPC admin)

#─────────────────────────────────────────────────
# Environment Setup
#─────────────────────────────────────────────────
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"

# Load required modules (exact names depend on your HPC — ask admin)
module load anaconda3/2023.09
module load cuda/12.0        # Only needed if using GPU with PyTorch

# Activate your conda environment
conda activate crop_env

#─────────────────────────────────────────────────
# Go to project directory
#─────────────────────────────────────────────────
cd /scratch/<your-username>/Indian-Farmer-Crop-Recommendation-System/agri_crop_recommendation

# Create logs directory if it doesn't exist
mkdir -p logs

#─────────────────────────────────────────────────
# Run your program
#─────────────────────────────────────────────────
python main.py

echo "Job finished at: $(date)"
```

### ▶️ Submit the job:
```bash
sbatch run_job.slurm
```

### 👀 Check job status:
```bash
squeue -u <your-username>     # See if your job is running/queued
sacct -j <job-id>             # See job details after it finishes
cat logs/output_<job-id>.log  # See your program's output
```

### ❌ Cancel a job:
```bash
scancel <job-id>
```

---

## STEP 6 — Run the Web Server (Optional)

If you want to run the **FastAPI web interface** on HPC:

### Option A: Interactive Session (for testing)
```bash
# Start an interactive session (not a batch job)
srun --nodes=1 --cpus-per-task=4 --mem=16G --time=01:00:00 --pty bash

# Then run the server
conda activate crop_env
cd /scratch/<your-username>/Indian-Farmer-Crop-Recommendation-System/agri_crop_recommendation
python run_website.py
```

Then use **SSH port forwarding** from your PC:
```powershell
# In a new PowerShell window on your Windows PC:
ssh -L 8000:localhost:8000 <your-username>@<hpc-address>

# Now open your browser at: http://localhost:8000
```

---

## 📁 FILES YOU MUST CHANGE

| File | What to Change |
|------|----------------|
| `agri_crop_recommendation/.env` | Set `LLM_PROVIDER=gemini`, add real `GEMINI_API_KEY`, comment out Ollama lines |
| `run_job.slurm` *(new file)* | Replace `<your-username>` and `<hpc-address>` with your actual details. Adjust `--partition`, `--time`, `--mem` based on your HPC |

---

## 📁 FILES YOU DO NOT NEED TO CHANGE

These files work on HPC **without any modification**:

| File | Reason |
|------|--------|
| `agri_crop_recommendation/src/services/llm_chat.py` | Already has Gemini fallback built-in |
| `agri_crop_recommendation/requirements.txt` | All packages are HPC-compatible |
| `agri_crop_recommendation/main.py` | Pure Python, works anywhere |
| `agri_crop_recommendation/run_website.py` | FastAPI works on HPC |
| All ML model files | PyTorch + XGBoost are HPC-native |

---

## 🐛 COMMON ERRORS AND FIXES

### Error: `ModuleNotFoundError: No module named 'xyz'`
```bash
# Re-install requirements
pip install -r requirements.txt

# Or install the specific missing module
pip install xyz
```

### Error: `ollama.ConnectionError` or `Ollama not available`
```
Fix: In your .env file, change LLM_PROVIDER=ollama → LLM_PROVIDER=gemini
     and add your GEMINI_API_KEY
```

### Error: `CUDA out of memory`
```
Fix: In run_job.slurm, either:
     - Reduce batch size in your model code
     - Request more GPU memory: #SBATCH --gres=gpu:2
     - Or remove GPU line and run on CPU only (slower but works)
```

### Error: `sbatch: error: invalid partition specified`
```
Fix: Ask your HPC system administrator what partition names are available.
     Common names: gpu, compute, standard, batch, normal
     Change: #SBATCH --partition=<correct-name>
```

### Error: `Permission denied` when running setup scripts
```bash
chmod +x setup.sh
./setup.sh
```

### Job stays in PENDING state too long
```
This is normal — the HPC queue may be busy.
Run: squeue -u <your-username>
Look at "REASON" column for why it's pending.
Try reducing --time or --mem to get faster allocation.
```

---

## 📞 WHO TO CONTACT

- **For module names and partition names** → Contact your HPC system administrator
- **For Gemini API key** → https://aistudio.google.com/app/apikey
- **For PARAM Shakti / PARAM Siddhi (C-DAC HPC)** → https://www.cdac.in/index.aspx?id=hpc

---

*Guide prepared for: Indian Farmer Crop Recommendation System*
*Last updated: 2026-05-27*
