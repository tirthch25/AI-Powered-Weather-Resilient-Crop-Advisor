# 🖥️ PARAM Shavak HPC — Complete Deployment Guide
## AI-Powered Weather-Resilient Crop Advisor

This guide tells you **exactly what to do** to run this project on **PARAM Shavak**,
the C-DAC compact supercomputer running BOSS Linux (Bharat Operating System Solutions).

---

## 📋 TABLE OF CONTENTS

1. [About PARAM Shavak](#about-param-shavak)
2. [Compatibility Summary](#compatibility-summary)
3. [Quick Checklist](#quick-checklist)
4. [Step 1 — Transfer Project to HPC](#step-1--transfer-project-to-hpc)
5. [Step 2 — Set Up Python Environment](#step-2--set-up-python-environment)
6. [Step 3 — Configure the .env File](#step-3--configure-the-env-file)
7. [Step 4 — Handle the LLM (Gemini vs Ollama)](#step-4--handle-the-llm-gemini-vs-ollama)
8. [Step 5 — Submit a SLURM Batch Job](#step-5--submit-a-slurm-batch-job)
9. [Step 6 — Run the Web Server](#step-6--run-the-web-server)
10. [Step 7 — States Not Loading Fix](#step-7--states-not-loading-fix)
11. [Files to Change vs Not Change](#files-to-change-vs-not-change)
12. [Common Errors and Fixes](#common-errors-and-fixes)
13. [Contact](#contact)

---

## 🖥️ ABOUT PARAM SHAVAK

| Property | Details |
|---|---|
| **System** | PARAM Shavak — C-DAC Supercomputing-in-a-Box |
| **OS** | BOSS Linux (Bharat Operating System Solutions) |
| **CPU** | Dual Intel Xeon Scalable (Gold 6240R or similar) |
| **GPU** | GPGPU/GPU accelerator cards (NVIDIA) |
| **Job Scheduler** | SLURM (Simple Linux Utility for Resource Management) |
| **Module System** | Environment Modules (`module load / avail`) |
| **Python** | Pre-installed; `conda` or `venv` for environments |
| **HPC Stack** | CHReME (C-DAC HPC Resource Management Engine) + ONAMA |

---

## ✅ COMPATIBILITY SUMMARY

| Component | Compatible? | Notes |
|---|---|---|
| FastAPI web server | ✅ Yes | Works on BOSS Linux (Python 3.10/3.11) |
| SLURM job scripts | ✅ Yes | `run_job.slurm` is already written for Param Shavak |
| Gemini API (LLM) | ✅ Yes | Recommended for HPC — no local server needed |
| PyTorch / XGBoost | ✅ Yes | GPU-accelerated on NVIDIA cards |
| Pandas / NumPy | ✅ Yes | Standard HPC scientific stack |
| Open-Meteo weather | ✅ Yes | Requires internet access on the compute node |
| Ollama (local LLM) | ⚠️ Limited | Possible in interactive sessions only |
| States dropdown | ✅ Yes | Uses Gemini API — fully works with `LLM_PROVIDER=gemini` |

---

## ✅ QUICK CHECKLIST

- [ ] Transferred project to `/scratch/<username>/` on Param Shavak
- [ ] Created conda environment `crop_env` with Python 3.11
- [ ] Installed all packages from `requirements.txt`
- [ ] Copied `.env.example` → `.env` and set `GEMINI_API_KEY`
- [ ] Set `LLM_PROVIDER=gemini` in `.env`
- [ ] Edited `run_job.slurm` with your username and correct partition
- [ ] Submitted job with `sbatch run_job.slurm`
- [ ] Verified output in `logs/` folder

---

## STEP 1 — Transfer Project to HPC

### From Windows PC (PowerShell):
```powershell
# Replace <your-username> and <hpc-address> with your actual login
scp -r "D:\AI-Powered-Weather-Resilient-Crop-Advisor" <your-username>@<hpc-address>:/scratch/<your-username>/
```

Or use **WinSCP** (GUI) for drag-and-drop transfer.

> **Where to put it:** Always use `/scratch/<your-username>/` — this is fast working
> storage. Your home directory `~/` usually has very low quota on HPC systems.

---

## STEP 2 — Set Up Python Environment

SSH into Param Shavak, then:

### Option A: Conda (Recommended)
```bash
# Step 1: Check what modules are available
module avail

# Step 2: Load Anaconda (find the exact name with: module spider anaconda)
module load anaconda3

# Step 3: Create a dedicated environment
conda create -n crop_env python=3.11 -y

# Step 4: Activate it
conda activate crop_env

# Step 5: Go to project folder
cd /scratch/<your-username>/AI-Powered-Weather-Resilient-Crop-Advisor/agri_crop_recommendation

# Step 6: Install all dependencies
pip install -r requirements.txt
```

### Option B: Python venv (If conda is unavailable)
```bash
module load python/3.11   # Check exact name with: module spider python

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

> **Tip:** After installing, run `python main.py` to confirm the setup works before
> submitting any SLURM jobs.

---

## STEP 3 — Configure the .env File

```bash
cd /scratch/<your-username>/AI-Powered-Weather-Resilient-Crop-Advisor/agri_crop_recommendation
cp .env.example .env
nano .env    # or: vi .env
```

**Set these values for Param Shavak:**

```dotenv
# ── LLM Provider — CHANGE THIS on HPC ──────────────────────────────────────
# Ollama cannot run as a persistent service on batch nodes.
# Use Gemini instead — it's already fully integrated.
LLM_PROVIDER=gemini
GEMINI_API_KEY=<paste your actual Gemini API key here>

# ── Comment out / remove Ollama settings ───────────────────────────────────
# OLLAMA_MODEL=llama3.2
# OLLAMA_BASE_URL=http://localhost:11434

# ── Weather API (no key needed — Open-Meteo is free) ───────────────────────
# No changes needed for weather.

# ── Web server port (optional) ──────────────────────────────────────────────
PORT=8000
HOST=0.0.0.0
```

> Get your free Gemini API key at: **https://aistudio.google.com/app/apikey**

---

## STEP 4 — Handle the LLM (Gemini vs Ollama)

### Why Ollama does NOT work on HPC batch nodes
Ollama runs as a background server (`ollama serve`) which requires a persistent
process. HPC batch nodes are ephemeral — they don't support persistent background
services in the same way.

### ✅ Solution: Use Gemini API (Already Built In!)
The project already has full Gemini support. Just set `LLM_PROVIDER=gemini` in `.env`.

All features that use LLM will automatically work:
- ✅ States and Districts dropdowns (via `/api/states/{code}`)
- ✅ Crop recommendations (AI-generated)
- ✅ Chat widget (Gemini-powered)
- ✅ Soil analysis
- ✅ Climate signals

### Advanced: Ollama in Interactive Session Only
```bash
# Start an interactive compute session first
srun --nodes=1 --cpus-per-task=4 --mem=16G --time=01:00:00 --pty bash

# Then manually start Ollama
ollama serve &
sleep 10
ollama pull llama3.2

# Now run the app — Ollama will work in this session
conda activate crop_env
cd /scratch/<your-username>/AI-Powered-Weather-Resilient-Crop-Advisor/agri_crop_recommendation
python run_website.py
```

---

## STEP 5 — Submit a SLURM Batch Job

The file `run_job.slurm` is already included in the project root.
Edit it first:

```bash
nano /scratch/<your-username>/AI-Powered-Weather-Resilient-Crop-Advisor/run_job.slurm
```

**Change these two lines:**
```bash
HPC_USER="<YOUR_USERNAME>"          # ← Replace with your actual username
#SBATCH --partition=gpu             # ← Run: sinfo  to find the correct partition name
```

**Submit the job:**
```bash
sbatch run_job.slurm
```

**Check job status:**
```bash
squeue -u <your-username>           # See running/queued jobs
sacct -j <job-id>                   # See job details
cat logs/output_<job-id>.log        # View program output
```

**Cancel a job:**
```bash
scancel <job-id>
```

---

## STEP 6 — Run the Web Server

The full web interface (FastAPI + UI) can be served from Param Shavak.

### Option A: Interactive Session (for testing)
```bash
srun --nodes=1 --cpus-per-task=4 --mem=16G --time=02:00:00 --pty bash

conda activate crop_env
cd /scratch/<your-username>/AI-Powered-Weather-Resilient-Crop-Advisor/agri_crop_recommendation
python run_website.py
```

### Option B: As a SLURM batch job
Modify `run_job.slurm` — replace `python main.py` with:
```bash
python run_website.py
```

### Access the UI from your Windows PC:
```powershell
# Open a new PowerShell window and run SSH port-forward:
ssh -L 8000:localhost:8000 <your-username>@<hpc-address>

# Then open your browser at:
# http://localhost:8000
```

---

## STEP 7 — States Not Loading Fix

If the State dropdown shows "⚠ No states data" on Param Shavak, it means the
Gemini API call is failing. Here is the fix:

### Check 1: Verify your GEMINI_API_KEY
```bash
cat /scratch/<your-username>/AI-Powered-Weather-Resilient-Crop-Advisor/agri_crop_recommendation/.env
# Confirm GEMINI_API_KEY is set and not empty
```

### Check 2: Test internet access from the compute node
```bash
curl -I https://generativelanguage.googleapis.com
# Should return: HTTP/2 200  (or a redirect)
# If it hangs or fails → compute node has no internet → contact HPC admin
```

### Check 3: Test the API endpoint manually
```bash
conda activate crop_env
cd /scratch/<your-username>/AI-Powered-Weather-Resilient-Crop-Advisor/agri_crop_recommendation
python -c "
from src.agents.location_agent import get_states
states = get_states('IN')
print(f'States loaded: {len(states)}')
print(states[:3])
"
```

### Fix: If internet is blocked (common on HPC compute nodes)
Some HPC nodes only allow outbound internet from the **login node**, not compute nodes.
In that case, run the web server from a login node interactive session:

```bash
# On the LOGIN NODE (not a batch job):
conda activate crop_env
cd /scratch/<your-username>/AI-Powered-Weather-Resilient-Crop-Advisor/agri_crop_recommendation
python run_website.py &
# Then SSH port-forward from your PC as shown in Step 6
```

---

## 📁 FILES TO CHANGE vs NOT CHANGE

### ✏️ Must Change
| File | What to Change |
|---|---|
| `agri_crop_recommendation/.env` | Set `LLM_PROVIDER=gemini`, add `GEMINI_API_KEY`, comment out Ollama |
| `run_job.slurm` | Replace `<YOUR_USERNAME>`, fix `--partition` name using `sinfo` |

### 🔒 Do NOT Change (Already HPC-Compatible)
| File | Reason |
|---|---|
| `src/services/llm_chat.py` | Already has Gemini fallback built-in |
| `src/agents/location_agent.py` | Fully LLM-based, works with Gemini |
| `src/agents/llm_location_agent.py` | Works with Gemini API |
| `requirements.txt` | All packages available via pip on BOSS Linux |
| `main.py` | Pure Python, works anywhere |
| `run_website.py` | FastAPI/Uvicorn works on BOSS Linux |
| All ML model files | PyTorch + XGBoost are standard HPC packages |
| `templates/index.html` | Static HTML, no changes needed |
| `static/js/app.js` | Client-side JS, no changes needed |

---

## 🐛 COMMON ERRORS AND FIXES

### `ModuleNotFoundError: No module named 'xyz'`
```bash
pip install -r requirements.txt   # Re-run full install
pip install xyz                   # Or install the specific package
```

### `ollama.ConnectionError` / `Ollama not available`
```
Fix: Set LLM_PROVIDER=gemini in your .env file and add your GEMINI_API_KEY.
     Ollama is NOT needed — Gemini works for all features.
```

### `CUDA out of memory`
```bash
# In run_job.slurm, either:
# Option 1: Request more GPU memory
#SBATCH --gres=gpu:2

# Option 2: Run on CPU only (remove GPU line)
# Comment out: #SBATCH --gres=gpu:1
```

### `sbatch: error: invalid partition specified`
```bash
# Check what partitions exist on your Param Shavak:
sinfo

# Common names on C-DAC systems: gpu, compute, standard, batch, normal
# Then change in run_job.slurm:
# #SBATCH --partition=<correct-name>
```

### States dropdown shows "⚠ No states data"
```
Fix: See Step 7 above.
     Most likely cause: GEMINI_API_KEY not set, or no internet on compute node.
```

### `Permission denied` when running setup scripts
```bash
chmod +x setup.sh
./setup.sh
```

### Job stays in PENDING state too long
```
This is normal — the HPC queue may be busy.
Run: squeue -u <your-username>
Look at the "REASON" column.
Try reducing --time or --mem for faster queue allocation.
```

### `Connection refused` when accessing web UI
```
Make sure SSH port-forwarding is active:
  ssh -L 8000:localhost:8000 <your-username>@<hpc-address>
Then go to: http://localhost:8000
```

---

## 📞 CONTACT / RESOURCES

| Resource | Link |
|---|---|
| C-DAC HPC / PARAM Shavak | https://www.cdac.in/index.aspx?id=hpc |
| Free Gemini API Key | https://aistudio.google.com/app/apikey |
| BOSS Linux | https://bosslinux.in |
| SLURM documentation | https://slurm.schedmd.com/documentation.html |

---

*Guide prepared for: AI-Powered Weather-Resilient Crop Advisor*  
*Compatible with: PARAM Shavak (C-DAC) · BOSS Linux · SLURM*  
*Last updated: 2026-06-18*
