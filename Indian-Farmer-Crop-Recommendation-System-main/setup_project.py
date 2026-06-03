"""
setup_project.py — One-command project setup for Indian Farmer Crop Recommendation System.

Usage:
    python setup_project.py            # Interactive (recommended)
    python setup_project.py --auto     # Non-interactive, all defaults

This script:
  1. Checks Python version (3.8+)
  2. Creates a virtual environment inside agri_crop_recommendation/
  3. Upgrades pip and installs all dependencies from requirements.txt
  4. Configures .env (LLM_PROVIDER=ollama, optional GEMINI_API_KEY fallback)
  5. Checks for Ollama and optionally pulls llama3.2 (primary local LLM)
  6. Verifies that data files are present
  7. Optionally runs the LLM district enrichment pipeline
  8. Optionally trains ML models (Random Forest, XGBoost, LSTM)
  9. Starts the web server
"""

import sys
import io
# Force UTF-8 output on all platforms (fixes Windows cp1252 errors)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
import os
import subprocess
import platform
import shutil
import textwrap

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR   = os.path.dirname(os.path.abspath(__file__))
APP_DIR    = os.path.join(ROOT_DIR, "agri_crop_recommendation")
VENV_DIR   = os.path.join(APP_DIR, ".venv")
REQ_FILE   = os.path.join(APP_DIR, "requirements.txt")
ENV_FILE   = os.path.join(APP_DIR, ".env")
ENV_EXAMPLE= os.path.join(APP_DIR, ".env.example")
RUN_FILE   = os.path.join(APP_DIR, "run_website.py")
REGIONAL_CROPS = os.path.join(APP_DIR, "data", "reference", "regional_crops.json")

IS_WINDOWS = platform.system() == "Windows"

# ── Helpers ───────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def _enable_ansi():
    """Enable ANSI colour codes on Windows 10+."""
    if IS_WINDOWS:
        try:
            import ctypes
            kernel = ctypes.windll.kernel32
            kernel.SetConsoleMode(kernel.GetStdHandle(-11), 7)
        except Exception:
            pass

def ok(msg):    print(f"{GREEN}  [OK]  {msg}{RESET}")
def info(msg):  print(f"{CYAN}  [i]   {msg}{RESET}")
def warn(msg):  print(f"{YELLOW}  [!]   {msg}{RESET}")
def err(msg):   print(f"{RED}  [X]   {msg}{RESET}")
def step(msg):  print(f"\n{BOLD}{CYAN}>> {msg}{RESET}")
def hr():       print(f"{CYAN}{'-' * 64}{RESET}")

def ask(prompt, default="y"):
    """Ask a yes/no question; return True for yes."""
    choices = "[Y/n]" if default == "y" else "[y/N]"
    try:
        answer = input(f"{YELLOW}  ? {prompt} {choices}: {RESET}").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return default == "y"
    if answer == "":
        return default == "y"
    return answer in ("y", "yes")

def run(cmd, cwd=None, capture=False):
    """Run a command, optionally capturing output. Raises on failure."""
    kwargs = dict(cwd=cwd or APP_DIR)
    if capture:
        kwargs["capture_output"] = True
        kwargs["text"] = True
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(str(c) for c in cmd)}")
    return result

def python_in_venv():
    """Return the path to the Python executable inside .venv."""
    if IS_WINDOWS:
        return os.path.join(VENV_DIR, "Scripts", "python.exe")
    return os.path.join(VENV_DIR, "bin", "python")

def pip_in_venv():
    if IS_WINDOWS:
        return os.path.join(VENV_DIR, "Scripts", "pip.exe")
    return os.path.join(VENV_DIR, "bin", "pip")

# ── Steps ─────────────────────────────────────────────────────────────────────

def check_python():
    step("Checking Python version")
    major, minor = sys.version_info[:2]
    info(f"Detected Python {major}.{minor} ({sys.executable})")
    if (major, minor) < (3, 8):
        err(f"Python 3.8 or newer is required. You have {major}.{minor}.")
        err("Download from https://www.python.org/downloads/")
        sys.exit(1)
    ok(f"Python {major}.{minor} — OK")


def create_venv():
    step("Creating virtual environment")
    if os.path.isdir(VENV_DIR) and os.path.isfile(python_in_venv()):
        ok(f"Virtual environment already exists at {VENV_DIR}")
        return
    info(f"Creating .venv in {APP_DIR} ...")
    run([sys.executable, "-m", "venv", VENV_DIR], cwd=ROOT_DIR)
    ok("Virtual environment created")


def install_deps():
    step("Installing dependencies")
    info("Upgrading pip ...")
    run([python_in_venv(), "-m", "pip", "install", "--upgrade", "pip", "--quiet"])
    info(f"Installing packages from requirements.txt ...")
    run([pip_in_venv(), "install", "-r", REQ_FILE, "--quiet"])
    ok("All dependencies installed (includes ollama + google-genai)")


def setup_env():
    """
    Create .env with Ollama as primary LLM provider.
    Gemini API key is optional — used only as automatic fallback.
    """
    step("Configuring environment (.env)")

    # Read existing .env if present
    existing_content = ""
    if os.path.isfile(ENV_FILE):
        with open(ENV_FILE) as f:
            existing_content = f.read()

    # Check if already configured properly
    if (
        os.path.isfile(ENV_FILE)
        and "LLM_PROVIDER=ollama" in existing_content
    ):
        ok(f".env already configured with LLM_PROVIDER=ollama")
        # Still show Gemini key status
        if "GEMINI_API_KEY=" in existing_content:
            key_val = [l for l in existing_content.splitlines() if l.startswith("GEMINI_API_KEY=")]
            key_val = key_val[0].split("=", 1)[-1].strip() if key_val else ""
            if key_val and "your_" not in key_val and len(key_val) > 10:
                ok("GEMINI_API_KEY (Gemini fallback) is configured")
            else:
                info("GEMINI_API_KEY not set — Gemini fallback will be skipped (Ollama is primary)")
        return

    # Create base .env from example or from scratch
    if os.path.isfile(ENV_EXAMPLE):
        shutil.copy(ENV_EXAMPLE, ENV_FILE)
    else:
        with open(ENV_FILE, "w", encoding="utf-8") as f:
            f.write(
                "# Indian Farmer Crop Recommendation System — environment config\n\n"
                "# ── Primary LLM (Local, Free, Private) ──────────────────────\n"
                "LLM_PROVIDER=ollama\n"
                "OLLAMA_MODEL=llama3.2\n"
                "OLLAMA_BASE_URL=http://localhost:11434\n\n"
                "# ── Optional Gemini Fallback ──────────────────────────────────\n"
                "# Used automatically if Ollama is not running.\n"
                "GEMINI_API_KEY=\n"
            )

    # Update LLM_PROVIDER in .env to ollama (in case example had gemini)
    with open(ENV_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.splitlines()
    new_lines = []
    has_provider = has_model = has_url = False
    for line in lines:
        if line.startswith("LLM_PROVIDER="):
            new_lines.append("LLM_PROVIDER=ollama")
            has_provider = True
        elif line.startswith("OLLAMA_MODEL="):
            new_lines.append("OLLAMA_MODEL=llama3.2")
            has_model = True
        elif line.startswith("OLLAMA_BASE_URL="):
            new_lines.append("OLLAMA_BASE_URL=http://localhost:11434")
            has_url = True
        else:
            new_lines.append(line)

    # Append missing keys
    if not has_provider:
        new_lines.append("LLM_PROVIDER=ollama")
    if not has_model:
        new_lines.append("OLLAMA_MODEL=llama3.2")
    if not has_url:
        new_lines.append("OLLAMA_BASE_URL=http://localhost:11434")

    with open(ENV_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + "\n")

    ok("LLM_PROVIDER=ollama configured as primary")

    # Optional: Gemini API key for fallback
    print()
    print(f"  {BOLD}GEMINI_API_KEY{RESET} — optional Gemini fallback (used if Ollama is down):")
    print(f"  Get a free key at: {CYAN}https://aistudio.google.com/app/apikey{RESET}")
    print()
    try:
        key = input(f"{YELLOW}  ? Paste Gemini API key for fallback (or press Enter to skip): {RESET}").strip()
    except (EOFError, KeyboardInterrupt):
        key = ""

    if key:
        with open(ENV_FILE, "r", encoding="utf-8") as f:
            content = f.read()
        if "GEMINI_API_KEY=" in content:
            content = "\n".join(
                f"GEMINI_API_KEY={key}" if l.startswith("GEMINI_API_KEY=") else l
                for l in content.splitlines()
            ) + "\n"
        else:
            content += f"GEMINI_API_KEY={key}\n"
        with open(ENV_FILE, "w", encoding="utf-8") as f:
            f.write(content)
        ok("GEMINI_API_KEY saved as fallback")
    else:
        info("No Gemini key — Ollama (LLaMA) will be the only LLM provider")


def setup_ollama(auto=False):
    """
    Check if Ollama is installed and the llama3.2 model is available.
    Offer to guide the user through setup if not found.
    """
    step("Ollama / LLaMA 3.2 Setup (Primary LLM)")

    # Read configured model from .env
    ollama_model = "llama3.2"
    if os.path.isfile(ENV_FILE):
        with open(ENV_FILE) as f:
            for line in f:
                if line.startswith("OLLAMA_MODEL="):
                    ollama_model = line.split("=", 1)[-1].strip() or ollama_model

    # Check if ollama binary is available
    ollama_bin = shutil.which("ollama")
    if not ollama_bin:
        # Check common install paths on Windows
        candidate = os.path.join(
            os.environ.get("LOCALAPPDATA", ""),
            "Programs", "Ollama", "ollama.exe"
        )
        if os.path.isfile(candidate):
            ollama_bin = candidate

    if not ollama_bin:
        warn("Ollama not found in PATH")
        print()
        print(f"  {BOLD}Ollama{RESET} runs LLaMA 3.2 locally — free, private, no API key needed.")
        print(f"  Download from: {CYAN}https://ollama.com/download{RESET}")
        print()
        if IS_WINDOWS:
            setup_ps1 = os.path.join(APP_DIR, "scripts", "setup_ollama.ps1")
            if os.path.isfile(setup_ps1):
                print(f"  Or run the automated setup script:")
                print(f"    {CYAN}powershell -ExecutionPolicy Bypass -File scripts\\setup_ollama.ps1{RESET}")
                print()
        print("  The app will still run in ML-only mode without Ollama.")
        print("  You can set up Ollama later and restart the server.")
        info("Skipping Ollama setup — install it manually and re-run setup")
        return

    ok(f"Ollama found: {ollama_bin}")

    # Check if server is responding
    server_up = False
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434", timeout=3)
        server_up = True
        ok("Ollama server is running (http://localhost:11434)")
    except Exception:
        warn("Ollama server is not running — start it with: ollama serve")

    # Check if model is already pulled
    try:
        result = subprocess.run(
            [ollama_bin, "list"],
            capture_output=True, text=True, timeout=10
        )
        if ollama_model in result.stdout:
            ok(f"Model '{ollama_model}' is already available")
            return
        else:
            warn(f"Model '{ollama_model}' not found locally")
    except Exception:
        warn("Could not check model list (Ollama server may not be running)")

    # Offer to pull
    print()
    model_sizes = {
        "llama3.2": "~2 GB", "llama3.1": "~4.7 GB",
        "gemma3:2b": "~1.6 GB", "gemma3": "~3.3 GB"
    }
    size_hint = model_sizes.get(ollama_model, "~2 GB")
    print(f"  Pull '{ollama_model}' model now? ({size_hint} download)")
    print()

    if auto or ask(f"Pull {ollama_model} model now?", default="y"):
        info(f"Pulling {ollama_model} — this may take a few minutes ...")
        try:
            subprocess.run([ollama_bin, "pull", ollama_model], cwd=APP_DIR)
            ok(f"Model '{ollama_model}' pulled successfully")
        except Exception as e:
            warn(f"Could not pull model: {e}")
            info(f"Pull it manually later with: ollama pull {ollama_model}")
    else:
        info(f"Skipping. Pull later with: ollama pull {ollama_model}")


def check_data():
    step("Checking data files")
    regions_json = os.path.join(APP_DIR, "data", "reference", "regions.json")
    crop_knowledge = os.path.join(APP_DIR, "data", "reference", "crop_knowledge.json")

    missing = []
    for path in [regions_json, crop_knowledge]:
        if os.path.isfile(path):
            ok(os.path.basename(path))
        else:
            err(f"Missing: {path}")
            missing.append(path)

    if missing:
        err("Required data files are missing. Make sure you cloned the full repository.")
        sys.exit(1)

    # Check enrichment file
    if os.path.isfile(REGIONAL_CROPS):
        size = os.path.getsize(REGIONAL_CROPS)
        if size > 10_000:
            ok(f"regional_crops.json ({size // 1024} KB — enrichment data present)")
        else:
            warn("regional_crops.json exists but appears empty/incomplete")
    else:
        warn("regional_crops.json not found — system will use zone-based fallback scoring")


def run_enrichment(auto=False):
    step("Regional Crop Enrichment (LLaMA / Gemini)")

    enrichment_script = os.path.join(APP_DIR, "scripts", "enrich_regional_crops.py")
    if not os.path.isfile(enrichment_script):
        warn("Enrichment script not found — skipping")
        return

    # Check if any LLM is configured (Ollama running OR Gemini key set)
    ollama_bin = shutil.which("ollama") or os.path.join(
        os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe"
    )
    ollama_ok = os.path.isfile(ollama_bin) if not shutil.which("ollama") else bool(shutil.which("ollama"))
    gemini_key_set = False
    if os.path.isfile(ENV_FILE):
        with open(ENV_FILE) as f:
            for line in f:
                if line.startswith("GEMINI_API_KEY="):
                    val = line.split("=", 1)[-1].strip()
                    if val and "your_" not in val and len(val) > 10:
                        gemini_key_set = True

    if not ollama_ok and not gemini_key_set:
        warn("No LLM configured (Ollama not found + no Gemini key) — skipping enrichment")
        info("System will use zone-based fallback scoring (still works well)")
        return

    if not auto:
        print()
        print("  This step queries the LLM once per district to build a district→crop")
        print("  approval database. It covers 640 districts. Safe to stop and resume.")
        print(f"  Primary: LLaMA 3.2 via Ollama  |  Fallback: Gemini")
        print()
        if not ask("Run LLM district enrichment now?", default="n"):
            info("Skipping enrichment. Run later with:")
            info("  python scripts/enrich_regional_crops.py --only-missing")
            return

    info("Running enrichment (Ctrl+C to stop safely — progress is saved) ...")
    try:
        run([python_in_venv(), enrichment_script, "--only-missing"])
        ok("Enrichment completed")
    except KeyboardInterrupt:
        warn("Enrichment interrupted — progress saved. Resume anytime with --only-missing")
    except RuntimeError:
        warn("Enrichment encountered an error — check Ollama is running or your GEMINI_API_KEY")


def train_models(auto=False):
    step("ML Model Training")
    models_dir = os.path.join(APP_DIR, "models")

    rf_model   = os.path.join(models_dir, "crop_suitability", "rf_model.joblib")
    lstm_model = os.path.join(models_dir, "weather_lstm", "lstm_weights.pt")
    xgb_model  = os.path.join(models_dir, "weather_xgboost", "temp_max_model.joblib")

    if os.path.isfile(rf_model) and os.path.isfile(lstm_model) and os.path.isfile(xgb_model):
        ok("All ML models already trained and present")
        return

    missing_models = []
    if not os.path.isfile(rf_model):   missing_models.append("Random Forest (crop suitability)")
    if not os.path.isfile(lstm_model): missing_models.append("LSTM (weather forecast)")
    if not os.path.isfile(xgb_model):  missing_models.append("XGBoost (weather forecast)")

    warn(f"Missing models: {', '.join(missing_models)}")
    info("Without models the system uses rule-based scoring (still works, slightly less accurate)")
    print()

    if not auto:
        print("  Model training options:")
        print("    rf     — Random Forest only (fast, ~2 min, no district data needed)")
        print("    all    — RF + XGBoost + LSTM  (1–3 hrs, needs district weather data)")
        print()
        if not ask("Train the Random Forest crop suitability model now?", default="y"):
            info("Skipping. Train later with:")
            info("  python scripts/train_model.py --model rf")
            return

    train_script = os.path.join(APP_DIR, "scripts", "train_model.py")
    if not os.path.isfile(train_script):
        warn("train_model.py not found — skipping")
        return

    info("Training Random Forest model (this takes ~2 minutes) ...")
    try:
        run([python_in_venv(), train_script, "--model", "rf"])
        ok("Random Forest model trained and saved")
    except RuntimeError:
        warn("Model training failed — system will use rule-based scoring fallback")


def launch_server(auto=False):
    step("Starting Web Server")

    if not auto:
        print()
        if not ask("Start the web server now?", default="y"):
            print()
            info("To start the server later, run:")
            print(f"    {CYAN}cd agri_crop_recommendation{RESET}")
            print(f"    {CYAN}python run_website.py{RESET}")
            return

    print()
    ok("Server starting — press Ctrl+C to stop")
    info("Web Interface  →  http://localhost:8000")
    info("API Docs       →  http://localhost:8000/docs")
    info("Health Check   →  http://localhost:8000/health")
    print()
    try:
        subprocess.run([python_in_venv(), RUN_FILE], cwd=APP_DIR)
    except KeyboardInterrupt:
        print()
        ok("Server stopped")


# ── Banner ────────────────────────────────────────────────────────────────────

def print_banner():
    _enable_ansi()
    print()
    hr()
    print(f"{BOLD}{GREEN}")
    print("   INDIAN FARMER CROP RECOMMENDATION SYSTEM")
    print(f"   Automated Project Setup{RESET}")
    hr()
    print()
    print(textwrap.dedent(f"""\
  {CYAN}This script will:{RESET}
    1. Check your Python version (3.8+ required)
    2. Create a virtual environment
    3. Install all Python dependencies
    4. Configure .env  (LLM_PROVIDER=ollama as primary, Gemini as optional fallback)
    5. Check Ollama + pull LLaMA 3.2 model (local, free AI — ~2 GB download)
    6. Verify data files
    7. (Optional) Run LLM district crop enrichment
    8. (Optional) Train ML models
    9. (Optional) Start the web server

  {YELLOW}You can press Ctrl+C at any time to stop.{RESET}
    """))
    hr()


def print_summary():
    print()
    hr()
    print(f"{BOLD}{GREEN}  ✔  Setup complete!{RESET}")
    hr()
    print()
    print(f"  {CYAN}To start the server later:{RESET}")
    print(f"    cd agri_crop_recommendation")
    if IS_WINDOWS:
        print(f"    .venv\\Scripts\\activate")
    else:
        print(f"    source .venv/bin/activate")
    print(f"    python run_website.py")
    print()
    print(f"  {CYAN}LLM status:{RESET}")
    print(f"    Primary  : LLaMA 3.2 via Ollama  (ollama serve)")
    print(f"    Fallback : Gemini  (set GEMINI_API_KEY in .env)")
    print(f"    Health   : http://localhost:8000/health")
    print()
    print(f"  {CYAN}Useful commands:{RESET}")
    print(f"    ollama serve                                             # start local LLM server")
    print(f"    ollama pull llama3.2                                     # re-pull LLaMA model")
    print(f"    python scripts/enrich_regional_crops.py --only-missing   # district enrichment")
    print(f"    python scripts/train_model.py --model all                # train all ML models")
    print(f"    python scripts/fetch_district_weather.py                 # download weather data")
    print(f"    python scripts/test_llama_integration.py                 # verify LLM integration")
    print()
    hr()
    print()


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    auto = "--auto" in sys.argv

    print_banner()

    if not auto:
        try:
            input(f"  {YELLOW}Press Enter to begin setup (or Ctrl+C to exit)...{RESET} ")
        except (EOFError, KeyboardInterrupt):
            print("\n  Cancelled.")
            sys.exit(0)

    try:
        check_python()
        create_venv()
        install_deps()
        setup_env()
        setup_ollama(auto=auto)
        check_data()
        run_enrichment(auto=auto)
        train_models(auto=auto)
        print_summary()
        launch_server(auto=auto)
    except KeyboardInterrupt:
        print()
        warn("Setup interrupted by user.")
        sys.exit(0)
    except SystemExit:
        raise
    except Exception as e:
        print()
        err(f"Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
