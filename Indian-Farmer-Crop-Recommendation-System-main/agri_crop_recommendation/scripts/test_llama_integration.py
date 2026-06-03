"""
Test LLaMA Integration
======================
Run this script to verify that the Ollama / LLaMA replacement is working
correctly across all three LLM service files.

Usage:
    python scripts/test_llama_integration.py

Prerequisites:
    1. Ollama installed and running   (ollama serve)
    2. A model pulled                 (ollama pull llama3.2)
    3. pip install ollama
"""

import sys
import os
import json
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
INFO = "\033[94m  INFO\033[0m"
WARN = "\033[93m  WARN\033[0m"

results = []


def check(label, passed, detail=""):
    icon = PASS if passed else FAIL
    print(f"{icon}  {label}")
    if detail:
        print(f"       {detail}")
    results.append((label, passed))


# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  LLaMA / Ollama Integration Test")
print("=" * 60)

# ── 1. Ollama package installed ───────────────────────────────────────────────
print(f"\n{INFO}  [1/5] Checking ollama package...")
try:
    import ollama
    try:
        from importlib.metadata import version as pkg_version
        ver = pkg_version("ollama")
    except Exception:
        ver = "unknown"
    check("ollama package importable", True, f"version: {ver}")
except ImportError as e:
    check("ollama package importable", False, str(e))
    print("       Run: pip install ollama")

# ── 2. Ollama server reachable ────────────────────────────────────────────────
print(f"\n{INFO}  [2/5] Pinging Ollama server...")
ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")
try:
    client = ollama.Client(host=ollama_url)
    models_response = client.list()
    # Handle both old (list of dicts) and new (object with .models attr) response
    if hasattr(models_response, "models"):
        model_names = [m.model for m in models_response.models]
    else:
        model_names = [m.get("name", "") for m in models_response]
    check("Ollama server reachable", True, f"URL: {ollama_url}")
    print(f"       Available models: {model_names or ['(none pulled yet)']}")
except Exception as e:
    check("Ollama server reachable", False, str(e))
    print("       Start Ollama with:  ollama serve")
    model_names = []

# ── 3. Target model available ─────────────────────────────────────────────────
print(f"\n{INFO}  [3/5] Checking model '{ollama_model}'...")
model_available = any(ollama_model in n for n in model_names)
check(f"Model '{ollama_model}' pulled", model_available,
      f"Pull with: ollama pull {ollama_model}" if not model_available else "Ready to use")

# ── 4. Quick LLM chat test ────────────────────────────────────────────────────
print(f"\n{INFO}  [4/5] Testing llm_chat.py  (answer_farmer_question)...")
try:
    from src.services.llm_chat import answer_farmer_question
    t0 = time.time()
    answer, history = answer_farmer_question(
        question="What is a good crop for sandy soil in summer?",
        region_name="Jaipur",
        state_name="Rajasthan",
        season="Kharif",
        soil_info="Sandy, pH 7.5",
    )
    elapsed = time.time() - t0
    passed = bool(answer) and "unavailable" not in answer.lower()
    check("llm_chat answer_farmer_question", passed,
          f"Response in {elapsed:.1f}s | {len(answer)} chars | history turns: {len(history)}")
    if passed:
        print(f"\n       Preview: {answer[:200]}...\n")
except Exception as e:
    check("llm_chat answer_farmer_question", False, str(e))

# ── 5. LLM explainer test ─────────────────────────────────────────────────────
print(f"\n{INFO}  [5/5] Testing llm_explainer.py  (generate_crop_explanation)...")
try:
    from src.services.llm_explainer import generate_crop_explanation
    t0 = time.time()
    result = generate_crop_explanation(
        crop_name="Bajra",
        region_name="Visakhapatnam",
        region_id="AP_VISAKHAPATNAM",
        season="Kharif",
        suitability_score=90.34,
        avg_temp=29.87,
        expected_rainfall=1150.0,
        soil_texture="Clay",
        soil_ph=7.3,
        risk_note="Low risk",
    )
    elapsed = time.time() - t0
    passed = isinstance(result, dict) and "english" in result
    check("llm_explainer generate_crop_explanation", passed,
          f"Response in {elapsed:.1f}s | keys: {list(result.keys())}")
    if passed:
        print(f"\n       English: {result.get('english', '')}")
        print(f"       Why good: {result.get('why_good', '')}")
        print(f"       Watch out: {result.get('watch_out', '')}\n")
except Exception as e:
    check("llm_explainer generate_crop_explanation", False, str(e))

# ── Summary ───────────────────────────────────────────────────────────────────
print("=" * 60)
total  = len(results)
passed = sum(1 for _, ok in results if ok)
print(f"  Results: {passed}/{total} passed")
if passed == total:
    print("\033[92m  All tests passed! LLaMA integration is working.\033[0m")
else:
    failed = [label for label, ok in results if not ok]
    print(f"\033[91m  Failed: {failed}\033[0m")
print("=" * 60 + "\n")

sys.exit(0 if passed == total else 1)
