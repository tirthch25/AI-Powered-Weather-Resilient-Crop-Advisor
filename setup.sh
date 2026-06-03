#!/usr/bin/env bash
# setup.sh — One-click setup for Indian Farmer Crop Recommendation System
# Usage: bash setup.sh        (interactive)
#        bash setup.sh --auto (non-interactive, all defaults)
#
# LLM: LLaMA 3.2 via Ollama (local, free) — Gemini used as automatic fallback

set -e

echo ""
echo "================================================================"
echo "  Indian Farmer Crop Recommendation System — Setup"
echo "  LLM: LLaMA 3.2 (local via Ollama) + Gemini fallback"
echo "================================================================"
echo ""

# Check Python
if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
    echo " [ERROR] Python is not installed."
    echo ""
    echo " Install it via:"
    echo "   Ubuntu/Debian: sudo apt install python3 python3-pip python3-venv"
    echo "   macOS:         brew install python"
    echo "   Or download from: https://www.python.org/downloads/"
    exit 1
fi

# Prefer python3
PYTHON=$(command -v python3 || command -v python)
echo " Python found: $($PYTHON --version)"
echo ""

# Check for Ollama (optional pre-check — setup_project.py handles it fully)
if command -v ollama &>/dev/null; then
    echo " [i] Ollama found: $(ollama --version 2>/dev/null || echo 'installed')"
    echo " [i] The setup script will check for the llama3.2 model."
else
    echo " [i] Ollama not found — the setup script will guide you through installation."
    echo " [i] Download from: https://ollama.com/download"
    echo " [i] macOS/Linux quick install:  curl -fsSL https://ollama.com/install.sh | sh"
fi
echo ""

# Run the setup script
$PYTHON setup_project.py "$@"
