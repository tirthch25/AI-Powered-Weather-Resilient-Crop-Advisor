# setup_ollama.ps1
# ─────────────────────────────────────────────────────────────────────────────
# One-shot Ollama setup for the Indian Farmer Crop Recommendation System
# Run from PowerShell as Administrator:
#   .\scripts\setup_ollama.ps1
# ─────────────────────────────────────────────────────────────────────────────

param(
    [string]$Model = "llama3.2"    # Change to "gemma3:2b" for lighter option
)

$ErrorActionPreference = "Stop"

function Write-Step($msg) {
    Write-Host "`n>>> $msg" -ForegroundColor Cyan
}

function Write-Ok($msg) {
    Write-Host "  OK  $msg" -ForegroundColor Green
}

function Write-Warn($msg) {
    Write-Host "  WARN  $msg" -ForegroundColor Yellow
}

Write-Host "`n=============================================" -ForegroundColor Magenta
Write-Host "  Ollama Setup for Crop Recommendation System" -ForegroundColor Magenta
Write-Host "=============================================`n" -ForegroundColor Magenta

# ── Step 1: Check if Ollama already installed ─────────────────────────────────
Write-Step "Checking if Ollama is installed..."
$ollamaPath = Get-Command ollama -ErrorAction SilentlyContinue
if ($ollamaPath) {
    Write-Ok "Ollama already installed: $($ollamaPath.Source)"
} else {
    Write-Step "Ollama not found. Downloading installer (~100 MB)..."
    $installer = "$env:TEMP\OllamaSetup.exe"
    Invoke-WebRequest -Uri "https://ollama.com/download/OllamaSetup.exe" `
        -OutFile $installer -UseBasicParsing
    Write-Ok "Downloaded: $installer"

    Write-Step "Running Ollama installer (silent)..."
    Start-Process -FilePath $installer -ArgumentList "/S" -Wait
    Write-Ok "Ollama installed"

    # Refresh PATH
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("PATH", "User")
}

# ── Step 2: Start Ollama server ───────────────────────────────────────────────
Write-Step "Starting Ollama server..."
$running = $false
try {
    $response = Invoke-WebRequest -Uri "http://localhost:11434" -TimeoutSec 2 -UseBasicParsing -ErrorAction SilentlyContinue
    if ($response.StatusCode -eq 200) { $running = $true }
} catch {}

if ($running) {
    Write-Ok "Ollama server already running at http://localhost:11434"
} else {
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 3
    Write-Ok "Ollama server started in background"
}

# ── Step 3: Pull the model ────────────────────────────────────────────────────
Write-Step "Pulling model '$Model' (this may take a few minutes on first run)..."
Write-Warn "Download size: llama3.2=2 GB | gemma3:2b=1.6 GB | llama3.1:8b=4.7 GB"
ollama pull $Model
Write-Ok "Model '$Model' ready"

# ── Step 4: Quick smoke test ──────────────────────────────────────────────────
Write-Step "Running quick smoke test..."
$testResult = echo "Say 'ready' in one word only" | ollama run $Model 2>&1
Write-Ok "Model responded: $testResult"

# ── Step 5: Update .env ───────────────────────────────────────────────────────
Write-Step "Updating .env with selected model..."
$envPath = Join-Path $PSScriptRoot "..\. env" | Resolve-Path -ErrorAction SilentlyContinue
if (-not $envPath) {
    $envPath = Join-Path $PSScriptRoot "..\.env"
}
if (Test-Path $envPath) {
    $content = Get-Content $envPath -Raw
    $content = $content -replace "OLLAMA_MODEL=.*", "OLLAMA_MODEL=$Model"
    Set-Content $envPath $content
    Write-Ok ".env updated with OLLAMA_MODEL=$Model"
} else {
    Write-Warn ".env not found — please set OLLAMA_MODEL=$Model manually"
}

# ── Done ──────────────────────────────────────────────────────────────────────
Write-Host "`n=============================================" -ForegroundColor Green
Write-Host "  Setup complete! Run the app:" -ForegroundColor Green
Write-Host "    cd agri_crop_recommendation" -ForegroundColor White
Write-Host "    python main.py" -ForegroundColor White
Write-Host "=============================================`n" -ForegroundColor Green
