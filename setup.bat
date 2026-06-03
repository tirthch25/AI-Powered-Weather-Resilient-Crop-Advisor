@echo off
title Indian Farmer Crop Recommendation System — Setup
color 0A

echo.
echo  ================================================================
echo   Indian Farmer Crop Recommendation System — Setup
echo   LLM: LLaMA 3.2 (local via Ollama) + Gemini fallback
echo  ================================================================
echo.

:: Check Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python is not installed or not in PATH.
    echo.
    echo  Download Python 3.8+ from: https://www.python.org/downloads/
    echo  Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo  Python found:
python --version
echo.

:: Check if Ollama is already installed (optional, setup_project.py will handle it)
where ollama >nul 2>&1
if not errorlevel 1 (
    echo  Ollama found in PATH.
) else (
    if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
        echo  Ollama found at: %LOCALAPPDATA%\Programs\Ollama\ollama.exe
    ) else (
        echo  [i] Ollama not found -- setup script will guide you through installation.
        echo  [i] Download from: https://ollama.com/download
    )
)
echo.

:: Run the setup script
python setup_project.py %*

echo.
pause
