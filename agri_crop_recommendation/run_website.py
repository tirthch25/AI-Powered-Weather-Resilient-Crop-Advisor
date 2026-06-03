"""
Run the Farmer Crop Recommendation Website

This script starts the FastAPI server with the web interface.
All errors (including file name and line number) are printed clearly in the terminal.
"""

import uvicorn
import sys
import io
import logging
import traceback
from pathlib import Path

# Fix Windows terminal encoding (cp1252 can't print emojis)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# ── Configure logging: show file name + line number in every log message ──────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),          # terminal
    ]
)

# Show WARNING+ from third-party libs, but full DEBUG from our own code
logging.getLogger("src").setLevel(logging.DEBUG)
logging.getLogger("uvicorn").setLevel(logging.INFO)
logging.getLogger("fastapi").setLevel(logging.INFO)

logger = logging.getLogger(__name__)


def _check_imports():
    """
    Try importing the app before uvicorn starts.
    This catches missing packages / syntax errors and prints the exact
    file + line that caused the failure — making debugging much easier.
    """
    print("\n[CHECK] Verifying imports...")
    try:
        from src.api.app import app  # noqa: F401
        print("[CHECK] All imports OK\n")
    except Exception:
        print("\n" + "=" * 70)
        print("STARTUP ERROR — failed to import the app")
        print("=" * 70)
        print(traceback.format_exc())          # <-- shows exact file + line
        print("=" * 70)
        print("Fix the error above, then re-run this script.\n")
        sys.exit(1)


if __name__ == "__main__":
    print("=" * 70)
    print("  FARMER CROP RECOMMENDATION SYSTEM")
    print("=" * 70)
    print("\nStarting web server...")
    print(">> Access the website at: http://localhost:8000")
    print(">> API documentation at:  http://localhost:8000/docs")
    print("\nPress CTRL+C to stop the server")
    print("=" * 70)

    # Run import check BEFORE starting uvicorn
    _check_imports()

    try:
        uvicorn.run(
            "src.api.app:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
        )
    except KeyboardInterrupt:
        print("\n[Server stopped by user]")
    except Exception:
        print("\n" + "=" * 70)
        print("FATAL ERROR — server crashed")
        print("=" * 70)
        print(traceback.format_exc())          # <-- exact file + line
        print("=" * 70)
        sys.exit(1)
