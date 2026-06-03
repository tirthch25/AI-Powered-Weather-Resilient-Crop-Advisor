"""
Run the Farmer Crop Recommendation Website

This script starts the FastAPI server with the web interface.
"""

import uvicorn
import sys
import io
from pathlib import Path

# Fix Windows terminal encoding (cp1252 can't print emojis)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    print("=" * 70)
    print("[Crop Advisor] FARMER CROP RECOMMENDATION SYSTEM")
    print("=" * 70)
    print("\nStarting web server...")
    print("\n>> Access the website at: http://localhost:8000")
    print(">> API documentation at:  http://localhost:8000/docs")
    print("\nPress CTRL+C to stop the server\n")
    print("=" * 70)
    
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
