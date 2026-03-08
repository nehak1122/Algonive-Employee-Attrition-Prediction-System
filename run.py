"""
EAPS - Run Script
Trains the model and launches the API + Dashboard.
"""

import os
import sys
import subprocess
import time


def train_model():
    """Train the ML model."""
    print("=" * 60)
    print("  EAPS - Training ML Model")
    print("=" * 60)
    subprocess.run([sys.executable, "ml/train_model.py"], cwd=os.path.dirname(__file__))


def start_api():
    """Start the FastAPI server in background."""
    print("\n" + "=" * 60)
    print("  EAPS - Starting API Server (port 8000)")
    print("=" * 60)
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--port", "8000", "--reload"],
        cwd=os.path.dirname(__file__),
    )


def start_dashboard():
    """Start the Streamlit dashboard."""
    print("\n" + "=" * 60)
    print("  EAPS - Starting Dashboard (port 8501)")
    print("=" * 60)
    return subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "dashboard/app.py",
         "--server.port", "8501", "--server.headless", "true"],
        cwd=os.path.dirname(__file__),
    )


if __name__ == "__main__":
    # Step 1: Train model
    train_model()

    # Step 2: Start API
    api_proc = start_api()
    time.sleep(3)  # Wait for API to start

    # Step 3: Start Dashboard
    dash_proc = start_dashboard()

    print("\n" + "=" * 60)
    print("  EAPS is running!")
    print("  API:       http://localhost:8000")
    print("  Dashboard: http://localhost:8501")
    print("  API Docs:  http://localhost:8000/docs")
    print("=" * 60)
    print("\nPress Ctrl+C to stop all services.\n")

    try:
        api_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        api_proc.terminate()
        dash_proc.terminate()
