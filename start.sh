#!/usr/bin/env bash
set -e

# Install Python dependencies
pip install -r requirements.txt

# Provide a default port fallback for environments where $PORT is empty
PORT=${PORT:-8501}

# Ensure Streamlit sees a valid port env var (some runtimes set an empty STREAMLIT_SERVER_PORT)
export STREAMLIT_SERVER_PORT="$PORT"

# Start Streamlit (use $PORT provided by Replit or fallback)
streamlit run ser_ravdess_6class/app.py --server.port=$PORT --server.address=0.0.0.0 --server.enableCORS=false
