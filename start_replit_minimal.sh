#!/usr/bin/env bash
set -e

# Install only minimal dependencies to avoid heavy native wheels
pip install -r replit_requirements.txt

# Ensure STREAMLIT_SERVER_PORT is set
PORT=${PORT:-8501}
export STREAMLIT_SERVER_PORT="$PORT"

# Run the Streamlit app in demo mode (app will fallback if heavy model missing)
streamlit run ser_ravdess_6class/app.py --server.port=$PORT --server.address=0.0.0.0 --server.enableCORS=false
