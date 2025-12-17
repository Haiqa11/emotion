#!/usr/bin/env bash
set -e

# Install Python dependencies
pip install -r requirements.txt

# Start Streamlit (use $PORT provided by Replit)
streamlit run ser_ravdess_6class/app.py --server.port=$PORT --server.address=0.0.0.0 --server.enableCORS=false
