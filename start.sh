#!/bin/bash
set -e

if [ ! -f ".env" ]; then
  if [ -z "$FW_CLI_API_KEY" ]; then
    cp .env_example .env
    echo "First-time setup: .env created from .env_example"
    echo "  → Open .env and set FW_CLI_API_KEY=<your_api_key>"
    echo "  → Then re-run: bash start.sh"
    exit 1
  fi
fi

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install -q -r requirements.txt
streamlit run Home.py
