#!/usr/bin/env bash
# One-shot setup: create venv and install dependencies

python -m venv .venv

# Activate (works on bash/zsh; on Windows CMD use: .venv\Scripts\activate)
source .venv/Scripts/activate 2>/dev/null || source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✅  Setup complete!"
echo "   Activate:  source .venv/Scripts/activate"
echo "   Run app:   streamlit run app.py"
