#!/bin/bash
DIR=`dirname "$0"`	# Get the directory of this shell-script
cd "$DIR"

if [ -d .venv ]; then
  python3 -m venv .venv
  source .venv/bin/activate
  # pip install --upgrade avitr-commons
else
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
fi
