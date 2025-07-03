#!/bin/bash
DIR=`dirname "$0"`	# Get the directory of this shell-script
cd "$DIR" || exit
DIR=`pwd`	# Get the absolute path of this shell-script
echo "Running from $DIR"

./set-venv.sh
source .venv/bin/activate



python refactor_agent.py index --src ../my_code --index ../.code_index

