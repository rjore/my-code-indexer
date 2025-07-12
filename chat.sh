#!/bin/bash
DIR=`dirname "$0"`	# Get the directory of this shell-script
cd "$DIR" || exit
DIR=`pwd`	# Get the absolute path of this shell-script
echo "Running from $DIR"

./set-venv.sh
source .venv/bin/activate


# python refactor_agent.py chat --index ../.code_index --workspace ../refactored-code --model llama3.2:latest

python refactor_agent.py chat --store pg \
  --pg-uri "postgresql://vctr:vctr123@localhost/vctr" \
  --table vectors \
  --workspace ~/codes/refactored-code \
  --model llama3.2:latest
