#!/bin/bash
DIR=`dirname "$0"`	# Get the directory of this shell-script
cd "$DIR" || exit
DIR=`pwd`	# Get the absolute path of this shell-script
echo "Running from $DIR"

./set-venv.sh
source .venv/bin/activate



# python refactor_agent.py index --src ~/codes/algo-service --index ~/codes/.code_index
python refactor_agent.py index \
  --src ~/codes/algo-service \
  --store pg \
  --pg-uri "postgresql://vctr:vctr123@localhost/vctr" \
  --table vectors

