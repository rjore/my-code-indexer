Make sure virtuan env is activated:
> source .venv/bin/activate
Note: If venv is not present then create it first by: python3 -m venv .venv ; source .venv/bin/activate ; pip install -r requirements.txt

1. To create index from code:
> python refactor_agent.py index --src /path/to/code --index ../.code_index


2. To chat and cerate refactored code
> python refactor_agent.py chat --index ../.code_index --workspace ../refactored-code --model llama3:70b
