#!/usr/bin/env python3
"""
Large‑Codebase Refactor Assistant
================================
This script lets you **index** a big repository, then run an **LLM‑powered chat loop** that
can _read_, _search_, and _write_ files inside a **target workspace** using Ollama’s
tool‑calling.

Usage
-----
# 1) Build (or rebuild) the FAISS index from your source code
python refactor_agent.py index --src /path/to/legacy_code --index .code_index

# 2) Start an interactive chat session powered by Ollama (e.g. Llama‑3‑70B)
python refactor_agent.py chat --index .code_index --workspace refactored --model llama3:70b

The model will be able to call these tools:
- **read_file(path)**
- **write_file(path, content)**
- **search_code(query, k)**

Each tool call is executed by *this* script; the model never touches the
filesystem directly.

Dependencies
------------
 pip install "faiss-cpu>=1.7" llama-index==0.10.* langchain tiktoken ollama

Tested on Python 3.10+.
"""
from __future__ import annotations
import argparse, json, os, readline, shutil, sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import ollama  # pip install ollama‑python
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext, StorageContext
from llama_index.embeddings.huggingface.base import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings

# ---------- Configuration ---------- #
CHUNK_SIZE = 4000          # characters per chunk when indexing
CHUNK_OVERLAP = 200        # overlap to keep context
EMBED_MODEL = "nomic-ai/nomic-embed-text-v1"

# ---------- Indexing Phase ---------- #

# Settings.llm = Ollama(model="llama2", request_timeout=120.0)

def build_index(src_dir: Path, index_dir: Path) -> None:
    """Walk *src_dir*, chunk code/doc files, and build a FAISS vector store."""
    print(f"[index] Scanning {src_dir} …")
    reader = SimpleDirectoryReader(str(src_dir), recursive=True, required_exts=(".xml", ".scss", ".html", ".ts", ".java", ".sql", ".js"))
    docs = reader.load_data()

    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL, trust_remote_code=True)
    # service_context = ServiceContext.from_defaults(embed_model=embed_model, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    Settings.embed_model = embed_model
    Settings.node_parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    # index = VectorStoreIndex.from_documents(docs, service_context=service_context, show_progress=True)
    index = VectorStoreIndex.from_documents(docs, embed_model=embed_model, show_progress=True)

    index_dir.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(index_dir))
    print(f"[index] Saved index to {index_dir}")

# ---------- Chat / Agent Phase ---------- #
@dataclass
class AgentConfig:
    index_dir: Path
    workspace: Path
    model: str


def load_index(index_dir: Path):
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    storage_context = StorageContext.from_defaults(persist_dir=str(index_dir))
    service_context = ServiceContext.from_defaults(embed_model=embed_model)
    return VectorStoreIndex.from_vector_store(FaissVectorStore.from_persist_dir(index_dir), storage_context=storage_context, service_context=service_context)

# --- Tool implementations --- #

def read_file(path: str, workspace: Path) -> str:
    fp = workspace / path
    if not fp.exists():
        return f"ERROR: {path} does not exist in workspace"
    return fp.read_text()


def write_file(path: str, content: str, workspace: Path) -> str:
    fp = workspace / path
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(content)
    return f"Wrote {len(content)} bytes to {fp}"


def search_code(query: str, k: int, vector_index, src_dir: Path) -> str:
    retriever = vector_index.as_retriever(similarity_top_k=k)
    nodes = retriever.retrieve(query)
    results = []
    for node in nodes:
        rel_path = Path(node.node.source_node.metadata.get("file_path", "?"))
        snippet = node.node.text[:400].replace("\n", " ") + " …"
        results.append(f"{rel_path}: {snippet}")
    return "\n".join(results)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file from the workspace directory and return its content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative file path"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file in the workspace (creates parent dirs as needed).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Semantic search over the indexed source code, returns top‑k snippets",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are Refactor‑GPT, a senior software engineer.
You have access to a 1.4 M LOC codebase via three tools: read_file, write_file, search_code.
Plan refactors step‑by‑step; for each code change call write_file.
Ask the user for clarification if needed. Do not overwrite files unless you have a very good reason.
"""


def interactive_chat(agent_cfg: AgentConfig):
    vector_index = load_index(agent_cfg.index_dir)
    agent_cfg.workspace.mkdir(parents=True, exist_ok=True)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    print("💬 Type your instructions (or 'exit'):")
    while True:
        user_input = input("🟢 > ")
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        messages.append({"role": "user", "content": user_input})

        response = ollama.chat(model=agent_cfg.model, messages=messages, tools=TOOLS)
        assistant_msg = response["message"]
        messages.append({"role": "assistant", **assistant_msg})

        # Execute any tool calls
        for call in assistant_msg.get("tool_calls", []):
            fn_name = call["function"]["name"]
            args = json.loads(call["function"]["arguments"])
            if fn_name == "read_file":
                result = read_file(args["path"], agent_cfg.workspace)
            elif fn_name == "write_file":
                result = write_file(args["path"], args["content"], agent_cfg.workspace)
            elif fn_name == "search_code":
                k = args.get("k", 5)
                result = search_code(args["query"], k, vector_index, agent_cfg.index_dir)
            else:
                result = f"ERROR: unknown tool {fn_name}"

            # Send result back so model can continue
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": result,
            })

            # let the model process the tool result
            follow_up = ollama.chat(model=agent_cfg.model, messages=messages)
            messages.append({"role": "assistant", **follow_up["message"]})
            print(follow_up["message"]["content"])

# ---------- CLI ---------- #

def main():
    parser = argparse.ArgumentParser(description="Large‑Codebase Refactor Assistant (Ollama)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="Build FAISS vector index from source code")
    p_index.add_argument("--src", required=True, type=Path, help="Source code directory")
    p_index.add_argument("--index", required=True, type=Path, help="Index output directory")

    p_chat = sub.add_parser("chat", help="Start interactive refactor chat")
    p_chat.add_argument("--index", required=True, type=Path, help="Existing index directory")
    p_chat.add_argument("--workspace", required=True, type=Path, help="Directory to write refactored files")
    p_chat.add_argument("--model", default="llama3", help="Ollama model name (default: llama3)")

    args = parser.parse_args()

    if args.cmd == "index":
        build_index(args.src, args.index)
    elif args.cmd == "chat":
        cfg = AgentConfig(index_dir=args.index, workspace=args.workspace, model=args.model)
        interactive_chat(cfg)

if __name__ == "__main__":
    main()
