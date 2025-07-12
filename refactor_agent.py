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
import argparse, json, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import ollama  # pip install ollama‑python
from llama_index.embeddings.huggingface.base import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
# from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import Settings

# ---------- Configuration ---------- #
CHUNK_SIZE = 4000          # characters per chunk when indexing
CHUNK_OVERLAP = 200        # overlap to keep context
EMBED_MODEL = "nomic-ai/nomic-embed-text-v1"

# ---------- Indexing Phase ---------- #

# Helper: create & register embed model once

def make_embed_model():
    emb = HuggingFaceEmbedding(model_name=EMBED_MODEL, trust_remote_code=True)
    Settings.embed_model = emb  # global—needed by retrievers
    return emb

# ───────────────── Indexing ─────────────────────── #

def build_index(
    src_dir: Path,
    faiss_dir: Optional[Path],
    store: str,
    pg_uri: Optional[str] = None,
    pg_table: str = "vectors",
):
    print(f"[index] Scanning {src_dir} …")
    reader = SimpleDirectoryReader(
        str(src_dir),
        recursive=True,
        required_exts=(".py", ".java", ".ts", ".js", ".sql", ".md"),
    )
    docs = reader.load_data()
    embed_model = make_embed_model()

    if store == "faiss":
        if faiss_dir is None:
            raise ValueError("--index DIRECTORY is required for faiss store")
        index = VectorStoreIndex.from_documents(
            docs,
            embed_model=embed_model,
            show_progress=True,
        )
        faiss_dir.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=str(faiss_dir))
        print(f"[index] Saved FAISS index to {faiss_dir}")

    elif store == "pg":
        if pg_uri is None:
            raise ValueError("--pg-uri must be supplied for pg store")
        vector_store = PGVectorStore(
            connection_string=pg_uri,
            # prefer_async=False,
            table_name=pg_table,
        )
        index = VectorStoreIndex.from_documents(
            docs,
            embed_model=embed_model,
            vector_store=vector_store,
            show_progress=True,
        )
        print(f"[index] Stored embeddings in PostgreSQL table '{pg_table}'")
    else:
        raise ValueError("store must be 'faiss' or 'pg'")

# ───────────────── Agent Config / Loader ────────── #
@dataclass
class AgentConfig:
    store: str
    faiss_dir: Optional[Path]
    pg_uri:   Optional[str]
    pg_table: str
    workspace: Path
    model: str


def load_index(cfg: AgentConfig):
    embed_model = make_embed_model()

    if cfg.store == "faiss":
        if cfg.faiss_dir is None:
            raise ValueError("--index DIR needed for faiss")
        storage_ctx = StorageContext.from_defaults(persist_dir=str(cfg.faiss_dir))
        vs = FaissVectorStore.from_persist_dir(cfg.faiss_dir)
        return VectorStoreIndex.from_vector_store(vs, storage_context=storage_ctx, embed_model=embed_model)

    if cfg.store == "pg":
        vs = PGVectorStore(
            connection_string=cfg.pg_uri,
            # prefer_async=False,
            table_name=cfg.pg_table,
        )
        return VectorStoreIndex.from_vector_store(vs, embed_model=embed_model)

    raise ValueError("Unsupported store type")

# ───────────────── LLM Tools ───────────────────── #

def read_file(path: str, workspace: Path) -> str:
    fp = workspace / path
    return fp.read_text() if fp.exists() else f"ERROR: {path} not found"


def write_file(path: str, content: str, workspace: Path) -> str:
    fp = workspace / path
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(content)
    return f"Wrote {len(content)} bytes to {fp}"


def search_code(query: str, k: int, vector_index) -> str:
    retriever = vector_index.as_retriever(similarity_top_k=k)
    nodes = retriever.retrieve(query)
    return "\n".join(
        f"{Path(n.node.source_node.metadata.get('file_path','?'))}: " +
        (n.node.text[:200].replace('\n',' ') + " …") for n in nodes
    )

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read workspace file.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write file to workspace.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Vector search over indexed codebase.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "k": {"type": "integer", "default": 5}}, "required": ["query"]},
        },
    },
]
SYSTEM_PROMPT = (
    "You are Refactor‑GPT, a senior software engineer. "
    "Use read_file, write_file, and search_code to plan and apply refactors step‑by‑step."
)

# ───────────────── Chat Loop ───────────────────── #

def interactive_chat(cfg: AgentConfig):
    vector_index = load_index(cfg)
    cfg.workspace.mkdir(parents=True, exist_ok=True)
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("💬  Enter instructions (exit/quit to stop)…")
    while True:
        try:
            user = input("🟢 > ")
        except EOFError:
            break
        if user.lower() in {"exit", "quit"}:
            break
        msgs.append({"role": "user", "content": user})
        rsp = ollama.chat(model=cfg.model, messages=msgs, tools=TOOLS)["message"]
        # msgs.append({"role": "assistant", **rsp})
        msgs.append({"role": "assistant", "content": rsp["content"]})
        print(rsp["content"].strip())
        for call in rsp.get("tool_calls", []):
            fn = call["function"]["name"]
            print(f"FN: {fn}")
            fno: ollama._types.Message.ToolCall.Function = call["function"]
            args = fno.get("arguments", "{}")
            if fn == "read_file":
                out = read_file(args["path"], cfg.workspace)
            elif fn == "write_file":
                out = write_file(args["path"], args["content"], cfg.workspace)
            elif fn == "search_code":
                out = search_code(args["query"], args.get("k", 5), vector_index)
            else:
                out = f"ERROR: unknown tool {fn}"
            msgs.append({"role": "tool", "tool_call_id": call["id"], "content": out})
            follow = ollama.chat(model=cfg.model, messages=msgs)["message"]
            msgs.append({"role": "assistant", **follow})
            print(follow["content"].strip())

# ───────────────── CLI ─────────────────────────── #

def main():
    p = argparse.ArgumentParser(description="Large‑Codebase Refactor Assistant with FAISS or PGVector backend")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("index", help="Create embedding index from source code")
    pi.add_argument("--src", required=True, type=Path)
    pi.add_argument("--store", choices=["faiss", "pg"], default="faiss")
    pi.add_argument("--index", type=Path, help="Directory for FAISS (store=faiss)")
    pi.add_argument("--pg-uri", help="PostgreSQL URI (store=pg)")
    pi.add_argument("--table", default="vectors")

    pc = sub.add_parser("chat", help="Chat with the indexed codebase")
    pc.add_argument("--store", choices=["faiss", "pg"], default="faiss")
    pc.add_argument("--index", type=Path, help="Directory with FAISS index")
    pc.add_argument("--pg-uri", help="PostgreSQL URI (store=pg)")
    pc.add_argument("--table", default="vectors")
    pc.add_argument("--workspace", required=True, type=Path)
    pc.add_argument("--model", default="llama3")

    args = p.parse_args()

    if args.cmd == "index":
        build_index(args.src, args.index, args.store, args.pg_uri, args.table)

    elif args.cmd == "chat":
        cfg = AgentConfig(
            store=args.store,
            faiss_dir=args.index,
            pg_uri=args.pg_uri,
            pg_table=args.table,
            workspace=args.workspace,
            model=args.model,
        )
        interactive_chat(cfg)

if __name__ == "__main__":
    main()
