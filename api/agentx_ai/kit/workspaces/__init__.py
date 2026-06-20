"""File Workspaces & Document RAG.

A persistent, named container of user-provided files with a searchable manifest.
Three-store separation (todo/backlog/workspaces.md §6):
  - bytes   → content-addressed blob store on disk (``storage.py``)
  - manifest→ Postgres ``workspaces`` / ``documents`` (``repository.py``)
  - vectors → Postgres + pgvector ``document_chunks`` (``repository.py`` + ``retrieval.py``)

Ingestion (``ingestion.py``) parses → chunks → embeds → stores, then auto-tags and
summarizes each file in the background so the manifest reads like a data-warehouse index.
"""
