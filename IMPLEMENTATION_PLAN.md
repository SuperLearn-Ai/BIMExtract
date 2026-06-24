# Implementation Plan: BIMExtract

This document tracks the implementation progress, technical debt, and future roadmap for the `BIMExtract` repository.

## Currently Implemented

### Agentic Orchestration
- `[x]` Google Antigravity `preprocessing_agent.py` integration for managing the ingestion workflow.
- `[x]` Pre-tool call hooks and session initialization logic.

### Ultra-Cost-Optimized Pipeline
- `[x]` Stage 1: Visual Parsing using `PaddleOCR`.
- `[x]` Stage 2: Qwen3-30B-A3B LangChain orchestration for complexity-routed chunking.
- `[x]` Redis caching layer achieving ~95% cache hit rates.
- `[x]` Unified CLI runner (`run.py`) supporting local, langchain, and runpod modes.
- `[x]` RunPod deployment generation script (`deploy_runpod.py`).
- `[x]` Comprehensive test suites (`simple_pdf_test.py`, `test_pdf_extraction.py`).

### Processing Features
- `[x]` Basic PDF conversion and error handling.
- `[x]` Inverse-HyDE and Contextual Enrichment foundations.

## Remaining Work & Roadmap

### Short-Term Tasks
- `[ ]` **Robust Error Handling**: Enhance fallback mechanisms for corrupted or complex PDF types.
- `[ ]` **Poppler Checks**: Add explicit system-level checks for `poppler-utils` in the initialization sequence.
- `[ ]` **KV-Cache Materialization**: Complete the end-to-end integration of pre-computed KV-cache storing logic.

### 2025 Agentic Hybrid Vision (Mid-to-Long Term)
*As outlined in the AgenticPipeline research documents:*
- `[ ]` **HPC-ColPali Integration**: Replace or augment Stage 1 with HPC-ColPali for extreme visual compression (32x32 patch embeddings).
- `[ ]` **Symbol-Layout-Tree (SLT) Parser**: Integrate LGAP for ≥98% formula and LaTeX accuracy.
- `[ ]` **SuperRAG Integration**: Build a layout-aware graph over text blocks, tables, and figures to maintain spatial hierarchy.
- `[ ]` **Milvus 2.6 Migration**: Establish vector storage using Milvus 2.6 with Int8 compression and tiered hot/cold storage.
