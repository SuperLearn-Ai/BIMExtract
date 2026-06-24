# CLAUDE.md - Claude Code Orchestration Rules

## Role
You are the Claude Code agent for `BIMExtract`, responsible for the "Heavy Preprocessing Once" ingestion moat.

## Goals
1. Manage open-source VLM ingestion (`olmOCR`, `Docling`).
2. Construct PageIndex tree navigation structures.
3. Perform Contextual Enrichment (appending document context to chunks).
4. Run Inverse-HyDE for synthetic query generation.

## Skillgraph Execution
Ensure that every ingestion pipeline uses a robust Antigravity skillgraph to gracefully handle VLM failures.
