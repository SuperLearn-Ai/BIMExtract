# Changelog: BIMExtract

All notable changes to the `BIMExtract` repository will be documented in this file.

## [v1.0.0] - Current Baseline State

### Added
- **Antigravity Orchestration**: `preprocessing_agent.py` with hooks to oversee document ingestion and caching.
- **Ultra-Cost-Optimized Pipeline**: 
  - Implementation of Qwen3-30B-A3B Stage 2 chunking via LangChain.
  - Stage 1 Visual Parsing utilizing PaddleOCR, pdf2image, and PIL.
  - Redis caching logic for optimizing repetitive document processing.
  - RunPod deployment configurations and script (`deploy_runpod.py`) for scaling A5000 cloud instances.
  - Unified CLI runner (`run.py`) handling different execution modes.
- **Testing Suite**: Basic and comprehensive PDF extraction tests (`simple_pdf_test.py`, `test_pdf_extraction.py`).
- **Research Documents**: Architectural blueprints in `AgenticPipeline` folder for future integration (HPC-ColPali, SuperRAG, Milvus 2.6).

### Changed
- Standardized the documentation to accurately reflect the multi-stage pipeline and extreme cost reductions ($0.012/1K pages).
