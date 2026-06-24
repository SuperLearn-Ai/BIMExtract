---
name: vlm_ingestion
description: "Use this skill to ingest documents using Docling and PaddleOCR-VL. It extracts highly accurate text, markdown, and layout information from PDFs and images."
---

# VLM Ingestion Skill

This skill allows the agent to process documents (like PDFs or images) using state-of-the-art vision models: Docling (for layout and tables) and PaddleOCR-VL (for highly accurate text extraction).

## Usage

Use the provided `ingest.py` script to process a document.

### Command

```bash
python scripts/ingest.py <path_to_document> [--output_dir <output_dir>]
```

### Arguments
- `<path_to_document>`: Absolute or relative path to the PDF or image file.
- `--output_dir`: (Optional) Directory to save the extracted JSON and Markdown. Defaults to `./output`.

The script will return a JSON structure containing:
- `pages`: A list of page texts and layout elements.
- `metadata`: Document metadata.
- `markdown`: The full reconstructed markdown text.
