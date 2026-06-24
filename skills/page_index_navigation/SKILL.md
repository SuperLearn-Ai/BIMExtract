---
name: page_index_navigation
description: "Builds deterministic tree structures for documents to preserve layout and allow exact matching during retrieval."
---

# PageIndex Navigation Skill

This skill creates a deterministic tree structure from ingested document data.

## Usage

```bash
python scripts/build_tree.py <path_to_ingested_json> [--output_dir <output_dir>]
```

The script generates a `tree.json` representing the hierarchical layout of the document.
