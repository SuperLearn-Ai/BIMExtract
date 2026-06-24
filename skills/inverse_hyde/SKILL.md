---
name: inverse_hyde
description: "Pre-computes synthetic user queries for each chunk to accelerate test-time retrieval."
---

# Inverse-HyDE Skill

Generates synthetic queries that would logically retrieve a given chunk.

## Usage

```bash
python scripts/generate.py <path_to_enriched_json> [--output_dir <output_dir>]
```
