import json
import argparse
import os
from pathlib import Path

def enrich_chunks(tree_data):
    """Append global context to chunks"""
    doc_context = f"Source: {tree_data.get('metadata', {}).get('source', 'Unknown')}"
    
    enriched_chunks = []
    for child in tree_data.get("children", []):
        enriched_content = f"[{doc_context}] {child.get('content', '')}"
        enriched_chunks.append({
            "id": child.get("id"),
            "content": enriched_content,
            "original_content": child.get('content', '')
        })
        
    return enriched_chunks

def main():
    parser = argparse.ArgumentParser(description="Enrich chunks with context")
    parser.add_argument("input", help="Path to PageIndex tree JSON")
    parser.add_argument("--output_dir", default="./output", help="Output directory")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        return
        
    with open(args.input, 'r') as f:
        data = json.load(f)
        
    print(f"Enriching chunks for {args.input}...")
    enriched = enrich_chunks(data)
    
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, f"{Path(args.input).stem}_enriched.json")
    with open(out_file, 'w') as f:
        json.dump(enriched, f, indent=2)
        
    print(f"Enriched chunks saved to {out_file}")

if __name__ == "__main__":
    main()
