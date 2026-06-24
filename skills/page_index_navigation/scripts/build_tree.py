import json
import argparse
import os
from pathlib import Path

def build_tree(ingested_data):
    """Build a deterministic tree from ingested text content and layout"""
    # Simple stub implementation for tree building
    tree = {
        "root": "document",
        "metadata": ingested_data.get("metadata", {}),
        "children": []
    }
    
    text_content = ingested_data.get("text_content", "")
    lines = text_content.split('\n') if text_content else ["Empty document"]
    
    # Mock hierarchy
    for i, line in enumerate(lines[:10]):  # Limit to first 10 for stub
        if line.strip():
            tree["children"].append({
                "type": "node",
                "id": f"node_{i}",
                "content": line.strip()
            })
            
    return tree

def main():
    parser = argparse.ArgumentParser(description="Build PageIndex tree")
    parser.add_argument("input", help="Path to ingested JSON")
    parser.add_argument("--output_dir", default="./output", help="Output directory")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        return
        
    with open(args.input, 'r') as f:
        data = json.load(f)
        
    print(f"Building PageIndex tree for {args.input}...")
    tree = build_tree(data)
    
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, f"{Path(args.input).stem}_tree.json")
    with open(out_file, 'w') as f:
        json.dump(tree, f, indent=2)
        
    print(f"PageIndex tree saved to {out_file}")

if __name__ == "__main__":
    main()
