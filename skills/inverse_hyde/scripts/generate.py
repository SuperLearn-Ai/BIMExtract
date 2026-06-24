import json
import argparse
import os
from pathlib import Path
import asyncio

async def generate_synthetic_queries(chunks):
    """Generate synthetic queries for each chunk using Inverse-HyDE"""
    print("Generating synthetic queries...")
    await asyncio.sleep(1) # simulate model call
    
    results = []
    for chunk in chunks:
        # Simple stub for query generation
        query = f"What is the context regarding: {chunk['original_content'][:30]}...?"
        results.append({
            "chunk_id": chunk["id"],
            "synthetic_query": query,
            "chunk_content": chunk["content"]
        })
    return results

def main():
    parser = argparse.ArgumentParser(description="Generate Inverse-HyDE queries")
    parser.add_argument("input", help="Path to enriched chunks JSON")
    parser.add_argument("--output_dir", default="./output", help="Output directory")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        return
        
    with open(args.input, 'r') as f:
        chunks = json.load(f)
        
    print(f"Running Inverse-HyDE for {len(chunks)} chunks...")
    queries = asyncio.run(generate_synthetic_queries(chunks))
    
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, f"{Path(args.input).stem}_queries.json")
    with open(out_file, 'w') as f:
        json.dump(queries, f, indent=2)
        
    print(f"Synthetic queries saved to {out_file}")

if __name__ == "__main__":
    main()
