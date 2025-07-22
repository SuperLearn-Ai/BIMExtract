#!/usr/bin/env python3
"""
Simple runner for Ultra-Cost-Optimized Pipeline with Qwen3-30B-A3B
Provides command-line interface for easy testing and production use
"""

import os
import sys
import argparse
import asyncio
import logging
from pathlib import Path
from typing import List, Optional

# Add src directory to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

try:
    from main_pipeline import UltraOptimizedPipeline
except ImportError as e:
    print(f"❌ Error importing pipeline: {e}")
    print("Make sure you're in the ultra_cost_optimized_pipeline directory")
    print("and have installed all requirements: pip install -r requirements.txt")
    sys.exit(1)


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pipeline_run.log'),
            logging.StreamHandler()
        ]
    )


def find_documents(paths: Optional[List[str]] = None) -> List[str]:
    """Find documents to process"""
    if paths:
        # Use provided paths
        valid_paths = []
        for path in paths:
            if os.path.exists(path):
                valid_paths.append(path)
            else:
                print(f"⚠️  Warning: Document not found: {path}")
        return valid_paths
    
    # Auto-discover documents in documents/ directory
    documents_dir = Path("documents")
    if not documents_dir.exists():
        documents_dir.mkdir()
        print(f"📁 Created documents directory: {documents_dir}")
        return []
    
    # Find PDF files
    pdf_files = list(documents_dir.glob("*.pdf"))
    return [str(f) for f in pdf_files]


async def run_pipeline(
    documents: List[str],
    config_path: Optional[str] = None,
    batch_size: int = 2,
    show_metrics: bool = True
):
    """Run the pipeline on documents"""
    
    if not documents:
        print("❌ No documents found to process")
        print("Add PDF files to the documents/ directory or specify paths with --files")
        return
    
    print(f"🚀 Starting Qwen3-30B-A3B Pipeline")
    print(f"Documents to process: {len(documents)}")
    
    try:
        # Initialize pipeline
        pipeline = UltraOptimizedPipeline(config_path)
        print("✅ Pipeline initialized successfully")
        
        # Process documents
        if len(documents) == 1:
            # Single document processing
            print(f"\n📄 Processing: {os.path.basename(documents[0])}")
            result = await pipeline.process_document(documents[0])
            results = [result]
        else:
            # Batch processing
            print(f"\n📚 Batch processing {len(documents)} documents...")
            results = await pipeline.process_documents(documents)
        
        # Display results
        print("\n" + "="*60)
        print("📊 PROCESSING RESULTS")
        print("="*60)
        
        success_count = 0
        total_cost = 0.0
        total_time = 0.0
        
        for result in results:
            doc_name = os.path.basename(result.document_path)
            if result.success:
                success_count += 1
                total_cost += result.total_cost
                total_time += result.total_processing_time
                
                print(f"✅ {doc_name}")
                print(f"   Time: {result.total_processing_time:.2f}s")
                print(f"   Cost: ${result.total_cost:.4f}")
                
                if result.stage2_result:
                    print(f"   Chunks: {len(result.stage2_result.chunks)}")
                    print(f"   Cache Hit: {result.stage2_result.cache_hit_rate:.1%}")
                    
                    # Show sample chunks if requested
                    if len(result.stage2_result.chunks) > 0:
                        sample_chunk = result.stage2_result.chunks[0]
                        print(f"   Complexity: {sample_chunk.complexity.value}")
                        print(f"   Model: {sample_chunk.model_used}")
                        print(f"   Thinking: {'Yes' if sample_chunk.thinking_used else 'No'}")
            else:
                print(f"❌ {doc_name}")
                print(f"   Error: {result.error_message}")
            print()
        
        # Overall metrics
        print("📈 OVERALL METRICS")
        print("="*60)
        print(f"Success Rate: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
        
        if success_count > 0:
            avg_cost = total_cost / success_count
            avg_time = total_time / success_count
            cost_per_1k_pages = avg_cost * 1000  # Rough estimate
            
            print(f"Average Cost per Document: ${avg_cost:.4f}")
            print(f"Average Processing Time: {avg_time:.2f}s")
            print(f"Estimated Cost per 1K Pages: ${cost_per_1k_pages:.4f}")
            print(f"Total Pipeline Cost: ${total_cost:.4f}")
            
            # Cost breakdown
            if results[0].success:
                stage1_cost = results[0].stage1_result.cost_estimate if results[0].stage1_result else 0
                stage2_cost = results[0].stage2_result.cost_estimate if results[0].stage2_result else 0
                stage3_cost = results[0].stage3_result.cost_estimate if results[0].stage3_result else 0
                
                print(f"\n💰 Cost Breakdown (per document):")
                print(f"  Stage 1 (Visual): ${stage1_cost:.4f}")
                print(f"  Stage 2 (Qwen3): ${stage2_cost:.4f}")  
                print(f"  Stage 3 (Vector): ${stage3_cost:.4f}")
                
            # Performance targets check
            print(f"\n🎯 Performance Targets:")
            if success_count > 0 and results[0].stage2_result:
                stage2_cost = results[0].stage2_result.cost_estimate
                target_met = "✅" if stage2_cost <= 0.015 else "❌"
                print(f"  Stage 2 Cost Target ($0.012): {target_met} ${stage2_cost:.4f}")
                
                time_met = "✅" if avg_time <= 10.0 else "❌"
                print(f"  Processing Time Target (<10s): {time_met} {avg_time:.2f}s")
        
        # Pipeline metrics
        if show_metrics:
            pipeline_metrics = pipeline.get_performance_metrics()
            if 'stage2_metrics' in pipeline_metrics:
                stage2_metrics = pipeline_metrics['stage2_metrics']
                print(f"\n🔍 Stage 2 Detailed Metrics:")
                print(f"  Cache Hit Rate: {stage2_metrics.get('cache_hit_rate', 0):.1%}")
                print(f"  Total Chunks Created: {stage2_metrics.get('total_chunks_created', 0)}")
                print(f"  Avg Cost per Document: ${stage2_metrics.get('cost_per_document', 0):.4f}")
        
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        logging.error(f"Pipeline error: {e}", exc_info=True)


def main():
    """Main command-line interface"""
    parser = argparse.ArgumentParser(
        description="Run Ultra-Cost-Optimized Pipeline with Qwen3-30B-A3B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all PDFs in documents/ directory
  python run_pipeline.py
  
  # Process specific files
  python run_pipeline.py --files documents/sample.pdf documents/sample2.pdf
  
  # Verbose logging
  python run_pipeline.py --verbose
  
  # Custom configuration
  python run_pipeline.py --config my_config.yaml
        """
    )
    
    parser.add_argument(
        '--files', '-f',
        nargs='+',
        help='Specific PDF files to process'
    )
    
    parser.add_argument(
        '--config', '-c',
        help='Path to custom configuration file'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=2,
        help='Number of documents to process concurrently (default: 2)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--no-metrics',
        action='store_true',
        help='Disable detailed metrics display'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run in test mode (create mock documents if needed)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Find documents
    documents = find_documents(args.files)
    
    if not documents and args.test:
        print("🧪 Test mode: Creating mock documents...")
        # Create simple test file
        documents_dir = Path("documents")
        documents_dir.mkdir(exist_ok=True)
        
        # Create a simple text file as PDF substitute for testing
        test_file = documents_dir / "test_document.txt"
        with open(test_file, 'w') as f:
            f.write("""
Test Document for Qwen3 Pipeline

Abstract:
This is a test document for validating the Qwen3-30B-A3B pipeline implementation.

Introduction:
The document extraction pipeline processes academic papers and technical documents
with high efficiency and cost optimization.

Mathematical Content:
Let f(x) = x^2 + 3x + 2 be our test function.
The derivative is f'(x) = 2x + 3.

Conclusion:
This pipeline achieves excellent cost-performance balance.
            """)
        
        print(f"📝 Created test document: {test_file}")
        # For testing, we'll treat .txt as .pdf
        documents = [str(test_file)]
    
    if not documents:
        print("❌ No documents found!")
        print("Options:")
        print("1. Add PDF files to the documents/ directory")
        print("2. Use --files to specify document paths")
        print("3. Use --test to create mock documents")
        return
    
    # Run pipeline
    try:
        asyncio.run(run_pipeline(
            documents=documents,
            config_path=args.config,
            batch_size=args.batch_size,
            show_metrics=not args.no_metrics
        ))
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logging.error(f"Unexpected error: {e}", exc_info=True)


if __name__ == "__main__":
    main() 