#!/usr/bin/env python3
"""
Consolidated CLI Runner for Ultra-Cost-Optimized Pipeline
Supports both local and RunPod execution with LangChain orchestration
"""

import os
import sys
import argparse
import asyncio
import logging
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add src directory to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

def setup_logging(verbose: bool = False, log_file: str = "pipeline.log"):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

class DocumentFinder:
    """Utility to find and validate documents"""
    
    @staticmethod
    def find_documents(paths: Optional[List[str]] = None, docs_dir: str = "docs") -> List[str]:
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
        
        # Auto-discover documents in docs directory
        documents_dir = Path(docs_dir)
        if not documents_dir.exists():
            documents_dir.mkdir()
            print(f"📁 Created documents directory: {documents_dir}")
            return []
        
        # Find PDF files
        pdf_files = list(documents_dir.glob("*.pdf"))
        return [str(f) for f in pdf_files]

class PipelineRunner:
    """Main pipeline runner with support for different execution modes"""
    
    def __init__(self, config_path: Optional[str] = None, mode: str = "langchain"):
        self.config_path = config_path
        self.mode = mode
        self.pipeline = None
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Initialize the appropriate pipeline based on mode"""
        try:
            if self.mode == "langchain":
                from pipeline.pipeline_orchestrator import AdvancedPipelineOrchestrator
                self.pipeline = AdvancedPipelineOrchestrator(self.config_path)
                self.logger.info("✅ LangChain pipeline initialized")
            
            elif self.mode == "runpod":
                # Try to import RunPod pipeline components
                try:
                    from pipeline.runpod_langchain_api import RunPodLangChainPipeline
                    self.pipeline = RunPodLangChainPipeline(self.config_path)
                    self.logger.info("✅ RunPod LangChain pipeline initialized")
                except ImportError:
                    self.logger.warning("RunPod components not available, falling back to local")
                    from pipeline.pipeline_orchestrator import AdvancedPipelineOrchestrator
                    self.pipeline = AdvancedPipelineOrchestrator(self.config_path)
            
            elif self.mode == "local":
                from pipeline.main_pipeline import UltraOptimizedPipeline
                self.pipeline = UltraOptimizedPipeline(self.config_path)
                self.logger.info("✅ Local pipeline initialized")
            
            else:
                raise ValueError(f"Unknown mode: {self.mode}")
                
        except ImportError as e:
            self.logger.error(f"❌ Error importing pipeline: {e}")
            print("Make sure you've installed all requirements: pip install -r requirements.txt")
            raise
    
    async def process_single_document(self, document_path: str) -> Dict[str, Any]:
        """Process a single document"""
        self.logger.info(f"📄 Processing: {os.path.basename(document_path)}")
        start_time = time.time()
        
        try:
            # Use appropriate processing method based on pipeline type
            if hasattr(self.pipeline, 'run'):
                # LangChain orchestrator
                result = await self.pipeline.run({"document_path": document_path})
            elif hasattr(self.pipeline, 'process_document'):
                # Traditional pipeline
                result = await self.pipeline.process_document(document_path)
            else:
                raise AttributeError("Pipeline doesn't have recognized processing method")
            
            processing_time = time.time() - start_time
            
            return {
                "document": document_path,
                "status": "success",
                "processing_time": processing_time,
                "result": result
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"❌ Error processing {document_path}: {e}")
            
            return {
                "document": document_path,
                "status": "failed",
                "processing_time": processing_time,
                "error": str(e)
            }
    
    async def process_documents_batch(self, documents: List[str], batch_size: int = 2) -> List[Dict[str, Any]]:
        """Process documents in batches"""
        results = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            self.logger.info(f"📚 Processing batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
            
            # Process batch concurrently
            batch_tasks = [self.process_single_document(doc) for doc in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Handle exceptions in batch results
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    results.append({
                        "document": batch[j],
                        "status": "failed",
                        "error": str(result)
                    })
                else:
                    results.append(result)
        
        return results
    
    def display_results(self, results: List[Dict[str, Any]], show_metrics: bool = True):
        """Display processing results with metrics"""
        print("\n" + "="*70)
        print("📊 PROCESSING RESULTS")
        print("="*70)
        
        successful = [r for r in results if r.get("status") == "success"]
        failed = [r for r in results if r.get("status") == "failed"]
        
        print(f"📄 Total Documents: {len(results)}")
        print(f"✅ Successful: {len(successful)}")
        print(f"❌ Failed: {len(failed)}")
        
        if show_metrics and successful:
            total_time = sum(r.get("processing_time", 0) for r in successful)
            avg_time = total_time / len(successful)
            
            print(f"\n⏱️  PERFORMANCE METRICS")
            print(f"   Total Processing Time: {total_time:.2f}s")
            print(f"   Average Time per Document: {avg_time:.2f}s")
            print(f"   Throughput: {len(successful)/total_time:.2f} docs/sec")
        
        # Show detailed results
        for i, result in enumerate(results, 1):
            status_icon = "✅" if result.get("status") == "success" else "❌"
            doc_name = os.path.basename(result["document"])
            
            if result.get("status") == "success":
                time_info = f"({result.get('processing_time', 0):.2f}s)"
                print(f"{status_icon} {i:2d}. {doc_name} {time_info}")
            else:
                error = result.get("error", "Unknown error")
                print(f"{status_icon} {i:2d}. {doc_name} - Error: {error}")
        
        if failed:
            print(f"\n❌ Failed Documents:")
            for result in failed:
                print(f"   • {os.path.basename(result['document'])}: {result.get('error', 'Unknown error')}")

async def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Ultra-Cost-Optimized Document Extraction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                                    # Process all PDFs in docs/
  python run.py --files doc1.pdf doc2.pdf         # Process specific files
  python run.py --mode runpod --batch-size 5      # Use RunPod with larger batches
  python run.py --mode local --verbose            # Local mode with debug logs
  python run.py --docs-dir /path/to/docs          # Custom docs directory
        """
    )
    
    parser.add_argument("--files", nargs="+", help="Specific PDF files to process")
    parser.add_argument("--docs-dir", default="docs", help="Directory to search for PDFs (default: docs)")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--mode", choices=["langchain", "local", "runpod"], default="langchain",
                       help="Execution mode (default: langchain)")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for concurrent processing")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--no-metrics", action="store_true", help="Disable metrics display")
    parser.add_argument("--log-file", default="pipeline.log", help="Log file path")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose, args.log_file)
    logger = logging.getLogger(__name__)
    
    # Find documents
    documents = DocumentFinder.find_documents(args.files, args.docs_dir)
    
    if not documents:
        print("❌ No documents found to process")
        if not args.files:
            print(f"Add PDF files to the {args.docs_dir}/ directory or specify paths with --files")
        return 1
    
    print(f"🚀 Starting Ultra-Cost-Optimized Pipeline")
    print(f"Mode: {args.mode.upper()}")
    print(f"Documents to process: {len(documents)}")
    
    try:
        # Initialize and run pipeline
        runner = PipelineRunner(args.config, args.mode)
        await runner.initialize()
        
        # Process documents
        if len(documents) == 1:
            # Single document processing
            result = await runner.process_single_document(documents[0])
            results = [result]
        else:
            # Batch processing
            results = await runner.process_documents_batch(documents, args.batch_size)
        
        # Display results
        runner.display_results(results, not args.no_metrics)
        
        # Return exit code based on results
        failed_count = len([r for r in results if r.get("status") == "failed"])
        return 1 if failed_count > 0 else 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"❌ Pipeline failed: {e}")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        sys.exit(1) 