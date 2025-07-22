#!/usr/bin/env python3
"""
Run Ultra-Cost-Optimized Pipeline with RunPod Models
Simple script to run the pipeline using RunPod for all AI models
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

# Import RunPod-enabled stages
try:
    from stage1_visual_parsing_runpod import VisualParsingStageRunPod
    from stage2_local_llm_runpod import LocalLLMStageRunPod
    from stage3_vector_storage import VectorStorageStage  # This stays local
    from runpod_client import check_runpod_config, load_runpod_config
except ImportError as e:
    print(f"❌ Error importing RunPod pipeline: {e}")
    print("Make sure you've deployed models to RunPod first: bash runpod_setup.sh")
    sys.exit(1)


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pipeline_runpod.log'),
            logging.StreamHandler()
        ]
    )


class RunPodPipeline:
    """Pipeline orchestrator using RunPod models"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Check if RunPod is configured
        if not check_runpod_config():
            raise FileNotFoundError(
                "RunPod configuration not found. Please run 'bash runpod_setup.sh' first."
            )
        
        # Load RunPod configuration
        self.runpod_config = load_runpod_config()
        
        # Initialize stages
        self.visual_stage = VisualParsingStageRunPod(config_path)
        self.llm_stage = LocalLLMStageRunPod(config_path)
        self.storage_stage = VectorStorageStage(config_path)  # Stays local
        
        logging.info("✅ RunPod pipeline initialized successfully")
    
    async def process_document(self, document_path: str) -> dict:
        """Process a single document through the RunPod pipeline"""
        logging.info(f"🚀 Processing document: {os.path.basename(document_path)}")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Stage 1: Visual parsing with RunPod
            visual_results = await self.visual_stage.process_document(document_path)
            
            if not visual_results:
                return {
                    "status": "failed",
                    "error": "Visual parsing failed",
                    "document": document_path
                }
            
            # Stage 2: LLM processing with RunPod
            chunking_result = await self.llm_stage.process_visual_results(visual_results)
            
            # Stage 3: Vector storage (local)
            storage_results = self.storage_stage.process_chunking_results([chunking_result])
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            # Calculate total cost
            total_cost = (
                sum(r.cost_estimate for r in visual_results) +
                chunking_result.cost_estimate +
                (storage_results[0].cost_estimate if storage_results else 0)
            )
            
            result = {
                "status": "completed",
                "document": document_path,
                "processing_time": processing_time,
                "total_cost": total_cost,
                "pages_processed": len(visual_results),
                "chunks_generated": len(chunking_result.chunks),
                "vectors_stored": storage_results[0].stored_chunks if storage_results else 0,
                "cache_hit_rate": chunking_result.cache_hit_rate
            }
            
            logging.info(f"✅ Document processed successfully in {processing_time:.2f}s")
            logging.info(f"💰 Total cost: ${total_cost:.6f}")
            
            return result
            
        except Exception as e:
            logging.error(f"❌ Document processing failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "document": document_path
            }
    
    async def process_documents(self, document_paths: List[str]) -> List[dict]:
        """Process multiple documents"""
        logging.info(f"📚 Processing {len(document_paths)} documents with RunPod")
        
        results = []
        for doc_path in document_paths:
            result = await self.process_document(doc_path)
            results.append(result)
        
        return results
    
    def cleanup(self):
        """Clean up pipeline resources"""
        self.visual_stage.cleanup()
        self.llm_stage.cleanup()
        logging.info("🧹 Pipeline cleanup completed")


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
        print("Add PDF files to this directory and run again")
        return []
    
    # Find PDF files
    pdf_files = list(documents_dir.glob("*.pdf"))
    return [str(f) for f in pdf_files]


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run Ultra-Cost-Optimized Pipeline with RunPod")
    parser.add_argument("--files", nargs="+", help="Specific files to process")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    print("🚀 Ultra-Cost-Optimized Pipeline with RunPod")
    print("=" * 50)
    
    try:
        # Check RunPod configuration
        if not check_runpod_config():
            print("❌ RunPod not configured!")
            print("Please run: bash runpod_setup.sh")
            return
        
        # Load RunPod info
        runpod_config = load_runpod_config()
        endpoints = runpod_config["runpod"]["endpoints"]
        print(f"✅ RunPod configured with {len(endpoints)} model endpoints")
        
        # Find documents
        documents = find_documents(args.files)
        
        if not documents:
            print("❌ No documents found to process")
            print("Add PDF files to the documents/ directory or use --files")
            return
        
        print(f"📄 Found {len(documents)} documents to process")
        
        # Initialize pipeline
        pipeline = RunPodPipeline(args.config)
        
        # Process documents
        results = await pipeline.process_documents(documents)
        
        # Display results
        print("\n" + "=" * 60)
        print("📊 PROCESSING RESULTS")
        print("=" * 60)
        
        successful = [r for r in results if r["status"] == "completed"]
        failed = [r for r in results if r["status"] == "failed"]
        
        print(f"✅ Successful: {len(successful)}")
        print(f"❌ Failed: {len(failed)}")
        
        if successful:
            total_cost = sum(r["total_cost"] for r in successful)
            total_time = sum(r["processing_time"] for r in successful)
            total_pages = sum(r["pages_processed"] for r in successful)
            
            print(f"\n💰 Cost Analysis:")
            print(f"   Total cost: ${total_cost:.6f}")
            print(f"   Cost per document: ${total_cost/len(successful):.6f}")
            print(f"   Cost per page: ${total_cost/max(1, total_pages):.6f}")
            print(f"   Cost per 1K pages: ${(total_cost * 1000)/max(1, total_pages):.6f}")
            
            print(f"\n⚡ Performance:")
            print(f"   Total time: {total_time:.2f}s")
            print(f"   Average time per document: {total_time/len(successful):.2f}s")
            print(f"   Pages per hour: {(total_pages * 3600)/max(1, total_time):.0f}")
            
            avg_cache_hit = sum(r.get("cache_hit_rate", 0) for r in successful) / len(successful)
            print(f"   Cache hit rate: {avg_cache_hit:.1%}")
        
        if failed:
            print(f"\n❌ Failed documents:")
            for result in failed:
                print(f"   {os.path.basename(result['document'])}: {result['error']}")
        
        # Cleanup
        pipeline.cleanup()
        
        print(f"\n🎉 Pipeline completed!")
        print(f"📝 Logs saved to: pipeline_runpod.log")
        
    except Exception as e:
        logging.error(f"Pipeline error: {e}")
        print(f"\n💥 Pipeline failed: {e}")


if __name__ == "__main__":
    print("Initializing RunPod pipeline...")
    asyncio.run(main()) 