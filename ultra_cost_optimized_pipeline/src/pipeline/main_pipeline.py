"""
Ultra-Cost-Optimized Document Extraction Pipeline
Main orchestrator integrating all stages with Qwen3-30B-A3B Stage 2
"""

import os
import sys
import time
import logging
import asyncio
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

import yaml

# Add the current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import all pipeline stages
try:
    from stage1_visual_parsing import Stage1VisualParser, VisualParsingResult
    from stage2_qwen3_optimized import Qwen3OptimizedChunker, ChunkingResult
    from stage3_vector_storage import VectorStorageProcessor, StorageResult
    from stage4_orchestration import OrchestrationProcessor
    from utils.cost_tracking import CostTracker
except ImportError as e:
    logging.error(f"Failed to import pipeline components: {e}")
    sys.exit(1)


@dataclass
class PipelineResult:
    """Complete pipeline processing result"""
    document_path: str
    stage1_result: VisualParsingResult
    stage2_result: ChunkingResult
    stage3_result: StorageResult
    total_processing_time: float
    total_cost: float
    success: bool
    error_message: Optional[str] = None


class UltraOptimizedPipeline:
    """Main pipeline orchestrator with integrated Qwen3 Stage 2"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or "ultra_cost_optimized_pipeline/config/pipeline_config.yaml"
        self.stage2_config_path = "ultra_cost_optimized_pipeline/config/stage2_qwen3_config.yaml"
        
        self.setup_logging()
        self.load_config()
        self.initialize_stages()
        
        # Performance tracking
        self.cost_tracker = CostTracker("main_pipeline")
        self.processed_documents = 0
        self.total_costs = []
        self.processing_times = []
        
        logging.info("Ultra-Optimized Pipeline initialized successfully")
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('pipeline.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('UltraOptimizedPipeline')
    
    def load_config(self):
        """Load pipeline configuration"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            # Load Stage 2 specific config
            with open(self.stage2_config_path, 'r') as f:
                self.stage2_config = yaml.safe_load(f)
                
            self.logger.info("Configuration loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise
    
    def initialize_stages(self):
        """Initialize all pipeline stages"""
        try:
            # Stage 1: Visual Parsing
            self.stage1 = Stage1VisualParser(self.config['visual_parsing'])
            
            # Stage 2: Qwen3 Optimized Chunking
            self.stage2 = Qwen3OptimizedChunker(self.stage2_config_path)
            
            # Stage 3: Vector Storage  
            self.stage3 = VectorStorageProcessor(self.config)
            
            # Stage 4: Orchestration
            self.stage4 = OrchestrationProcessor(self.config)
            
            self.logger.info("All pipeline stages initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize stages: {e}")
            raise
    
    async def process_document(self, document_path: str) -> PipelineResult:
        """Process a single document through the complete pipeline"""
        start_time = time.time()
        operation_id = self.cost_tracker.start_operation("process_document", {
            "document_path": document_path
        })
        
        try:
            self.logger.info(f"Processing document: {document_path}")
            
            # Validate document
            if not os.path.exists(document_path):
                raise FileNotFoundError(f"Document not found: {document_path}")
            
            # Stage 1: Visual Parsing
            self.logger.info("Starting Stage 1: Visual Parsing")
            stage1_start = time.time()
            stage1_result = await self.stage1.process_pdf(document_path)
            stage1_time = time.time() - stage1_start
            self.logger.info(f"Stage 1 completed in {stage1_time:.2f}s, cost: ${stage1_result.cost_estimate:.4f}")
            
            # Stage 2: Qwen3 Optimized Chunking
            self.logger.info("Starting Stage 2: Qwen3 Chunking")
            stage2_start = time.time()
            stage2_result = await self.stage2.process_document(stage1_result)
            stage2_time = time.time() - stage2_start
            self.logger.info(f"Stage 2 completed in {stage2_time:.2f}s, cost: ${stage2_result.cost_estimate:.4f}")
            
            # Stage 3: Vector Storage
            self.logger.info("Starting Stage 3: Vector Storage")
            stage3_start = time.time()
            stage3_result = await self.stage3.store_chunks(stage2_result)
            stage3_time = time.time() - stage3_start
            self.logger.info(f"Stage 3 completed in {stage3_time:.2f}s, cost: ${stage3_result.cost_estimate:.4f}")
            
            # Calculate total metrics
            total_time = time.time() - start_time
            total_cost = (stage1_result.cost_estimate + 
                         stage2_result.cost_estimate + 
                         stage3_result.cost_estimate)
            
            self.cost_tracker.end_operation(operation_id, total_cost)
            
            # Update tracking
            self.processed_documents += 1
            self.total_costs.append(total_cost)
            self.processing_times.append(total_time)
            
            result = PipelineResult(
                document_path=document_path,
                stage1_result=stage1_result,
                stage2_result=stage2_result,
                stage3_result=stage3_result,
                total_processing_time=total_time,
                total_cost=total_cost,
                success=True
            )
            
            self.logger.info(f"Document processed successfully: {document_path}")
            self.logger.info(f"Total time: {total_time:.2f}s, Total cost: ${total_cost:.4f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing document {document_path}: {e}")
            self.cost_tracker.end_operation(operation_id, 0.0)
            
            return PipelineResult(
                document_path=document_path,
                stage1_result=None,
                stage2_result=None,
                stage3_result=None,
                total_processing_time=time.time() - start_time,
                total_cost=0.0,
                success=False,
                error_message=str(e)
            )
    
    async def process_documents(self, document_paths: List[str]) -> List[PipelineResult]:
        """Process multiple documents in parallel"""
        self.logger.info(f"Processing {len(document_paths)} documents")
        
        # Process documents concurrently (with limit to avoid memory issues)
        max_concurrent = self.config.get('performance', {}).get('max_workers', 2)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(doc_path):
            async with semaphore:
                return await self.process_document(doc_path)
        
        # Execute all document processing tasks
        tasks = [process_with_semaphore(doc_path) for doc_path in document_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Exception processing {document_paths[i]}: {result}")
                processed_results.append(PipelineResult(
                    document_path=document_paths[i],
                    stage1_result=None,
                    stage2_result=None, 
                    stage3_result=None,
                    total_processing_time=0.0,
                    total_cost=0.0,
                    success=False,
                    error_message=str(result)
                ))
            else:
                processed_results.append(result)
        
        self.logger.info(f"Batch processing complete. Success rate: "
                        f"{sum(1 for r in processed_results if r.success)}/{len(processed_results)}")
        
        return processed_results
    
    def get_performance_metrics(self) -> Dict:
        """Get comprehensive performance metrics"""
        if not self.total_costs:
            return {"message": "No documents processed yet"}
        
        avg_cost = sum(self.total_costs) / len(self.total_costs)
        avg_time = sum(self.processing_times) / len(self.processing_times)
        cost_per_1k_pages = avg_cost * 1000  # Assuming 1 doc ≈ 1 page for now
        
        # Get Stage 2 specific metrics
        stage2_metrics = self.stage2.get_metrics()
        
        return {
            "documents_processed": self.processed_documents,
            "average_cost_per_document": avg_cost,
            "average_processing_time": avg_time,
            "cost_per_1k_pages": cost_per_1k_pages,
            "total_pipeline_cost": sum(self.total_costs),
            "stage2_metrics": stage2_metrics,
            "target_achievement": {
                "cost_target": 0.15,  # $0.15 per 1K pages total pipeline
                "current_cost": cost_per_1k_pages,
                "target_met": cost_per_1k_pages <= 0.15
            }
        }
    
    def create_test_documents(self):
        """Create sample test documents if they don't exist"""
        documents_dir = Path("documents")
        documents_dir.mkdir(exist_ok=True)
        
        sample_files = ["sample.pdf", "sample2.pdf", "sample3.pdf"]
        
        for sample_file in sample_files:
            sample_path = documents_dir / sample_file
            if not sample_path.exists():
                self.logger.warning(f"Test document not found: {sample_path}")
                self.logger.info(f"Please add {sample_file} to the documents/ directory for testing")
        
        return [str(documents_dir / f) for f in sample_files if (documents_dir / f).exists()]


async def main():
    """Main execution function for testing"""
    # Initialize pipeline
    pipeline = UltraOptimizedPipeline()
    
    # Check for test documents
    test_documents = pipeline.create_test_documents()
    
    if not test_documents:
        print("No test documents found. Please add sample.pdf, sample2.pdf, and sample3.pdf to the documents/ directory.")
        return
    
    print(f"Found {len(test_documents)} test documents: {test_documents}")
    
    # Process test documents
    print("\n🚀 Starting pipeline processing...")
    results = await pipeline.process_documents(test_documents)
    
    # Display results
    print("\n📊 Processing Results:")
    print("=" * 60)
    
    for result in results:
        status = "✅ SUCCESS" if result.success else "❌ FAILED"
        print(f"{status}: {os.path.basename(result.document_path)}")
        
        if result.success:
            print(f"  ⏱️  Processing time: {result.total_processing_time:.2f}s")
            print(f"  💰 Total cost: ${result.total_cost:.4f}")
            print(f"  📄 Chunks created: {len(result.stage2_result.chunks)}")
            print(f"  🎯 Cache hit rate: {result.stage2_result.cache_hit_rate:.2f}")
        else:
            print(f"  ❌ Error: {result.error_message}")
        print()
    
    # Display overall performance metrics
    metrics = pipeline.get_performance_metrics()
    print("📈 Overall Performance Metrics:")
    print("=" * 60)
    print(f"Documents processed: {metrics['documents_processed']}")
    print(f"Average cost per document: ${metrics['average_cost_per_document']:.4f}")
    print(f"Average processing time: {metrics['average_processing_time']:.2f}s")
    print(f"Cost per 1K pages: ${metrics['cost_per_1k_pages']:.4f}")
    print(f"Target cost: ${metrics['target_achievement']['cost_target']:.2f}")
    print(f"Target met: {'✅' if metrics['target_achievement']['target_met'] else '❌'}")
    
    if 'stage2_metrics' in metrics:
        print(f"\n🔍 Stage 2 (Qwen3) Specific Metrics:")
        stage2 = metrics['stage2_metrics']
        print(f"Cache hit rate: {stage2.get('cache_hit_rate', 0):.2%}")
        print(f"Total chunks created: {stage2.get('total_chunks_created', 0)}")
        print(f"Stage 2 cost per document: ${stage2.get('cost_per_document', 0):.4f}")


if __name__ == "__main__":
    # Run the pipeline
    asyncio.run(main()) 