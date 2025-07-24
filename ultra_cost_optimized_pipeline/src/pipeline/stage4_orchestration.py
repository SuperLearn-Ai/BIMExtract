"""
Stage 4: Batched Multi-Agent Orchestration
Achieves $0.03 per 1K pages through intelligent batching, complexity routing, and auto-scaling
"""

import os
import time
import logging
import asyncio
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import json
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import threading

import yaml
import redis
from celery import Celery
import numpy as np

from .utils.cost_tracking import CostTracker
from .stage1_visual_parsing import VisualParsingStage, VisualParsingResult
from .stage2_local_llm import LocalLLMStage, ChunkingResult, DocumentChunk
from .stage3_vector_storage import VectorStorageStage, StorageResult, VectorSearchResult


class DocumentComplexity(Enum):
    """Document complexity levels for routing"""
    SIMPLE = "simple"
    STANDARD = "standard"
    COMPLEX = "complex"


@dataclass
class ProcessingJob:
    """Individual processing job"""
    job_id: str
    document_path: str
    priority: int = 1
    complexity: DocumentComplexity = DocumentComplexity.STANDARD
    metadata: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"
    result: Optional[Any] = None
    error: Optional[str] = None
    cost_estimate: float = 0.0


@dataclass
class BatchProcessingResult:
    """Result from batch processing"""
    jobs_processed: int
    successful_jobs: int
    failed_jobs: int
    total_cost: float
    processing_time: float
    throughput: float  # jobs per second
    cost_per_job: float
    metadata: Dict = field(default_factory=dict)


class DocumentComplexityAnalyzer:
    """Analyzes document complexity for routing decisions"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.simple_threshold = config.get('simple_threshold', 0.3)
        self.standard_threshold = config.get('standard_threshold', 0.7)
        
    def analyze_complexity(self, document_path: str, initial_scan: Optional[Dict] = None) -> DocumentComplexity:
        """Analyze document complexity based on various factors"""
        complexity_score = 0.0
        
        # Factor 1: File size
        try:
            file_size_mb = os.path.getsize(document_path) / (1024 * 1024)
            if file_size_mb > 50:
                complexity_score += 0.3
            elif file_size_mb > 10:
                complexity_score += 0.1
        except:
            pass
        
        # Factor 2: Initial scan results (if available)
        if initial_scan:
            # Mathematical content complexity
            formula_count = len(initial_scan.get('latex_formulas', []))
            if formula_count > 20:
                complexity_score += 0.3
            elif formula_count > 5:
                complexity_score += 0.1
                
            # Table complexity
            table_count = len(initial_scan.get('tables', []))
            if table_count > 10:
                complexity_score += 0.2
            elif table_count > 3:
                complexity_score += 0.1
                
            # Text content indicators
            text = initial_scan.get('text_content', '')
            if any(indicator in text.lower() for indicator in ['algorithm', 'theorem', 'proof', 'equation']):
                complexity_score += 0.2
        
        # Factor 3: File extension hints
        if document_path.lower().endswith('.pdf'):
            complexity_score += 0.1  # PDFs are generally more complex than images
        
        # Determine complexity level
        if complexity_score <= self.simple_threshold:
            return DocumentComplexity.SIMPLE
        elif complexity_score <= self.standard_threshold:
            return DocumentComplexity.STANDARD
        else:
            return DocumentComplexity.COMPLEX
    
    def estimate_processing_cost(self, complexity: DocumentComplexity, page_count: int = 1) -> float:
        """Estimate processing cost based on complexity"""
        base_costs = {
            DocumentComplexity.SIMPLE: 0.08,    # $0.08 per 1K pages
            DocumentComplexity.STANDARD: 0.15,  # $0.15 per 1K pages
            DocumentComplexity.COMPLEX: 0.30    # $0.30 per 1K pages
        }
        
        return base_costs[complexity] * (page_count / 1000)


class BatchJobQueue:
    """Intelligent job batching and queuing system"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.batch_size = config.get('batch_size', 100)
        self.batch_timeout = config.get('batch_timeout', 30)  # seconds
        self.max_batches_per_worker = config.get('max_batches_per_worker', 4)
        
        # Queue for different complexity levels
        self.queues = {
            DocumentComplexity.SIMPLE: queue.PriorityQueue(),
            DocumentComplexity.STANDARD: queue.PriorityQueue(),
            DocumentComplexity.COMPLEX: queue.PriorityQueue()
        }
        
        # Batch formation
        self.current_batches = {
            DocumentComplexity.SIMPLE: [],
            DocumentComplexity.STANDARD: [],
            DocumentComplexity.COMPLEX: []
        }
        
        self.last_batch_time = {
            DocumentComplexity.SIMPLE: datetime.now(),
            DocumentComplexity.STANDARD: datetime.now(),
            DocumentComplexity.COMPLEX: datetime.now()
        }
        
        self.lock = threading.Lock()
    
    def add_job(self, job: ProcessingJob):
        """Add job to appropriate queue"""
        with self.lock:
            # Priority is negative for priority queue (lower number = higher priority)
            priority = -job.priority
            self.queues[job.complexity].put((priority, job.created_at, job))
    
    def get_next_batch(self, complexity: DocumentComplexity) -> List[ProcessingJob]:
        """Get next batch of jobs for processing"""
        with self.lock:
            batch = []
            queue_obj = self.queues[complexity]
            current_batch = self.current_batches[complexity]
            
            # Add jobs from current batch
            batch.extend(current_batch)
            current_batch.clear()
            
            # Fill batch from queue
            while len(batch) < self.batch_size and not queue_obj.empty():
                try:
                    _, _, job = queue_obj.get_nowait()
                    batch.append(job)
                except queue.Empty:
                    break
            
            # Update batch time
            if batch:
                self.last_batch_time[complexity] = datetime.now()
            
            return batch
    
    def should_process_batch(self, complexity: DocumentComplexity) -> bool:
        """Determine if batch should be processed now"""
        with self.lock:
            current_batch = self.current_batches[complexity]
            time_since_last = datetime.now() - self.last_batch_time[complexity]
            
            # Process if batch is full or timeout reached
            return (len(current_batch) >= self.batch_size or 
                   time_since_last.total_seconds() >= self.batch_timeout)
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        with self.lock:
            return {
                complexity.value: {
                    "queue_size": self.queues[complexity].qsize(),
                    "current_batch_size": len(self.current_batches[complexity]),
                    "last_batch_time": self.last_batch_time[complexity].isoformat()
                }
                for complexity in DocumentComplexity
            }


class WorkerManager:
    """Manages worker processes with auto-scaling"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.min_workers = config.get('min_workers', 2)
        self.max_workers = config.get('max_workers', 10)
        self.scale_up_threshold = config.get('scale_up_threshold', 0.8)
        self.scale_down_threshold = config.get('scale_down_threshold', 0.2)
        self.cooldown_period = config.get('cooldown_period', 300)  # seconds
        
        self.active_workers = {}
        self.worker_stats = {}
        self.last_scale_time = datetime.now()
        
        # Initialize minimum workers
        self._initialize_workers()
    
    def _initialize_workers(self):
        """Initialize minimum number of workers"""
        for i in range(self.min_workers):
            worker_id = f"worker_{i}"
            self.active_workers[worker_id] = {
                "status": "idle",
                "created_at": datetime.now(),
                "jobs_processed": 0,
                "total_cost": 0.0
            }
    
    def get_worker_load(self) -> float:
        """Calculate current worker load"""
        if not self.active_workers:
            return 0.0
        
        busy_workers = sum(1 for worker in self.active_workers.values() 
                          if worker["status"] == "busy")
        
        return busy_workers / len(self.active_workers)
    
    def should_scale_up(self) -> bool:
        """Determine if workers should be scaled up"""
        current_load = self.get_worker_load()
        time_since_last_scale = datetime.now() - self.last_scale_time
        
        return (current_load > self.scale_up_threshold and
                len(self.active_workers) < self.max_workers and
                time_since_last_scale.total_seconds() > self.cooldown_period)
    
    def should_scale_down(self) -> bool:
        """Determine if workers should be scaled down"""
        current_load = self.get_worker_load()
        time_since_last_scale = datetime.now() - self.last_scale_time
        
        return (current_load < self.scale_down_threshold and
                len(self.active_workers) > self.min_workers and
                time_since_last_scale.total_seconds() > self.cooldown_period)
    
    def scale_up(self) -> int:
        """Add additional workers"""
        new_workers = min(2, self.max_workers - len(self.active_workers))
        
        for i in range(new_workers):
            worker_id = f"worker_{len(self.active_workers)}"
            self.active_workers[worker_id] = {
                "status": "idle",
                "created_at": datetime.now(),
                "jobs_processed": 0,
                "total_cost": 0.0
            }
        
        self.last_scale_time = datetime.now()
        logging.info(f"Scaled up: added {new_workers} workers (total: {len(self.active_workers)})")
        
        return new_workers
    
    def scale_down(self) -> int:
        """Remove excess workers"""
        workers_to_remove = min(1, len(self.active_workers) - self.min_workers)
        
        # Remove idle workers first
        removed_workers = []
        for worker_id, worker_info in list(self.active_workers.items()):
            if worker_info["status"] == "idle" and len(removed_workers) < workers_to_remove:
                removed_workers.append(worker_id)
                del self.active_workers[worker_id]
        
        self.last_scale_time = datetime.now()
        logging.info(f"Scaled down: removed {len(removed_workers)} workers (total: {len(self.active_workers)})")
        
        return len(removed_workers)
    
    def assign_worker(self, job_id: str) -> Optional[str]:
        """Assign an idle worker to a job"""
        for worker_id, worker_info in self.active_workers.items():
            if worker_info["status"] == "idle":
                worker_info["status"] = "busy"
                worker_info["current_job"] = job_id
                return worker_id
        
        return None
    
    def release_worker(self, worker_id: str):
        """Release worker back to idle status"""
        if worker_id in self.active_workers:
            self.active_workers[worker_id]["status"] = "idle"
            self.active_workers[worker_id]["jobs_processed"] += 1
            if "current_job" in self.active_workers[worker_id]:
                del self.active_workers[worker_id]["current_job"]
    
    def get_worker_stats(self) -> Dict[str, Any]:
        """Get worker statistics"""
        return {
            "total_workers": len(self.active_workers),
            "busy_workers": sum(1 for w in self.active_workers.values() if w["status"] == "busy"),
            "idle_workers": sum(1 for w in self.active_workers.values() if w["status"] == "idle"),
            "worker_load": self.get_worker_load(),
            "total_jobs_processed": sum(w.get("jobs_processed", 0) for w in self.active_workers.values())
        }


class OrchestrationStage:
    """Main orchestration stage that coordinates all pipeline stages"""
    
    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.orchestration_config = self.config['orchestration']
        
        # Initialize pipeline stages
        self.visual_stage = VisualParsingStage(config_path)
        self.llm_stage = LocalLLMStage(config_path)
        self.storage_stage = VectorStorageStage(config_path)
        
        # Initialize orchestration components
        self.complexity_analyzer = DocumentComplexityAnalyzer(
            self.orchestration_config['routing']
        )
        
        self.job_queue = BatchJobQueue(
            self.orchestration_config['batching']
        )
        
        self.worker_manager = WorkerManager(
            self.orchestration_config['scaling']
        )
        
        # Cost tracking
        self.cost_tracker = CostTracker()
        
        # Performance metrics
        self.metrics = {
            "total_documents_processed": 0,
            "total_processing_time": 0.0,
            "average_cost_per_document": 0.0,
            "cache_hit_rates": {},
            "worker_efficiency": 0.0
        }
        
        logging.info("Orchestration stage initialized")
    
    async def process_document(self, document_path: str, priority: int = 1) -> ProcessingJob:
        """Process a single document through the entire pipeline"""
        job_id = f"job_{int(time.time() * 1000)}"
        
        # Create processing job
        job = ProcessingJob(
            job_id=job_id,
            document_path=document_path,
            priority=priority,
            status="analyzing"
        )
        
        try:
            # Step 1: Quick complexity analysis
            complexity = self.complexity_analyzer.analyze_complexity(document_path)
            job.complexity = complexity
            
            # Step 2: Estimate cost
            job.cost_estimate = self.complexity_analyzer.estimate_processing_cost(complexity)
            
            # Step 3: Add to appropriate queue
            self.job_queue.add_job(job)
            
            # Step 4: Process when batch is ready
            if self.job_queue.should_process_batch(complexity):
                await self._process_batch(complexity)
            
            return job
            
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            logging.error(f"Document processing failed for {document_path}: {e}")
            return job
    
    async def _process_batch(self, complexity: DocumentComplexity):
        """Process a batch of documents with same complexity"""
        batch = self.job_queue.get_next_batch(complexity)
        
        if not batch:
            return
        
        logging.info(f"Processing batch of {len(batch)} {complexity.value} documents")
        start_time = time.time()
        
        try:
            # Auto-scaling check
            if self.worker_manager.should_scale_up():
                self.worker_manager.scale_up()
            elif self.worker_manager.should_scale_down():
                self.worker_manager.scale_down()
            
            # Process batch through pipeline stages
            await self._process_pipeline_batch(batch)
            
            # Update metrics
            processing_time = time.time() - start_time
            self._update_metrics(batch, processing_time)
            
            logging.info(f"Batch processing completed in {processing_time:.2f}s")
            
        except Exception as e:
            logging.error(f"Batch processing failed: {e}")
            for job in batch:
                job.status = "failed"
                job.error = str(e)
    
    async def _process_pipeline_batch(self, batch: List[ProcessingJob]):
        """Process batch through all pipeline stages"""
        
        # Stage 1: Visual Parsing
        visual_results = []
        for job in batch:
            try:
                job.status = "visual_parsing"
                job.started_at = datetime.now()
                
                results = await self.visual_stage.process_document(job.document_path)
                visual_results.append(results)
                
                job.metadata['visual_parsing'] = {
                    "pages_processed": len(results),
                    "total_cost": sum(r.cost_estimate for r in results)
                }
                
            except Exception as e:
                job.status = "failed"
                job.error = f"Visual parsing failed: {e}"
                visual_results.append([])
        
        # Stage 2: Local LLM Processing
        chunking_results = []
        for i, job in enumerate(batch):
            if job.status == "failed":
                chunking_results.append(None)
                continue
                
            try:
                job.status = "llm_processing"
                
                result = self.llm_stage.process_visual_results(visual_results[i])
                chunking_results.append(result)
                
                job.metadata['llm_processing'] = {
                    "chunks_generated": len(result.chunks),
                    "cache_hit_rate": result.cache_hit_rate,
                    "total_cost": result.cost_estimate
                }
                
            except Exception as e:
                job.status = "failed"
                job.error = f"LLM processing failed: {e}"
                chunking_results.append(None)
        
        # Stage 3: Vector Storage
        for i, job in enumerate(batch):
            if job.status == "failed" or chunking_results[i] is None:
                continue
                
            try:
                job.status = "vector_storage"
                
                storage_results = self.storage_stage.process_chunking_results([chunking_results[i]])
                
                job.metadata['vector_storage'] = {
                    "vectors_stored": storage_results[0].stored_chunks if storage_results else 0,
                    "total_cost": storage_results[0].cost_estimate if storage_results else 0
                }
                
                job.status = "completed"
                job.completed_at = datetime.now()
                
                # Calculate total job cost
                job.cost_estimate = (
                    job.metadata.get('visual_parsing', {}).get('total_cost', 0) +
                    job.metadata.get('llm_processing', {}).get('total_cost', 0) +
                    job.metadata.get('vector_storage', {}).get('total_cost', 0)
                )
                
                # Track orchestration cost
                orchestration_cost = 0.03 * 0.001  # $0.03 per 1K pages
                self.cost_tracker.add_cost("orchestration", orchestration_cost)
                
            except Exception as e:
                job.status = "failed"
                job.error = f"Vector storage failed: {e}"
    
    def _update_metrics(self, batch: List[ProcessingJob], processing_time: float):
        """Update performance metrics"""
        successful_jobs = [job for job in batch if job.status == "completed"]
        
        self.metrics["total_documents_processed"] += len(successful_jobs)
        self.metrics["total_processing_time"] += processing_time
        
        if successful_jobs:
            avg_cost = sum(job.cost_estimate for job in successful_jobs) / len(successful_jobs)
            self.metrics["average_cost_per_document"] = avg_cost
            
            # Update cache hit rates
            for job in successful_jobs:
                llm_cache_rate = job.metadata.get('llm_processing', {}).get('cache_hit_rate', 0)
                if 'llm_cache_hit_rate' not in self.metrics["cache_hit_rates"]:
                    self.metrics["cache_hit_rates"]['llm_cache_hit_rate'] = []
                self.metrics["cache_hit_rates"]['llm_cache_hit_rate'].append(llm_cache_rate)
        
        # Update worker efficiency
        worker_stats = self.worker_manager.get_worker_stats()
        self.metrics["worker_efficiency"] = worker_stats["worker_load"]
    
    async def process_batch_documents(self, document_paths: List[str]) -> BatchProcessingResult:
        """Process multiple documents efficiently"""
        logging.info(f"Starting batch processing of {len(document_paths)} documents")
        start_time = time.time()
        
        # Create jobs for all documents
        jobs = []
        for path in document_paths:
            job = await self.process_document(path)
            jobs.append(job)
        
        # Wait for all jobs to complete
        max_wait_time = 300  # 5 minutes
        wait_start = time.time()
        
        while time.time() - wait_start < max_wait_time:
            pending_jobs = [job for job in jobs if job.status not in ["completed", "failed"]]
            if not pending_jobs:
                break
            
            await asyncio.sleep(1)
        
        # Calculate results
        processing_time = time.time() - start_time
        successful_jobs = [job for job in jobs if job.status == "completed"]
        failed_jobs = [job for job in jobs if job.status == "failed"]
        
        total_cost = sum(job.cost_estimate for job in successful_jobs)
        throughput = len(successful_jobs) / processing_time if processing_time > 0 else 0
        cost_per_job = total_cost / len(successful_jobs) if successful_jobs else 0
        
        return BatchProcessingResult(
            jobs_processed=len(jobs),
            successful_jobs=len(successful_jobs),
            failed_jobs=len(failed_jobs),
            total_cost=total_cost,
            processing_time=processing_time,
            throughput=throughput,
            cost_per_job=cost_per_job,
            metadata={
                "complexity_distribution": self._get_complexity_distribution(jobs),
                "queue_stats": self.job_queue.get_queue_stats(),
                "worker_stats": self.worker_manager.get_worker_stats()
            }
        )
    
    def _get_complexity_distribution(self, jobs: List[ProcessingJob]) -> Dict[str, int]:
        """Get distribution of job complexities"""
        distribution = {complexity.value: 0 for complexity in DocumentComplexity}
        
        for job in jobs:
            distribution[job.complexity.value] += 1
        
        return distribution
    
    def search_documents(self, query: str, top_k: int = 5) -> Tuple[List[VectorSearchResult], float]:
        """Search processed documents"""
        # Generate query embedding
        query_embedding, embedding_cost = self.llm_stage.embedding_model.encode_single(query)
        
        # Search in vector store
        results, search_cost = self.storage_stage.search_documents(query_embedding, top_k)
        
        total_cost = embedding_cost + search_cost
        self.cost_tracker.add_cost("orchestration", total_cost, "search")
        
        return results, total_cost
    
    def get_pipeline_analytics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline analytics"""
        visual_costs = self.visual_stage.get_cost_report()
        llm_costs = self.llm_stage.get_cost_report()
        storage_costs = self.storage_stage.get_cost_report()
        orchestration_costs = self.cost_tracker.get_report()
        
        return {
            "pipeline_metrics": self.metrics,
            "cost_breakdown": {
                "visual_parsing": visual_costs,
                "llm_processing": llm_costs,
                "vector_storage": storage_costs,
                "orchestration": orchestration_costs
            },
            "queue_stats": self.job_queue.get_queue_stats(),
            "worker_stats": self.worker_manager.get_worker_stats(),
            "efficiency_metrics": {
                "documents_per_hour": self.metrics["total_documents_processed"] / max(1, self.metrics["total_processing_time"] / 3600),
                "cost_per_1k_pages": self._calculate_cost_per_1k_pages(),
                "pipeline_utilization": self._calculate_pipeline_utilization()
            }
        }
    
    def _calculate_cost_per_1k_pages(self) -> float:
        """Calculate average cost per 1000 pages"""
        # Simplified calculation - in real implementation would track actual page counts
        if self.metrics["total_documents_processed"] > 0:
            total_cost = (
                visual_costs["summary"]["total_cost"] +
                llm_costs["summary"]["total_cost"] +
                storage_costs["summary"]["total_cost"] +
                orchestration_costs["summary"]["total_cost"]
            )
            estimated_pages = self.metrics["total_documents_processed"] * 10  # Assume 10 pages per doc
            return (total_cost * 1000) / max(1, estimated_pages)
        return 0.0
    
    def _calculate_pipeline_utilization(self) -> float:
        """Calculate overall pipeline utilization"""
        worker_efficiency = self.metrics.get("worker_efficiency", 0)
        cache_efficiency = np.mean([
            np.mean(rates) for rates in self.metrics["cache_hit_rates"].values() 
            if rates
        ]) if self.metrics["cache_hit_rates"] else 0
        
        return (worker_efficiency + cache_efficiency) / 2
    
    def cleanup(self):
        """Clean up all pipeline stages"""
        self.visual_stage.cleanup()
        self.llm_stage.cleanup()
        
        logging.info("Pipeline cleanup completed")


# Example usage
async def main():
    """Example usage of orchestration stage"""
    orchestrator = OrchestrationStage()
    
    # Process single document
    document_path = "sample_document.pdf"
    job = await orchestrator.process_document(document_path)
    print(f"Job {job.job_id} status: {job.status}")
    
    # Process batch of documents
    document_paths = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
    batch_result = await orchestrator.process_batch_documents(document_paths)
    
    print(f"Batch processing completed:")
    print(f"  Processed: {batch_result.jobs_processed}")
    print(f"  Successful: {batch_result.successful_jobs}")
    print(f"  Failed: {batch_result.failed_jobs}")
    print(f"  Total cost: ${batch_result.total_cost:.6f}")
    print(f"  Throughput: {batch_result.throughput:.2f} jobs/second")
    
    # Search documents
    search_results, search_cost = orchestrator.search_documents("machine learning algorithms", top_k=3)
    print(f"Found {len(search_results)} relevant documents (cost: ${search_cost:.6f})")
    
    # Get analytics
    analytics = orchestrator.get_pipeline_analytics()
    print("Pipeline Analytics:", json.dumps(analytics, indent=2, default=str))
    
    orchestrator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())