"""
Stage 2: LLM Processing with RunPod Integration
Modified version that uses RunPod models for LLM inference and embeddings while keeping local operations
"""

import os
import time
import logging
import asyncio
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import numpy as np
import yaml

# Import existing utilities and classes
from .utils.cost_tracking import CostTracker
from .utils.caching import CacheManager, cached_function
from .stage1_visual_parsing import VisualParsingResult
from .stage2_local_llm import DocumentChunk, ChunkingResult, AdaptiveChunker
from .runpod_client import RunPodModelFactory, check_runpod_config

logger = logging.getLogger(__name__)


class LocalLLMStageRunPod:
    """LLM processing stage using RunPod for model inference"""
    
    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        # Check if RunPod is configured
        if not check_runpod_config():
            raise FileNotFoundError(
                "RunPod configuration not found. Please run 'bash runpod_setup.sh' first to deploy models."
            )
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.llm_config = self.config.get('local_llm', {
            'chunking': {},
            'model': {},
            'embeddings': {},
            'caching': {}
        })
        
        # Initialize local components (chunking and caching stay local)
        self.chunker = AdaptiveChunker(self.llm_config.get('chunking', {}))
        
        # Initialize RunPod components (LLM and embeddings on RunPod)
        try:
            self.llm = RunPodModelFactory.create_llama(self.llm_config.get('model', {}))
            self.embedding_model = RunPodModelFactory.create_embeddings(self.llm_config.get('embeddings', {}))
            logger.info("✅ RunPod LLM models initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RunPod LLM models: {e}")
            raise
        
        # Initialize caching (local)
        cache_config = self.llm_config.get('caching', {})
        self.cache_manager = CacheManager({
            "memory": {"max_size": 2000, "max_memory_mb": 2048},
            "redis": cache_config,
            "default_ttl": cache_config.get('cache_ttl', 86400),
            "similarity_threshold": 0.95
        })
        
        self.cache = self.cache_manager.get_cache("general")
        self.embedding_cache = self.cache_manager.get_cache("embeddings")
        
        # Cost tracking
        self.cost_tracker = CostTracker()
        
        logger.info("Local LLM stage with RunPod initialized")
    
    def _generate_chunk_id(self, text: str, metadata: Dict = None) -> str:
        """Generate unique ID for chunk"""
        content = text + json.dumps(metadata or {}, sort_keys=True)
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    @cached_function(cache=None, ttl=86400, key_prefix="chunk")
    def _cached_chunk_processing(self, text: str, page_number: int) -> List[Dict]:
        """Cache chunk processing results"""
        chunks = self.chunker.chunk_text(text, page_number)
        return [
            {
                "id": chunk.id,
                "text": chunk.text,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "metadata": chunk.metadata
            }
            for chunk in chunks
        ]
    
    async def process_visual_results(self, visual_results: List[VisualParsingResult]) -> ChunkingResult:
        """Process visual parsing results with RunPod models"""
        logger.info(f"Processing {len(visual_results)} pages with RunPod LLM")
        start_time = time.time()
        
        all_chunks = []
        total_tokens = 0
        total_cost = 0.0
        cache_hits = 0
        cache_misses = 0
        
        # Process each page
        for page_idx, visual_result in enumerate(visual_results):
            if not visual_result.text_content:
                continue
            
            # Check cache first (local operation)
            cache_key = self.cache._generate_key("page_chunks", {
                "text": visual_result.text_content[:1000],  # First 1000 chars for key
                "page": page_idx
            })
            
            cached_chunks = self.cache.get(cache_key)
            
            if cached_chunks:
                cache_hits += 1
                # Reconstruct DocumentChunk objects
                for chunk_data in cached_chunks:
                    chunk = DocumentChunk(**chunk_data)
                    all_chunks.append(chunk)
            else:
                cache_misses += 1
                
                # Process page text into chunks (local operation)
                page_chunks = self.chunker.chunk_text(visual_result.text_content, page_idx)
                
                # Enhance chunks with RunPod LLM
                enhanced_chunks = []
                for chunk in page_chunks:
                    # For cost optimization, only enhance complex chunks
                    if len(chunk.text) > 800 or any(formula in chunk.text for formula in visual_result.latex_formulas):
                        enhanced_text, enhancement_cost = await self.llm.enhance_chunk(chunk.text)
                        chunk.text = enhanced_text
                        chunk.processing_cost += enhancement_cost
                        total_cost += enhancement_cost
                    
                    enhanced_chunks.append(chunk)
                
                all_chunks.extend(enhanced_chunks)
                
                # Cache the results (local operation)
                chunk_data_list = [
                    {
                        "id": chunk.id,
                        "text": chunk.text,
                        "page_number": chunk.page_number,
                        "chunk_index": chunk.chunk_index,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                        "metadata": chunk.metadata,
                        "processing_cost": chunk.processing_cost
                    }
                    for chunk in enhanced_chunks
                ]
                
                self.cache.set(cache_key, chunk_data_list)
        
        # Generate embeddings using RunPod
        if all_chunks:
            chunk_texts = [chunk.text for chunk in all_chunks]
            embeddings, embedding_cost = await self.embedding_model.encode_batch(chunk_texts)
            total_cost += embedding_cost
            
            # Assign embeddings to chunks
            for chunk, embedding in zip(all_chunks, embeddings):
                chunk.embedding = embedding
                
                # Cache embedding separately (local operation)
                self.embedding_cache.cache_embedding(chunk.text, embedding)
        
        # Calculate metrics
        processing_time = time.time() - start_time
        cache_hit_rate = cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0.0
        
        # Track costs
        self.cost_tracker.add_cost("llm_processing_runpod", total_cost)
        
        logger.info(f"Processed {len(all_chunks)} chunks in {processing_time:.2f}s with RunPod")
        logger.info(f"Cache hit rate: {cache_hit_rate:.2%}")
        logger.info(f"Total cost: ${total_cost:.6f}")
        
        return ChunkingResult(
            chunks=all_chunks,
            total_tokens=total_tokens,
            processing_time=processing_time,
            cost_estimate=total_cost,
            cache_hit_rate=cache_hit_rate,
            metadata={
                "pages_processed": len(visual_results),
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "runpod_enabled": True
            }
        )
    
    async def process_batch(self, visual_results_batch: List[List[VisualParsingResult]]) -> List[ChunkingResult]:
        """Process multiple documents in parallel"""
        tasks = [
            asyncio.create_task(self.process_visual_results(visual_results))
            for visual_results in visual_results_batch
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = [result for result in results if not isinstance(result, Exception)]
        
        return valid_results
    
    def optimize_chunks_for_retrieval(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Optimize chunks for better retrieval performance (local operation)"""
        logger.info(f"Optimizing {len(chunks)} chunks for retrieval")
        
        optimized_chunks = []
        
        for chunk in chunks:
            # Add retrieval-optimized metadata
            chunk.metadata = chunk.metadata or {}
            
            # Calculate chunk quality score
            quality_score = self._calculate_chunk_quality(chunk)
            chunk.confidence_score = quality_score
            
            # Add semantic keywords (local operation)
            keywords = self._extract_keywords(chunk.text)
            chunk.metadata['keywords'] = keywords
            
            # Add chunk type classification (local operation)
            chunk_type = self._classify_chunk_type(chunk.text)
            chunk.metadata['type'] = chunk_type
            
            optimized_chunks.append(chunk)
        
        return optimized_chunks
    
    def _calculate_chunk_quality(self, chunk: DocumentChunk) -> float:
        """Calculate quality score for chunk (local operation)"""
        text = chunk.text
        
        # Basic quality indicators
        factors = {
            'length': min(len(text) / 500, 1.0),  # Optimal around 500 chars
            'completeness': 1.0 if text.endswith('.') else 0.7,
            'readability': len(text.split()) / len(text) * 100,  # Words per char ratio
            'academic_indicators': sum(1 for term in ['abstract', 'conclusion', 'method', 'result'] if term.lower() in text.lower()) / 4
        }
        
        return sum(factors.values()) / len(factors)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text (local operation)"""
        import re
        
        # Remove common words and extract meaningful terms
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        
        # Simple frequency-based keyword extraction
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return top keywords
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        return [word for word, freq in keywords if freq > 1]
    
    def _classify_chunk_type(self, text: str) -> str:
        """Classify chunk type based on content patterns (local operation)"""
        text_lower = text.lower()
        
        if any(term in text_lower for term in ['abstract', 'summary']):
            return 'abstract'
        elif any(term in text_lower for term in ['introduction', 'background']):
            return 'introduction'
        elif any(term in text_lower for term in ['method', 'approach', 'algorithm']):
            return 'methodology'
        elif any(term in text_lower for term in ['result', 'finding', 'analysis']):
            return 'results'
        elif any(term in text_lower for term in ['conclusion', 'discussion', 'future']):
            return 'conclusion'
        elif any(term in text_lower for term in ['reference', 'bibliography']):
            return 'references'
        else:
            return 'content'
    
    def get_cost_report(self) -> Dict:
        """Get detailed cost breakdown"""
        cache_stats = self.cache_manager.get_comprehensive_stats()
        cost_report = self.cost_tracker.get_report()
        
        return {
            "cost_breakdown": cost_report,
            "cache_performance": cache_stats,
            "efficiency_metrics": {
                "cache_hit_rate": cache_stats.get("hybrid_cache", {}).get("hit_rate", 0),
                "cost_per_chunk": cost_report["summary"]["total_cost"] / max(1, len(self.cache.cache.cache)),
                "memory_utilization": cache_stats.get("memory_cache", {}).get("utilization", 0)
            }
        }
    
    def cleanup(self):
        """Clean up resources"""
        # Clear caches (local operation)
        self.cache_manager.clear_all_caches()
        
        logger.info("Local LLM stage with RunPod cleanup completed") 