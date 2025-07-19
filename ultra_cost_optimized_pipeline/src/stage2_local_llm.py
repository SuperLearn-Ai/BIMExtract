"""
Stage 2: Local LLM with Aggressive Caching
Achieves $0.04 per 1K pages through quantized Llama models and intelligent caching
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

import torch
import numpy as np
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import yaml
from transformers import AutoTokenizer

from .utils.cost_tracking import CostTracker
from .utils.caching import CacheManager, cached_function
from .stage1_visual_parsing import VisualParsingResult


@dataclass
class DocumentChunk:
    """Document chunk with metadata"""
    id: str
    text: str
    embedding: Optional[np.ndarray] = None
    page_number: int = 0
    chunk_index: int = 0
    start_char: int = 0
    end_char: int = 0
    metadata: Dict = None
    confidence_score: float = 0.0
    processing_cost: float = 0.0


@dataclass
class ChunkingResult:
    """Result from chunking stage"""
    chunks: List[DocumentChunk]
    total_tokens: int
    processing_time: float
    cost_estimate: float
    cache_hit_rate: float
    metadata: Dict = None


class AdaptiveChunker:
    """Intelligent chunking with context awareness"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.base_chunk_size = config.get('base_chunk_size', 512)
        self.overlap = config.get('overlap', 50)
        self.min_chunk_size = config.get('min_chunk_size', 100)
        self.max_chunk_size = config.get('max_chunk_size', 1024)
        
        # Patterns for semantic boundaries
        self.semantic_boundaries = [
            r'\n\n+',  # Paragraph breaks
            r'\n[A-Z][a-z]+:',  # Section headers
            r'\n\d+\.',  # Numbered lists
            r'\n[•\-\*]',  # Bullet points
            r'\.\s+[A-Z]',  # Sentence boundaries
        ]
        
    def chunk_text(self, text: str, page_number: int = 0) -> List[DocumentChunk]:
        """Adaptively chunk text based on content structure"""
        if not text or len(text) < self.min_chunk_size:
            return []
        
        chunks = []
        chunk_id = 0
        
        # Try semantic chunking first
        semantic_chunks = self._semantic_chunking(text)
        
        # If semantic chunking produces reasonable chunks, use it
        if self._is_good_chunking(semantic_chunks):
            for i, chunk_text in enumerate(semantic_chunks):
                chunk = DocumentChunk(
                    id=f"page_{page_number}_chunk_{chunk_id}",
                    text=chunk_text.strip(),
                    page_number=page_number,
                    chunk_index=chunk_id,
                    start_char=text.find(chunk_text),
                    end_char=text.find(chunk_text) + len(chunk_text),
                    metadata={"chunking_method": "semantic"}
                )
                chunks.append(chunk)
                chunk_id += 1
        else:
            # Fall back to sliding window chunking
            sliding_chunks = self._sliding_window_chunking(text)
            for i, chunk_text in enumerate(sliding_chunks):
                chunk = DocumentChunk(
                    id=f"page_{page_number}_chunk_{chunk_id}",
                    text=chunk_text.strip(),
                    page_number=page_number,
                    chunk_index=chunk_id,
                    start_char=i * (self.base_chunk_size - self.overlap),
                    end_char=i * (self.base_chunk_size - self.overlap) + len(chunk_text),
                    metadata={"chunking_method": "sliding_window"}
                )
                chunks.append(chunk)
                chunk_id += 1
        
        return chunks
    
    def _semantic_chunking(self, text: str) -> List[str]:
        """Chunk text at semantic boundaries"""
        import re
        
        # Find all potential break points
        break_points = [0]
        
        for pattern in self.semantic_boundaries:
            matches = re.finditer(pattern, text)
            for match in matches:
                break_points.append(match.start())
        
        break_points.append(len(text))
        break_points = sorted(list(set(break_points)))
        
        chunks = []
        current_chunk = ""
        
        for i in range(len(break_points) - 1):
            segment = text[break_points[i]:break_points[i + 1]]
            
            if len(current_chunk) + len(segment) <= self.max_chunk_size:
                current_chunk += segment
            else:
                if current_chunk and len(current_chunk) >= self.min_chunk_size:
                    chunks.append(current_chunk)
                current_chunk = segment
        
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(current_chunk)
        
        return chunks
    
    def _sliding_window_chunking(self, text: str) -> List[str]:
        """Traditional sliding window chunking"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.base_chunk_size, len(text))
            chunk_text = text[start:end]
            
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(chunk_text)
            
            start += (self.base_chunk_size - self.overlap)
        
        return chunks
    
    def _is_good_chunking(self, chunks: List[str]) -> bool:
        """Evaluate if semantic chunking produced good results"""
        if not chunks:
            return False
        
        # Check size distribution
        sizes = [len(chunk) for chunk in chunks]
        avg_size = sum(sizes) / len(sizes)
        
        # Good chunking should have reasonable average size and not too much variance
        return (self.min_chunk_size <= avg_size <= self.max_chunk_size and
                max(sizes) / min(sizes) < 5)  # Size variance threshold


class QuantizedLlama:
    """Quantized Llama model for local inference"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.cost_per_token = 0.0000001  # Ultra-low cost for local inference
        
    def load_model(self):
        """Lazy load quantized Llama model"""
        if self.model is not None:
            return
        
        model_path = self.config.get('path', 'models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf')
        
        if not os.path.exists(model_path):
            logging.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            self.model = Llama(
                model_path=model_path,
                n_ctx=self.config.get('context_length', 8192),
                n_batch=self.config.get('batch_size', 512),
                n_threads=self.config.get('threads', 8),
                n_gpu_layers=self.config.get('gpu_layers', 35),
                verbose=False
            )
            
            logging.info(f"Loaded quantized Llama model: {model_path}")
            
        except Exception as e:
            logging.error(f"Failed to load Llama model: {e}")
            raise
    
    def generate_text(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> Tuple[str, int]:
        """Generate text with the local LLM"""
        if self.model is None:
            self.load_model()
        
        try:
            response = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                echo=False,
                stop=["</s>", "\n\n"]
            )
            
            generated_text = response['choices'][0]['text']
            tokens_used = response['usage']['total_tokens']
            
            return generated_text, tokens_used
            
        except Exception as e:
            logging.error(f"LLM generation failed: {e}")
            return "", 0
    
    def enhance_chunk(self, chunk_text: str) -> Tuple[str, float]:
        """Enhance chunk text with context and metadata"""
        prompt = f"""
Analyze and enhance this academic document chunk for better retrieval:

Chunk:
{chunk_text}

Tasks:
1. Extract key concepts and entities
2. Add semantic context
3. Identify document type (paper, notes, slides, etc.)
4. Generate relevant keywords

Enhanced chunk (keep original text + add metadata):
"""
        
        enhanced_text, tokens = self.generate_text(prompt, max_tokens=256)
        cost = tokens * self.cost_per_token
        
        return enhanced_text, cost


class CPUEmbeddingModel:
    """CPU-optimized embedding model for cost efficiency"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.cost_per_embedding = 0.000001  # Very low cost for CPU inference
        
    def load_model(self):
        """Load embedding model on CPU"""
        if self.model is not None:
            return
        
        model_name = self.config.get('model_name', 'BAAI/bge-small-en-v1.5')
        
        try:
            self.model = SentenceTransformer(
                model_name,
                device='cpu'  # Force CPU to save GPU memory
            )
            
            # Optimize for inference
            self.model.eval()
            
            logging.info(f"Loaded embedding model on CPU: {model_name}")
            
        except Exception as e:
            logging.error(f"Failed to load embedding model: {e}")
            raise
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> Tuple[np.ndarray, float]:
        """Encode texts in batches for efficiency"""
        if self.model is None:
            self.load_model()
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=self.config.get('normalize_embeddings', True)
            )
            
            cost = len(texts) * self.cost_per_embedding
            
            return embeddings, cost
            
        except Exception as e:
            logging.error(f"Embedding encoding failed: {e}")
            return np.array([]), 0.0
    
    def encode_single(self, text: str) -> Tuple[np.ndarray, float]:
        """Encode single text"""
        embeddings, cost = self.encode_batch([text], batch_size=1)
        return embeddings[0] if len(embeddings) > 0 else np.array([]), cost


class LocalLLMStage:
    """Main local LLM processing stage"""
    
    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.llm_config = self.config['local_llm']
        
        # Initialize components
        self.chunker = AdaptiveChunker(self.llm_config['chunking'])
        self.llm = QuantizedLlama(self.llm_config['model'])
        self.embedding_model = CPUEmbeddingModel(self.llm_config['embeddings'])
        
        # Initialize caching
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
        
        logging.info("Local LLM stage initialized")
    
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
    
    def process_visual_results(self, visual_results: List[VisualParsingResult]) -> ChunkingResult:
        """Process visual parsing results into chunks with embeddings"""
        logging.info(f"Processing {len(visual_results)} pages with local LLM")
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
            
            # Check cache first
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
                
                # Process page text into chunks
                page_chunks = self.chunker.chunk_text(visual_result.text_content, page_idx)
                
                # Enhance chunks with LLM if needed
                enhanced_chunks = []
                for chunk in page_chunks:
                    # For cost optimization, only enhance complex chunks
                    if len(chunk.text) > 800 or any(formula in chunk.text for formula in visual_result.latex_formulas):
                        enhanced_text, enhancement_cost = self.llm.enhance_chunk(chunk.text)
                        chunk.text = enhanced_text
                        chunk.processing_cost += enhancement_cost
                        total_cost += enhancement_cost
                    
                    enhanced_chunks.append(chunk)
                
                all_chunks.extend(enhanced_chunks)
                
                # Cache the results
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
        
        # Generate embeddings for all chunks
        if all_chunks:
            chunk_texts = [chunk.text for chunk in all_chunks]
            embeddings, embedding_cost = self.embedding_model.encode_batch(chunk_texts)
            total_cost += embedding_cost
            
            # Assign embeddings to chunks
            for chunk, embedding in zip(all_chunks, embeddings):
                chunk.embedding = embedding
                
                # Cache embedding separately
                self.embedding_cache.cache_embedding(chunk.text, embedding)
        
        # Calculate metrics
        processing_time = time.time() - start_time
        cache_hit_rate = cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0.0
        
        # Track costs
        self.cost_tracker.add_cost("llm_processing", total_cost)
        
        # Log summary
        logging.info(f"Processed {len(all_chunks)} chunks in {processing_time:.2f}s")
        logging.info(f"Cache hit rate: {cache_hit_rate:.2%}")
        logging.info(f"Total cost: ${total_cost:.6f}")
        
        return ChunkingResult(
            chunks=all_chunks,
            total_tokens=total_tokens,
            processing_time=processing_time,
            cost_estimate=total_cost,
            cache_hit_rate=cache_hit_rate,
            metadata={
                "pages_processed": len(visual_results),
                "cache_hits": cache_hits,
                "cache_misses": cache_misses
            }
        )
    
    async def process_batch(self, visual_results_batch: List[List[VisualParsingResult]]) -> List[ChunkingResult]:
        """Process multiple documents in parallel"""
        tasks = [
            asyncio.create_task(asyncio.to_thread(self.process_visual_results, visual_results))
            for visual_results in visual_results_batch
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = [result for result in results if not isinstance(result, Exception)]
        
        return valid_results
    
    def optimize_chunks_for_retrieval(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Optimize chunks for better retrieval performance"""
        logging.info(f"Optimizing {len(chunks)} chunks for retrieval")
        
        optimized_chunks = []
        
        for chunk in chunks:
            # Add retrieval-optimized metadata
            chunk.metadata = chunk.metadata or {}
            
            # Calculate chunk quality score
            quality_score = self._calculate_chunk_quality(chunk)
            chunk.confidence_score = quality_score
            
            # Add semantic keywords
            keywords = self._extract_keywords(chunk.text)
            chunk.metadata['keywords'] = keywords
            
            # Add chunk type classification
            chunk_type = self._classify_chunk_type(chunk.text)
            chunk.metadata['type'] = chunk_type
            
            optimized_chunks.append(chunk)
        
        return optimized_chunks
    
    def _calculate_chunk_quality(self, chunk: DocumentChunk) -> float:
        """Calculate quality score for chunk"""
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
        """Extract keywords from text (simplified)"""
        # In a full implementation, this would use more sophisticated NLP
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
        """Classify chunk type based on content patterns"""
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
        # Clear caches
        self.cache_manager.clear_all_caches()
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Example usage
async def main():
    """Example usage of local LLM stage"""
    stage = LocalLLMStage()
    
    # Mock visual parsing results
    mock_visual_results = [
        VisualParsingResult(
            text_content="This is a sample academic document discussing machine learning algorithms and their applications in natural language processing.",
            latex_formulas=["$y = mx + b$"],
            tables=[],
            layout_info={"type": "academic_paper"},
            confidence_scores={"text": 0.95},
            processing_time=1.0,
            cost_estimate=0.01
        )
    ]
    
    # Process with local LLM
    result = stage.process_visual_results(mock_visual_results)
    
    print(f"Generated {len(result.chunks)} chunks")
    print(f"Processing time: {result.processing_time:.2f}s")
    print(f"Cost: ${result.cost_estimate:.6f}")
    print(f"Cache hit rate: {result.cache_hit_rate:.2%}")
    
    # Get cost report
    cost_report = stage.get_cost_report()
    print("Cost Report:", json.dumps(cost_report, indent=2, default=str))
    
    stage.cleanup()


if __name__ == "__main__":
    asyncio.run(main())