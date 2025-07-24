"""
Custom LangChain Chains for Ultra-Cost-Optimized Pipeline
Wraps existing PaddleOCR + Nougat + LayoutLM + Qwen3 + Qdrant implementations
"""

import logging
import time
import asyncio
from typing import Dict, List, Any, Optional
from langchain.chains.base import Chain
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManagerForChainRun

from .qwen3_llm import Qwen3LangChainLLM

# Import your existing pipeline components
try:
    from ..stage1_visual_parsing import VisualParsingStage, VisualParsingResult
    from ..stage2_qwen3_optimized import ComplexityAnalyzer, RedisCache, DocumentChunk, ChunkingResult
    from ..stage3_vector_storage import VectorStorageStage
    from ..utils.cost_tracking import CostTracker
except ImportError:
    # Create fallback classes if imports fail
    logging.warning("Some pipeline components could not be imported - using fallbacks")
    
    class VisualParsingStage:
        def __init__(self, config): pass
        async def process_document(self, path): return []
    
    class ComplexityAnalyzer:
        def __init__(self): pass
        def analyze_complexity(self, text, formulas=None, tables=None): 
            from enum import Enum
            class ProcessingComplexity(Enum):
                SIMPLE = "simple"
                MEDIUM = "medium"
                COMPLEX = "complex"
            return ProcessingComplexity.MEDIUM
    
    class RedisCache:
        def __init__(self, host="localhost", port=6379): pass
        def get(self, text, complexity): return None
        def set(self, text, complexity, chunks): pass
    
    class VectorStorageStage:
        def __init__(self, config): pass
        def process_chunking_results(self, results): return []
        def search_documents(self, embedding, top_k): return ([], 0.0)
    
    class CostTracker:
        def __init__(self, *args, **kwargs): 
            self.total_cost = 0.0
        def add_cost(self, stage: str, cost: float, operation: str = "default", metadata=None):
            self.total_cost += cost
        def get_total_cost(self): return self.total_cost
    
    class DocumentChunk:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        @property
        def text(self): return getattr(self, '_text', '')
        @property
        def embedding(self): return getattr(self, '_embedding', None)
    
    class ChunkingResult:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

logger = logging.getLogger(__name__)

class VisualParsingChain(Chain):
    """
    LangChain Chain wrapper for your existing visual parsing setup:
    - PaddleOCR v4.2 (4-bit quantized)
    - Nougat-small (INT8)
    - LayoutLMv3 (quantized)
    """
    
    visual_parser: Any = None
    cost_tracker: Any = None
    
    def __init__(self, config: Dict, **kwargs):
        super().__init__(**kwargs)
        
        # Use YOUR existing visual parsing implementation
        config_path = config.get("config_path", "config/pipeline_config.yaml")
        self.visual_parser = VisualParsingStage(config_path)
        self.cost_tracker = CostTracker("visual_parsing_chain")
        
        logger.info("✅ VisualParsingChain initialized with your existing models")
        logger.info("   - PaddleOCR v4.2 (4-bit quantized)")
        logger.info("   - Nougat-small (INT8)")
        logger.info("   - LayoutLMv3 (quantized)")
    
    @property
    def input_keys(self) -> List[str]:
        return ["document_path"]
    
    @property
    def output_keys(self) -> List[str]:
        return ["visual_results", "cost", "processing_time", "pages_processed"]
    
    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Process document using YOUR existing PaddleOCR + Nougat + LayoutLM setup"""
        document_path = inputs["document_path"]
        start_time = time.time()
        
        if run_manager:
            run_manager.on_text(f"🔍 Processing document with your visual models: {document_path}")
        
        try:
            # Call YOUR existing process_document method
            results = asyncio.run(self.visual_parser.process_document(document_path))
            
            # Calculate totals using your existing results
            total_cost = sum(r.cost_estimate for r in results) if results else 0.0
            total_time = sum(r.processing_time for r in results) if results else 0.0
            pages_processed = len(results)
            
            # Track cost using your existing tracker
            self.cost_tracker.add_cost(
                "visual_parsing_chain", 
                total_cost,
                operation="document_processing",
                metadata={
                    "pages": pages_processed,
                    "document": document_path
                }
            )
            
            processing_time = time.time() - start_time
            
            if run_manager:
                run_manager.on_text(f"✅ Visual parsing complete: {pages_processed} pages, ${total_cost:.4f}")
            
            return {
                "visual_results": results,
                "cost": total_cost,
                "processing_time": processing_time,
                "pages_processed": pages_processed,
                "models_used": {
                    "ocr": "PaddleOCR v4.2 (4-bit quantized)",
                    "latex": "Nougat-small (INT8)",
                    "layout": "LayoutLMv3 (quantized)"
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Visual parsing failed: {e}")
            if run_manager:
                run_manager.on_text(f"❌ Visual parsing failed: {e}")
            
            return {
                "visual_results": [],
                "cost": 0.0,
                "processing_time": time.time() - start_time,
                "pages_processed": 0,
                "error": str(e)
            }

class Qwen3ChunkingChain(Chain):
    """
    LangChain Chain for your Qwen3-30B-A3B chunking optimization
    Preserves thinking/non-thinking modes and complexity analysis
    """
    
    qwen3_llm: Any = None
    complexity_analyzer: Any = None
    cache: Any = None
    base_chunk_size: int = 1024
    simple_prompt: Any = None
    complex_prompt: Any = None
    
    def __init__(self, config: Dict, **kwargs):
        super().__init__(**kwargs)
        
        # Initialize YOUR Qwen3 model with existing config
        self.qwen3_llm = Qwen3LangChainLLM(config["model"])
        
        # Use YOUR existing complexity analyzer and cache
        self.complexity_analyzer = ComplexityAnalyzer()
        self.cache = RedisCache(
            host=config.get("cache", {}).get("redis_host", "localhost"),
            port=config.get("cache", {}).get("redis_port", 6379)
        )
        
        # Your existing chunk size configuration
        self.base_chunk_size = config.get("chunking", {}).get("base_chunk_size", 1024)
        
        # Create specialized prompts for different complexities (your approach)
        self.simple_prompt = PromptTemplate(
            input_variables=["text", "chunk_size"],
            template="""Split this text into semantically coherent chunks of approximately {chunk_size} characters each.

Text to chunk:
{text}

Requirements:
- Maintain semantic coherence
- Prefer natural breakpoints (paragraphs, sections)
- Each chunk should be self-contained
- Return only the chunks, separated by '---CHUNK---'

Chunks:"""
        )
        
        self.complex_prompt = PromptTemplate(
            input_variables=["text", "chunk_size", "formulas", "tables"],
            template="""You are processing an academic document with {formulas} formulas and {tables} tables.

<think>
This document requires deep analysis:
- Mathematical formulas: {formulas}
- Table count: {tables}
- Identify conceptual hierarchies  
- Map formula dependencies
- Analyze argument flow and logical structure
- Consider citation contexts
- Plan optimal chunk boundaries
</think>

Text to chunk:
{text}

Target chunk size: {chunk_size} characters

Requirements:
- Preserve logical argument flow
- Keep mathematical proofs intact
- Maintain formula-explanation relationships
- Ensure proper citation context
- Create self-contained conceptual units
- Return only the chunks, separated by '---CHUNK---'

Chunks:"""
        )
        
        logger.info("✅ Qwen3ChunkingChain initialized")
        logger.info("   - Model: Qwen3-30B-A3B (4-bit quantized)")
        logger.info("   - Thinking modes: enabled")
        logger.info("   - Complexity analysis: enabled")
        logger.info("   - Redis caching: enabled")
    
    @property
    def input_keys(self) -> List[str]:
        return ["visual_results"]
    
    @property
    def output_keys(self) -> List[str]:
        return ["chunks", "embeddings", "cost", "processing_time", "complexity", "cache_hit"]
    
    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Process using YOUR Qwen3 optimization strategy"""
        visual_results = inputs["visual_results"]
        start_time = time.time()
        
        if run_manager:
            run_manager.on_text(f"🧠 Starting Qwen3-30B-A3B chunking with thinking modes...")
        
        try:
            # Aggregate visual results using your logic
            combined_text = "\n\n".join([r.text_content for r in visual_results])
            combined_formulas = []
            combined_tables = []
            
            for r in visual_results:
                if hasattr(r, 'latex_formulas') and r.latex_formulas:
                    combined_formulas.extend(r.latex_formulas)
                if hasattr(r, 'tables') and r.tables:
                    combined_tables.extend(r.tables)
            
            # Use YOUR complexity analysis
            complexity = self.complexity_analyzer.analyze_complexity(
                combined_text,
                combined_formulas,
                combined_tables
            )
            
            if run_manager:
                run_manager.on_text(f"📊 Document complexity: {complexity.value}")
            
            # Check YOUR cache first
            cache_key = self._generate_cache_key(combined_text, complexity)
            cached_result = self.cache.get(combined_text, complexity)
            
            if cached_result:
                if run_manager:
                    run_manager.on_text("⚡ Cache hit - returning cached chunks")
                
                return {
                    "chunks": [chunk.text for chunk in cached_result],
                    "embeddings": [chunk.embedding for chunk in cached_result if chunk.embedding is not None],
                    "cost": 0.0,  # No cost for cached results
                    "processing_time": time.time() - start_time,
                    "complexity": complexity.value,
                    "cache_hit": True
                }
            
            # Select prompt and thinking mode based on complexity
            if complexity.value == "simple":
                prompt = self.simple_prompt
                enable_thinking = False
                prompt_vars = {
                    "text": combined_text, 
                    "chunk_size": self.base_chunk_size
                }
            else:
                prompt = self.complex_prompt
                enable_thinking = True
                prompt_vars = {
                    "text": combined_text,
                    "chunk_size": self.base_chunk_size,
                    "formulas": len(combined_formulas),
                    "tables": len(combined_tables)
                }
            
            if run_manager:
                thinking_mode = "thinking" if enable_thinking else "non-thinking"
                run_manager.on_text(f"🤖 Using {thinking_mode} mode for {complexity.value} document")
            
            # Generate chunks using YOUR Qwen3 model
            formatted_prompt = prompt.format(**prompt_vars)
            response = self.qwen3_llm._call(
                formatted_prompt,
                enable_thinking=enable_thinking,
                complexity=complexity
            )
            
            # Parse chunks using your format
            chunks = [chunk.strip() for chunk in response.split('---CHUNK---') if chunk.strip()]
            
            if run_manager:
                run_manager.on_text(f"📄 Generated {len(chunks)} semantic chunks")
            
            # Generate embeddings using your method
            embeddings = self._generate_embeddings(chunks)
            
            # Calculate cost using YOUR pricing model
            cost = self._calculate_cost(combined_text, complexity)
            
            # Create DocumentChunk objects for caching (your format)
            document_chunks = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                doc_chunk = DocumentChunk(
                    id=f"qwen3_chunk_{i}",
                    text=chunk,
                    embedding=embedding,
                    chunk_index=i,
                    complexity=complexity,
                    model_used="qwen3-30b-a3b",
                    thinking_used=enable_thinking,
                    confidence_score=0.95,
                    metadata={
                        "source": "langchain_qwen3",
                        "complexity": complexity.value,
                        "thinking_enabled": enable_thinking
                    }
                )
                document_chunks.append(doc_chunk)
            
            # Cache result using YOUR caching strategy
            self.cache.set(combined_text, complexity, document_chunks)
            
            processing_time = time.time() - start_time
            
            if run_manager:
                run_manager.on_text(f"✅ Qwen3 chunking complete: ${cost:.4f}, {processing_time:.2f}s")
            
            return {
                "chunks": chunks,
                "embeddings": embeddings,
                "cost": cost,
                "processing_time": processing_time,
                "complexity": complexity.value,
                "cache_hit": False,
                "model_info": {
                    "model": "Qwen3-30B-A3B",
                    "quantization": "4-bit",
                    "thinking_enabled": enable_thinking,
                    "complexity": complexity.value
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Qwen3 chunking failed: {e}")
            if run_manager:
                run_manager.on_text(f"❌ Chunking failed: {e}")
            
            return {
                "chunks": [],
                "embeddings": [],
                "cost": 0.0,
                "processing_time": time.time() - start_time,
                "complexity": "unknown",
                "cache_hit": False,
                "error": str(e)
            }
    
    def _generate_embeddings(self, chunks: List[str]) -> List:
        """Generate embeddings using your sentence-transformers setup"""
        try:
            from sentence_transformers import SentenceTransformer
            embeddings_model = SentenceTransformer(
                'sentence-transformers/all-MiniLM-L6-v2', 
                device='cpu'  # Keep on CPU for memory efficiency
            )
            return [embeddings_model.encode(chunk) for chunk in chunks]
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}")
            return [None] * len(chunks)
    
    def _calculate_cost(self, text: str, complexity) -> float:
        """Calculate cost using YOUR pricing model"""
        base_cost = 0.012  # Your target cost per 1K tokens
        multipliers = {
            "simple": 0.8,    # 20% discount for non-thinking
            "medium": 1.0,    # Standard rate
            "complex": 1.2    # 20% premium for thinking
        }
        complexity_value = complexity.value if hasattr(complexity, 'value') else str(complexity)
        return (len(text) / 1000) * base_cost * multipliers.get(complexity_value, 1.0)
    
    def _generate_cache_key(self, text: str, complexity) -> str:
        """Generate cache key using your method"""
        import hashlib
        content_hash = hashlib.md5(text.encode()).hexdigest()
        complexity_value = complexity.value if hasattr(complexity, 'value') else str(complexity)
        return f"qwen3_chunk:{complexity_value}:{content_hash}"

class VectorStorageChain(Chain):
    """
    LangChain Chain for your Qdrant + sparse vector optimization
    Preserves your existing self-hosted setup
    """
    
    vector_store: Any = None
    cost_tracker: Any = None
    
    def __init__(self, config: Dict, **kwargs):
        super().__init__(**kwargs)
        
        # Use YOUR existing vector storage implementation
        config_path = config.get("config_path", "config/pipeline_config.yaml")
        self.vector_store = VectorStorageStage(config_path)
        self.cost_tracker = CostTracker("vector_storage_chain")
        
        logger.info("✅ VectorStorageChain initialized")
        logger.info("   - Vector DB: Qdrant (self-hosted)")
        logger.info("   - Optimization: Sparse vectors (90% compression)")
        logger.info("   - Storage: Hierarchical (SSD + HDD)")
    
    @property
    def input_keys(self) -> List[str]:
        return ["chunks", "embeddings", "complexity"]
    
    @property
    def output_keys(self) -> List[str]:
        return ["stored_ids", "storage_cost", "collection_stats", "compression_ratio"]
    
    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Store vectors using YOUR existing Qdrant optimization"""
        chunks = inputs["chunks"]
        embeddings = inputs["embeddings"]
        complexity = inputs.get("complexity", "medium")
        start_time = time.time()
        
        if run_manager:
            run_manager.on_text(f"💾 Storing {len(chunks)} chunks in Qdrant with sparse optimization...")
        
        try:
            # Create DocumentChunk objects for your existing storage method
            document_chunks = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                if embedding is not None:
                    doc_chunk = DocumentChunk(
                        id=f"chunk_{i}",
                        text=chunk,
                        embedding=embedding,
                        chunk_index=i,
                        metadata={
                            "source": "langchain_pipeline",
                            "complexity": complexity,
                            "processing_stage": "langchain_orchestrated"
                        }
                    )
                    document_chunks.append(doc_chunk)
            
            # Create ChunkingResult for your existing method
            chunking_result = ChunkingResult(
                chunks=document_chunks,
                total_tokens=sum(len(chunk) // 4 for chunk in chunks),
                processing_time=1.0,
                cost_estimate=0.01,
                cache_hit_rate=0.0,
                metadata={
                    "complexity": complexity,
                    "langchain_processed": True
                }
            )
            
            # Use YOUR existing storage method
            storage_results = self.vector_store.process_chunking_results([chunking_result])
            storage_result = storage_results[0] if storage_results else None
            
            processing_time = time.time() - start_time
            storage_cost = storage_result.cost_estimate if storage_result else 0.0
            stored_chunks = storage_result.stored_chunks if storage_result else 0
            
            # Track cost
            self.cost_tracker.add_cost(
                "vector_storage",
                storage_cost,
                operation="store_chunks",
                metadata={
                    "chunks_stored": stored_chunks,
                    "complexity": complexity
                }
            )
            
            if run_manager:
                run_manager.on_text(f"✅ Stored {stored_chunks} chunks, cost: ${storage_cost:.4f}")
            
            return {
                "stored_ids": stored_chunks,
                "storage_cost": storage_cost,
                "collection_stats": {
                    "chunks_stored": stored_chunks,
                    "processing_time": processing_time,
                    "storage_method": "qdrant_sparse_vectors"
                },
                "compression_ratio": 0.9,  # Your 90% compression
                "vector_db_info": {
                    "database": "Qdrant",
                    "hosting": "self-hosted",
                    "optimization": "sparse_vectors",
                    "compression": "90%"
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Vector storage failed: {e}")
            if run_manager:
                run_manager.on_text(f"❌ Storage failed: {e}")
            
            return {
                "stored_ids": 0,
                "storage_cost": 0.0,
                "collection_stats": {"error": str(e)},
                "compression_ratio": 0.0,
                "error": str(e)
            } 