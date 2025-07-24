"""
Stage 2: Qwen3-30B-A3B Optimized Chunking Pipeline
Target: $0.012 per 1K pages with 96-99% accuracy
Features: Proper thinking/non-thinking modes, LangChain orchestration, Redis caching
"""

import os
import time
import logging
import asyncio
import hashlib
import json
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import sys
import yaml

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import redis
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun

# Import existing pipeline components
try:
    from .utils.cost_tracking import CostTracker
    from .stage1_visual_parsing import VisualParsingResult
except ImportError:
    # Fallback imports
    class CostTracker:
        def __init__(self, *args, **kwargs): 
            self.total_cost = 0.0
        def start_operation(self, *args, **kwargs): return None
        def end_operation(self, *args, **kwargs): return 0.0
        def add_cost(self, stage: str, cost: float, operation: str = "default", metadata=None):
            self.total_cost += cost
        def get_total_cost(self): return self.total_cost
    
    @dataclass
    class VisualParsingResult:
        text_content: str
        latex_formulas: List[str] = None
        tables: List[Dict] = None
        layout_info: Dict = None
        confidence_scores: Dict[str, float] = None
        processing_time: float = 0.0
        cost_estimate: float = 0.0


class ProcessingComplexity(Enum):
    """Document processing complexity levels"""
    SIMPLE = "simple"       # Basic text, no complex structures
    MEDIUM = "medium"       # Some formulas, tables, or technical content
    COMPLEX = "complex"     # Heavy academic content, complex formulas


@dataclass
class DocumentChunk:
    """Enhanced document chunk with comprehensive metadata"""
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
    complexity: ProcessingComplexity = ProcessingComplexity.SIMPLE
    model_used: str = "qwen3-30b-a3b"
    thinking_used: bool = False


@dataclass
class ChunkingResult:
    """Enhanced chunking result with detailed metrics"""
    chunks: List[DocumentChunk]
    total_tokens: int
    processing_time: float
    cost_estimate: float
    cache_hit_rate: float
    metadata: Dict = None
    model_performance: Dict = None
    complexity_distribution: Dict = None


class Qwen3LangChainLLM(LLM):
    """LangChain wrapper for Qwen3-30B-A3B model"""
    
    def __init__(self, model_path: str, tokenizer_path: str, device: str = "cuda"):
        super().__init__()
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """Load Qwen3-30B-A3B model with optimal configuration"""
        try:
            # Configure quantization for memory efficiency
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path,
                trust_remote_code=True,
                use_fast=True
            )
            
            # Load model with quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            
            logging.info(f"Successfully loaded Qwen3-30B-A3B model from {self.model_path}")
            
        except Exception as e:
            logging.error(f"Failed to load Qwen3 model: {e}")
            raise
    
    @property
    def _llm_type(self) -> str:
        return "qwen3-30b-a3b"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        enable_thinking: bool = True,
        complexity: ProcessingComplexity = ProcessingComplexity.MEDIUM,
        **kwargs: Any,
    ) -> str:
        """Call Qwen3 model with proper thinking/non-thinking configuration"""
        
        # Configure generation parameters based on thinking mode
        if enable_thinking:
            # Thinking mode parameters (as per Qwen3 docs)
            generation_kwargs = {
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0.0,
                "do_sample": True,  # Never use greedy decoding
                "max_new_tokens": 2048,
                "enable_thinking": True
            }
        else:
            # Non-thinking mode parameters
            generation_kwargs = {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "min_p": 0.0,
                "do_sample": True,
                "max_new_tokens": 1024,
                "enable_thinking": False
            }
        
        # Prepare chat template
        messages = [{"role": "user", "content": prompt}]
        
        try:
            # Apply chat template with thinking configuration
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking
            )
            
            # Tokenize input
            inputs = self.tokenizer.encode(text, return_tensors="pt").to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **generation_kwargs
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
            
            # Extract final response from thinking blocks if present
            if enable_thinking and "<think>" in response:
                # Extract content after </think> tag
                parts = response.split("</think>")
                if len(parts) > 1:
                    response = parts[-1].strip()
                else:
                    # Fallback: use full response
                    response = response.replace("<think>", "").replace("</think>", "").strip()
            
            return response
            
        except Exception as e:
            logging.error(f"Error during Qwen3 generation: {e}")
            raise


class ComplexityAnalyzer:
    """Analyze document complexity to determine processing strategy"""
    
    def __init__(self):
        # Patterns indicating complexity
        self.formula_patterns = [r'\$.*?\$', r'\\begin\{.*?\}', r'\\[a-zA-Z]+\{']
        self.table_indicators = ['|', '\\hline', '\\begin{table}', '\\begin{tabular}']
        self.citation_patterns = [r'\[[0-9,\-\s]+\]', r'\([A-Za-z]+,?\s*\d{4}\)']
        self.technical_terms = ['theorem', 'lemma', 'proof', 'algorithm', 'equation']
    
    def analyze_complexity(self, text: str, latex_formulas: List[str] = None, 
                          tables: List[Dict] = None) -> ProcessingComplexity:
        """Analyze document complexity and return processing strategy"""
        
        complexity_score = 0.0
        
        # Text-based indicators
        formula_count = sum(len(__import__('re').findall(pattern, text)) 
                          for pattern in self.formula_patterns)
        complexity_score += min(formula_count * 0.1, 0.3)
        
        # Table indicators
        table_count = sum(text.count(indicator) for indicator in self.table_indicators)
        complexity_score += min(table_count * 0.05, 0.2)
        
        # Citation density
        citation_count = sum(len(__import__('re').findall(pattern, text)) 
                           for pattern in self.citation_patterns)
        complexity_score += min(citation_count * 0.02, 0.2)
        
        # Technical term density
        tech_count = sum(text.lower().count(term) for term in self.technical_terms)
        complexity_score += min(tech_count * 0.03, 0.15)
        
        # LaTeX formulas from Stage 1
        if latex_formulas:
            complexity_score += min(len(latex_formulas) * 0.05, 0.25)
        
        # Tables from Stage 1
        if tables:
            complexity_score += min(len(tables) * 0.1, 0.3)
        
        # Document length factor
        length_factor = min(len(text) / 10000, 0.2)
        complexity_score += length_factor
        
        # Determine complexity level
        if complexity_score < 0.3:
            return ProcessingComplexity.SIMPLE
        elif complexity_score < 0.7:
            return ProcessingComplexity.MEDIUM
        else:
            return ProcessingComplexity.COMPLEX


class RedisCache:
    """Redis caching with intelligent TTL and compression"""
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        try:
            self.redis_client = redis.Redis(
                host=host, port=port, db=db,
                decode_responses=False,  # Handle binary data
                socket_timeout=5.0,
                socket_connect_timeout=5.0
            )
            # Test connection
            self.redis_client.ping()
            self.enabled = True
            logging.info("Redis cache initialized successfully")
        except Exception as e:
            logging.warning(f"Redis cache unavailable: {e}. Using in-memory cache.")
            self.enabled = False
            self.memory_cache = {}
    
    def _generate_key(self, text: str, complexity: ProcessingComplexity) -> str:
        """Generate cache key from text and complexity"""
        content_hash = hashlib.md5(text.encode()).hexdigest()
        return f"qwen3_chunk:{complexity.value}:{content_hash}"
    
    def get(self, text: str, complexity: ProcessingComplexity) -> Optional[List[DocumentChunk]]:
        """Get cached chunks"""
        key = self._generate_key(text, complexity)
        
        try:
            if self.enabled:
                cached_data = self.redis_client.get(key)
                if cached_data:
                    # Decompress and deserialize
                    import pickle, gzip
                    data = pickle.loads(gzip.decompress(cached_data))
                    return [DocumentChunk(**chunk_data) for chunk_data in data]
            else:
                return self.memory_cache.get(key)
        except Exception as e:
            logging.warning(f"Cache get error: {e}")
        
        return None
    
    def set(self, text: str, complexity: ProcessingComplexity, 
            chunks: List[DocumentChunk], ttl: int = 86400):
        """Set cached chunks with compression"""
        key = self._generate_key(text, complexity)
        
        try:
            # Serialize chunk data (excluding embeddings for cache)
            chunk_data = []
            for chunk in chunks:
                data = asdict(chunk)
                data['embedding'] = None  # Don't cache embeddings
                chunk_data.append(data)
            
            if self.enabled:
                # Compress and store
                import pickle, gzip
                compressed_data = gzip.compress(pickle.dumps(chunk_data))
                self.redis_client.setex(key, ttl, compressed_data)
            else:
                self.memory_cache[key] = chunks
                
        except Exception as e:
            logging.warning(f"Cache set error: {e}")


class Qwen3OptimizedChunker:
    """Main Qwen3-30B-A3B chunking pipeline with cost optimization"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.setup_logging()
        
        # Initialize components
        self.complexity_analyzer = ComplexityAnalyzer()
        self.cache = RedisCache(
            host=self.config.get('redis_host', 'localhost'),
            port=self.config.get('redis_port', 6379)
        )
        self.cost_tracker = CostTracker("stage2_qwen3")
        
        # Model and tokenizer paths
        self.model_path = self.config.get('qwen3_model_path', 'Qwen/Qwen3-30B-A3B')
        self.tokenizer_path = self.config.get('qwen3_tokenizer_path', 'Qwen/Qwen3-30B-A3B')
        
        # Initialize LLM
        self.llm = Qwen3LangChainLLM(
            model_path=self.model_path,
            tokenizer_path=self.tokenizer_path
        )
        
        # Initialize embeddings model
        self.embeddings_model = SentenceTransformer(
            'sentence-transformers/all-MiniLM-L6-v2',
            device='cpu'  # Keep embeddings on CPU for memory efficiency
        )
        
        # LangChain components
        self.setup_langchain_chains()
        
        # Performance metrics
        self.metrics = {
            'total_documents': 0,
            'total_chunks': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'processing_times': [],
            'costs': []
        }
        
        logging.info("Qwen3 Optimized Chunker initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config.get('stage2_qwen3', {})
        except Exception as e:
            logging.warning(f"Could not load config from {config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'base_chunk_size': 1024,
            'overlap': 128,
            'min_chunk_size': 256,
            'max_chunk_size': 2048,
            'enable_thinking_for_complex': True,
            'cache_ttl': 86400,
            'cost_per_1k_tokens': 0.012,
            'redis_host': 'localhost',
            'redis_port': 6379
        }
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('stage2_qwen3.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('Qwen3Stage2')
    
    def setup_langchain_chains(self):
        """Setup LangChain chains for different complexity levels"""
        
        # Simple chunking prompt (no thinking needed)
        simple_prompt = PromptTemplate(
            input_variables=["text", "chunk_size"],
            template="""
You are an expert document chunker. Split this text into semantically coherent chunks of approximately {chunk_size} characters each.

Text to chunk:
{text}

Requirements:
- Maintain semantic coherence
- Prefer natural breakpoints (paragraphs, sections)
- Each chunk should be self-contained
- Return only the chunks, separated by '---CHUNK---'

Chunks:
"""
        )
        
        # Medium complexity prompt (light thinking)
        medium_prompt = PromptTemplate(
            input_variables=["text", "chunk_size"],
            template="""
You are an expert academic document chunker. This document contains technical content that requires careful analysis.

/think
Analyze the structure and identify:
- Key concepts and their boundaries
- Mathematical formulas and their context
- Table references and explanations
- Section transitions

Text to chunk:
{text}

Target chunk size: {chunk_size} characters

Requirements:
- Maintain semantic and conceptual coherence
- Keep formulas with their explanations
- Preserve table-text relationships
- Ensure each chunk provides complete context
- Return only the chunks, separated by '---CHUNK---'

Chunks:
"""
        )
        
        # Complex chunking prompt (full thinking)
        complex_prompt = PromptTemplate(
            input_variables=["text", "chunk_size", "formulas", "tables"],
            template="""
You are an expert academic document processor specializing in complex research papers.

/think
This document requires deep analysis:
- Mathematical formulas: {formulas}
- Table count: {tables}
- Identify conceptual hierarchies
- Map formula dependencies
- Analyze argument flow and logical structure
- Consider citation contexts
- Plan optimal chunk boundaries

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

Chunks:
"""
        )
        
        # Create LangChain chains
        self.simple_chain = LLMChain(llm=self.llm, prompt=simple_prompt)
        self.medium_chain = LLMChain(llm=self.llm, prompt=medium_prompt)
        self.complex_chain = LLMChain(llm=self.llm, prompt=complex_prompt)
    
    async def process_document(self, visual_result: VisualParsingResult) -> ChunkingResult:
        """Main processing pipeline for document chunking"""
        start_time = time.time()
        operation_id = self.cost_tracker.start_operation("chunk_document", {
            "text_length": len(visual_result.text_content),
            "has_formulas": bool(visual_result.latex_formulas),
            "has_tables": bool(visual_result.tables)
        })
        
        try:
            # Analyze document complexity
            complexity = self.complexity_analyzer.analyze_complexity(
                visual_result.text_content,
                visual_result.latex_formulas,
                visual_result.tables
            )
            
            self.logger.info(f"Document complexity: {complexity.value}")
            
            # Check cache first
            cached_chunks = self.cache.get(visual_result.text_content, complexity)
            if cached_chunks:
                self.metrics['cache_hits'] += 1
                self.logger.info("Cache hit - returning cached chunks")
                
                # Add embeddings to cached chunks
                for chunk in cached_chunks:
                    chunk.embedding = self.embeddings_model.encode(chunk.text)
                
                processing_time = time.time() - start_time
                return ChunkingResult(
                    chunks=cached_chunks,
                    total_tokens=sum(len(chunk.text) // 4 for chunk in cached_chunks),
                    processing_time=processing_time,
                    cost_estimate=0.0,  # No cost for cached results
                    cache_hit_rate=1.0,
                    metadata={
                        "complexity": complexity.value,
                        "source": "cache"
                    }
                )
            
            # Cache miss - process with Qwen3
            self.metrics['cache_misses'] += 1
            chunks = await self._chunk_with_qwen3(
                visual_result, complexity
            )
            
            # Generate embeddings
            for chunk in chunks:
                chunk.embedding = self.embeddings_model.encode(chunk.text)
            
            # Cache results
            self.cache.set(visual_result.text_content, complexity, chunks)
            
            # Calculate final metrics
            processing_time = time.time() - start_time
            total_tokens = sum(len(chunk.text) // 4 for chunk in chunks)
            cost_estimate = self._calculate_cost(total_tokens, complexity)
            
            self.cost_tracker.end_operation(operation_id, cost_estimate)
            
            # Update metrics
            self.metrics['total_documents'] += 1
            self.metrics['total_chunks'] += len(chunks)
            self.metrics['processing_times'].append(processing_time)
            self.metrics['costs'].append(cost_estimate)
            
            result = ChunkingResult(
                chunks=chunks,
                total_tokens=total_tokens,
                processing_time=processing_time,
                cost_estimate=cost_estimate,
                cache_hit_rate=0.0,
                metadata={
                    "complexity": complexity.value,
                    "model_used": "qwen3-30b-a3b",
                    "thinking_enabled": complexity != ProcessingComplexity.SIMPLE
                },
                complexity_distribution={complexity.value: len(chunks)}
            )
            
            self.logger.info(f"Document processed: {len(chunks)} chunks, "
                           f"${cost_estimate:.4f} cost, {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing document: {e}")
            self.cost_tracker.end_operation(operation_id, 0.0)
            raise
    
    async def _chunk_with_qwen3(
        self, 
        visual_result: VisualParsingResult,
        complexity: ProcessingComplexity
    ) -> List[DocumentChunk]:
        """Chunk document using appropriate Qwen3 strategy"""
        
        text = visual_result.text_content
        target_chunk_size = self.config.get('base_chunk_size', 1024)
        
        # Select appropriate chain and parameters
        if complexity == ProcessingComplexity.SIMPLE:
            chain = self.simple_chain
            enable_thinking = False
            chain_input = {
                "text": text,
                "chunk_size": target_chunk_size
            }
        elif complexity == ProcessingComplexity.MEDIUM:
            chain = self.medium_chain
            enable_thinking = self.config.get('enable_thinking_for_medium', True)
            chain_input = {
                "text": text,
                "chunk_size": target_chunk_size
            }
        else:  # COMPLEX
            chain = self.complex_chain
            enable_thinking = True
            chain_input = {
                "text": text,
                "chunk_size": target_chunk_size,
                "formulas": len(visual_result.latex_formulas or []),
                "tables": len(visual_result.tables or [])
            }
        
        # Run chunking chain
        try:
            response = await asyncio.to_thread(
                chain.run,
                **chain_input
            )
            
            # Parse response into chunks
            chunk_texts = [chunk.strip() for chunk in response.split('---CHUNK---') 
                          if chunk.strip()]
            
            # Create DocumentChunk objects
            chunks = []
            for i, chunk_text in enumerate(chunk_texts):
                chunk = DocumentChunk(
                    id=f"qwen3_chunk_{i}",
                    text=chunk_text,
                    chunk_index=i,
                    start_char=text.find(chunk_text) if chunk_text in text else 0,
                    end_char=text.find(chunk_text) + len(chunk_text) if chunk_text in text else len(chunk_text),
                    complexity=complexity,
                    model_used="qwen3-30b-a3b",
                    thinking_used=enable_thinking,
                    confidence_score=0.95,  # High confidence for Qwen3
                    metadata={
                        "original_length": len(text),
                        "complexity": complexity.value,
                        "thinking_enabled": enable_thinking
                    }
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error in Qwen3 chunking: {e}")
            # Fallback to simple text splitting
            return self._fallback_chunking(text, complexity)
    
    def _fallback_chunking(self, text: str, complexity: ProcessingComplexity) -> List[DocumentChunk]:
        """Fallback chunking using RecursiveCharacterTextSplitter"""
        self.logger.warning("Using fallback chunking method")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.get('base_chunk_size', 1024),
            chunk_overlap=self.config.get('overlap', 128),
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunk_texts = text_splitter.split_text(text)
        
        chunks = []
        for i, chunk_text in enumerate(chunk_texts):
            chunk = DocumentChunk(
                id=f"fallback_chunk_{i}",
                text=chunk_text,
                chunk_index=i,
                start_char=text.find(chunk_text) if chunk_text in text else 0,
                end_char=text.find(chunk_text) + len(chunk_text) if chunk_text in text else len(chunk_text),
                complexity=complexity,
                model_used="fallback_splitter",
                thinking_used=False,
                confidence_score=0.7,  # Lower confidence for fallback
                metadata={
                    "fallback_reason": "qwen3_chunking_failed",
                    "complexity": complexity.value
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _calculate_cost(self, total_tokens: int, complexity: ProcessingComplexity) -> float:
        """Calculate processing cost based on tokens and complexity"""
        base_cost_per_1k = self.config.get('cost_per_1k_tokens', 0.012)
        
        # Complexity multipliers
        complexity_multipliers = {
            ProcessingComplexity.SIMPLE: 0.8,    # Faster, no thinking
            ProcessingComplexity.MEDIUM: 1.0,    # Standard rate
            ProcessingComplexity.COMPLEX: 1.2    # More compute for thinking
        }
        
        multiplier = complexity_multipliers.get(complexity, 1.0)
        cost = (total_tokens / 1000.0) * base_cost_per_1k * multiplier
        
        return cost
    
    def get_metrics(self) -> Dict:
        """Get comprehensive performance metrics"""
        cache_hit_rate = (
            self.metrics['cache_hits'] / 
            (self.metrics['cache_hits'] + self.metrics['cache_misses'])
            if (self.metrics['cache_hits'] + self.metrics['cache_misses']) > 0
            else 0.0
        )
        
        avg_processing_time = (
            sum(self.metrics['processing_times']) / len(self.metrics['processing_times'])
            if self.metrics['processing_times'] else 0.0
        )
        
        total_cost = sum(self.metrics['costs'])
        
        return {
            "total_documents_processed": self.metrics['total_documents'],
            "total_chunks_created": self.metrics['total_chunks'],
            "cache_hit_rate": cache_hit_rate,
            "average_processing_time": avg_processing_time,
            "total_cost": total_cost,
            "cost_per_document": total_cost / max(self.metrics['total_documents'], 1),
            "cost_per_1k_pages": total_cost * (1000 / max(self.metrics['total_documents'], 1))
        }


# Factory function for pipeline integration
def create_stage2_processor(config_path: str) -> Qwen3OptimizedChunker:
    """Factory function to create Stage 2 processor"""
    return Qwen3OptimizedChunker(config_path)


# Async processing function for integration
async def process_stage2(
    visual_result: VisualParsingResult, 
    config_path: str = "ultra_cost_optimized_pipeline/config/pipeline_config.yaml"
) -> ChunkingResult:
    """Main entry point for Stage 2 processing"""
    processor = create_stage2_processor(config_path)
    return await processor.process_document(visual_result)


if __name__ == "__main__":
    # Test and validation
    import asyncio
    
    async def test_stage2():
        """Test Stage 2 processing"""
        # Create test input
        test_result = VisualParsingResult(
            text_content="""
            This is a test document for the Qwen3 chunking pipeline.
            It contains multiple paragraphs and some mathematical content.
            
            The main theorem states that for any function f(x) = x^2 + 3x + 2,
            the derivative is f'(x) = 2x + 3.
            
            This demonstrates both simple text processing and mathematical content
            that should trigger different complexity levels in our pipeline.
            """,
            latex_formulas=["f(x) = x^2 + 3x + 2", "f'(x) = 2x + 3"],
            tables=[],
            layout_info={},
            confidence_scores={"overall": 0.95},
            processing_time=1.0,
            cost_estimate=0.01
        )
        
        # Process with Stage 2
        result = await process_stage2(test_result)
        
        print(f"Processing complete:")
        print(f"- Chunks created: {len(result.chunks)}")
        print(f"- Total tokens: {result.total_tokens}")
        print(f"- Processing time: {result.processing_time:.2f}s")
        print(f"- Cost estimate: ${result.cost_estimate:.4f}")
        print(f"- Cache hit rate: {result.cache_hit_rate:.2f}")
        
        for i, chunk in enumerate(result.chunks):
            print(f"\nChunk {i}:")
            print(f"- Text: {chunk.text[:100]}...")
            print(f"- Complexity: {chunk.complexity.value}")
            print(f"- Thinking used: {chunk.thinking_used}")
    
    asyncio.run(test_stage2()) 