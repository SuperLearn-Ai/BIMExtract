"""
RunPod Client Integration for Ultra-Cost-Optimized Pipeline
Provides seamless integration with RunPod endpoints while maintaining local pipeline architecture
"""

import os
import time
import logging
import asyncio
import aiohttp
import base64
import json
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import numpy as np
from PIL import Image
import io
import yaml

logger = logging.getLogger(__name__)

class RunPodAPIClient:
    """RunPod API client with retry logic and error handling"""
    
    def __init__(self, endpoint_id: str, api_key: str):
        self.endpoint_id = endpoint_id
        self.api_key = api_key
        self.base_url = f"https://api.runpod.ai/v2/{endpoint_id}"
        self.session = None
        
    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def _make_request(self, endpoint: str, data: Dict, max_retries: int = 3) -> Dict:
        """Make HTTP request to RunPod with retries"""
        session = await self._get_session()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        url = f"{self.base_url}/{endpoint}"
        
        for attempt in range(max_retries):
            try:
                async with session.post(url, json=data, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result
                    else:
                        error_text = await response.text()
                        logger.error(f"RunPod API error {response.status}: {error_text}")
                        
            except Exception as e:
                logger.error(f"RunPod request attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    
        raise Exception(f"RunPod API failed after {max_retries} attempts")
    
    async def run_inference(self, input_data: Dict) -> Dict:
        """Run inference on RunPod"""
        # Start the job
        response = await self._make_request("run", {"input": input_data})
        job_id = response.get("id")
        
        if not job_id:
            raise Exception("Failed to start RunPod job")
        
        # Poll for completion
        while True:
            status_response = await self._make_request(f"status/{job_id}", {})
            status = status_response.get("status")
            
            if status == "COMPLETED":
                return status_response.get("output", {})
            elif status == "FAILED":
                error = status_response.get("error", "Unknown error")
                raise Exception(f"RunPod job failed: {error}")
            
            await asyncio.sleep(2)  # Poll every 2 seconds
    
    async def cleanup(self):
        """Clean up session"""
        if self.session:
            await self.session.close()


class RunPodPaddleOCR:
    """RunPod-based PaddleOCR replacement"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Load RunPod configuration
        runpod_config = self._load_runpod_config()
        ocr_config = runpod_config["runpod"]["endpoints"]["paddleocr"]
        
        self.client = RunPodAPIClient(
            ocr_config["endpoint_id"],
            runpod_config["runpod"]["api_key"]
        )
        self.cost_per_page = ocr_config["cost_per_request"]
        self.initialization_errors = []
    
    def _load_runpod_config(self) -> Dict:
        """Load RunPod configuration"""
        config_path = Path("config/runpod_config.yaml")
        if not config_path.exists():
            raise FileNotFoundError("RunPod configuration not found. Run deployment script first.")
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def initialize_ocr(self) -> bool:
        """Initialize RunPod OCR (always returns True as it's cloud-based)"""
        logger.info("✅ RunPod PaddleOCR initialized successfully")
        return True
    
    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy image to base64 string"""
        pil_image = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    async def extract_text(self, image: np.ndarray) -> Tuple[str, float]:
        """Extract text using RunPod PaddleOCR"""
        start_time = time.time()
        
        try:
            # Convert image to base64
            image_b64 = self._image_to_base64(image)
            
            # Prepare input for RunPod
            input_data = {
                "image": image_b64,
                "use_angle_cls": self.config.get('use_angle_cls', True),
                "lang": self.config.get('lang', 'en')
            }
            
            # Run inference
            result = await self.client.run_inference(input_data)
            
            # Parse results
            text_content = result.get("text", "")
            confidence = result.get("confidence", 0.0)
            
            processing_time = time.time() - start_time
            logger.debug(f"RunPod OCR extraction took {processing_time:.2f}s")
            
            return text_content, confidence
            
        except Exception as e:
            logger.error(f"RunPod OCR extraction failed: {e}")
            return "", 0.0


class RunPodNougat:
    """RunPod-based Nougat LaTeX extraction"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Load RunPod configuration
        runpod_config = self._load_runpod_config()
        nougat_config = runpod_config["runpod"]["endpoints"]["nougat"]
        
        self.client = RunPodAPIClient(
            nougat_config["endpoint_id"],
            runpod_config["runpod"]["api_key"]
        )
        self.cost_per_page = nougat_config["cost_per_request"]
        self.initialization_errors = []
    
    def _load_runpod_config(self) -> Dict:
        """Load RunPod configuration"""
        config_path = Path("config/runpod_config.yaml")
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_model(self):
        """Load model (no-op for RunPod)"""
        pass
    
    async def extract_latex(self, image: Image.Image) -> Tuple[List[str], float]:
        """Extract LaTeX formulas using RunPod Nougat"""
        try:
            # Convert PIL image to base64
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            buffer.seek(0)
            image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Prepare input
            input_data = {
                "image": image_b64,
                "max_length": self.config.get('max_length', 4096)
            }
            
            # Run inference
            result = await self.client.run_inference(input_data)
            
            # Parse LaTeX formulas
            formulas = result.get("formulas", [])
            confidence = result.get("confidence", 0.9)
            
            return formulas, confidence
            
        except Exception as e:
            logger.error(f"RunPod Nougat extraction failed: {e}")
            return [], 0.0


class RunPodLayoutLM:
    """RunPod-based LayoutLM document understanding"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Load RunPod configuration
        runpod_config = self._load_runpod_config()
        layoutlm_config = runpod_config["runpod"]["endpoints"]["layoutlm"]
        
        self.client = RunPodAPIClient(
            layoutlm_config["endpoint_id"],
            runpod_config["runpod"]["api_key"]
        )
        self.cost_per_page = layoutlm_config["cost_per_request"]
    
    def _load_runpod_config(self) -> Dict:
        """Load RunPod configuration"""
        config_path = Path("config/runpod_config.yaml")
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_model(self):
        """Load model (no-op for RunPod)"""
        pass
    
    async def extract_layout(self, image: Image.Image, text: str) -> Tuple[Dict, float]:
        """Extract layout information using RunPod LayoutLM"""
        try:
            # Convert image to base64
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            buffer.seek(0)
            image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Prepare input
            input_data = {
                "image": image_b64,
                "text": text
            }
            
            # Run inference
            result = await self.client.run_inference(input_data)
            
            # Parse layout information
            layout_info = result.get("layout", {})
            confidence = result.get("confidence", 0.85)
            
            return layout_info, confidence
            
        except Exception as e:
            logger.error(f"RunPod LayoutLM extraction failed: {e}")
            return {}, 0.0


class RunPodLlama:
    """RunPod-based Llama model for text generation"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Load RunPod configuration
        runpod_config = self._load_runpod_config()
        llama_config = runpod_config["runpod"]["endpoints"]["llama"]
        
        self.client = RunPodAPIClient(
            llama_config["endpoint_id"],
            runpod_config["runpod"]["api_key"]
        )
        self.cost_per_token = llama_config["cost_per_request"] / 1000  # Estimate tokens
    
    def _load_runpod_config(self) -> Dict:
        """Load RunPod configuration"""
        config_path = Path("config/runpod_config.yaml")
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_model(self):
        """Load model (no-op for RunPod)"""
        pass
    
    async def generate_text(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> Tuple[str, int]:
        """Generate text using RunPod Llama"""
        try:
            input_data = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "stop": ["</s>", "\n\n"]
            }
            
            result = await self.client.run_inference(input_data)
            
            generated_text = result.get("text", "")
            tokens_used = result.get("tokens_used", len(generated_text.split()))
            
            return generated_text, tokens_used
            
        except Exception as e:
            logger.error(f"RunPod Llama generation failed: {e}")
            return "", 0
    
    async def enhance_chunk(self, chunk_text: str) -> Tuple[str, float]:
        """Enhance chunk text using RunPod Llama"""
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
        
        enhanced_text, tokens = await self.generate_text(prompt, max_tokens=256)
        cost = tokens * self.cost_per_token
        
        return enhanced_text, cost


class RunPodEmbeddings:
    """RunPod-based embedding model"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Load RunPod configuration
        runpod_config = self._load_runpod_config()
        embeddings_config = runpod_config["runpod"]["endpoints"]["embeddings"]
        
        self.client = RunPodAPIClient(
            embeddings_config["endpoint_id"],
            runpod_config["runpod"]["api_key"]
        )
        self.cost_per_embedding = embeddings_config["cost_per_request"]
    
    def _load_runpod_config(self) -> Dict:
        """Load RunPod configuration"""
        config_path = Path("config/runpod_config.yaml")
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_model(self):
        """Load model (no-op for RunPod)"""
        pass
    
    async def encode_batch(self, texts: List[str], batch_size: int = 32) -> Tuple[np.ndarray, float]:
        """Encode texts in batches using RunPod"""
        try:
            input_data = {
                "texts": texts,
                "normalize_embeddings": self.config.get('normalize_embeddings', True)
            }
            
            result = await self.client.run_inference(input_data)
            
            embeddings = np.array(result.get("embeddings", []))
            cost = len(texts) * self.cost_per_embedding
            
            return embeddings, cost
            
        except Exception as e:
            logger.error(f"RunPod embedding encoding failed: {e}")
            return np.array([]), 0.0
    
    async def encode_single(self, text: str) -> Tuple[np.ndarray, float]:
        """Encode single text using RunPod"""
        embeddings, cost = await self.encode_batch([text], batch_size=1)
        return embeddings[0] if len(embeddings) > 0 else np.array([]), cost


class RunPodModelFactory:
    """Factory for creating RunPod model instances"""
    
    @staticmethod
    def create_paddleocr(config: Dict) -> RunPodPaddleOCR:
        """Create RunPod PaddleOCR instance"""
        return RunPodPaddleOCR(config)
    
    @staticmethod
    def create_nougat(config: Dict) -> RunPodNougat:
        """Create RunPod Nougat instance"""
        return RunPodNougat(config)
    
    @staticmethod
    def create_layoutlm(config: Dict) -> RunPodLayoutLM:
        """Create RunPod LayoutLM instance"""
        return RunPodLayoutLM(config)
    
    @staticmethod
    def create_llama(config: Dict) -> RunPodLlama:
        """Create RunPod Llama instance"""
        return RunPodLlama(config)
    
    @staticmethod
    def create_embeddings(config: Dict) -> RunPodEmbeddings:
        """Create RunPod Embeddings instance"""
        return RunPodEmbeddings(config)


def check_runpod_config() -> bool:
    """Check if RunPod configuration exists"""
    config_path = Path("config/runpod_config.yaml")
    return config_path.exists()


def load_runpod_config() -> Dict:
    """Load RunPod configuration"""
    config_path = Path("config/runpod_config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(
            "RunPod configuration not found. Please run 'bash runpod_setup.sh' first."
        )
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f) 