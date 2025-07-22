"""
Stage 1: Visual Parsing with RunPod Integration
Modified version that uses RunPod models for all AI processing while keeping local orchestration
"""

import os
import time
import logging
import asyncio
from typing import List, Dict, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pdf2image
import yaml

# Import existing utilities
from .utils.cost_tracking import CostTracker
from .runpod_client import RunPodModelFactory, check_runpod_config

# Import existing PDF converter and system checker
from .stage1_visual_parsing import (
    RobustPDFConverter, 
    SystemRequirementsChecker, 
    VisualParsingResult
)

logger = logging.getLogger(__name__)


class VisualParsingStageRunPod:
    """Visual parsing stage using RunPod for model inference"""
    
    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        # Check if RunPod is configured
        if not check_runpod_config():
            raise FileNotFoundError(
                "RunPod configuration not found. Please run 'bash runpod_setup.sh' first to deploy models."
            )
        
        # Load configuration
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            logging.error(f"Failed to load config from {config_path}: {e}")
            self.config = self._get_default_config()
        
        self.visual_config = self.config.get('visual_parsing', {})
        
        # Initialize local components (PDF processing stays local)
        self.pdf_converter = RobustPDFConverter(self.visual_config.get('preprocessing', {}))
        
        # Initialize RunPod components (AI models on RunPod)
        try:
            self.ocr = RunPodModelFactory.create_paddleocr(self.visual_config.get('paddleocr', {}))
            self.nougat = RunPodModelFactory.create_nougat(self.visual_config.get('nougat', {}))
            self.layoutlm = RunPodModelFactory.create_layoutlm(self.visual_config.get('layoutlm', {}))
            logger.info("✅ RunPod models initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RunPod models: {e}")
            raise
        
        # Cost tracking
        self.cost_tracker = CostTracker()
        
        # Thread pool for parallel processing (local operations)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # System info for debugging
        self.system_info = SystemRequirementsChecker.get_system_info()
        logger.info(f"System info: {self.system_info}")
        
        logger.info("Visual parsing stage with RunPod initialized")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration if config file is missing"""
        return {
            'visual_parsing': {
                'paddleocr': {
                    'use_angle_cls': True,
                    'lang': 'en',
                },
                'nougat': {
                    'model_name': 'facebook/nougat-small',
                    'max_length': 4096
                },
                'layoutlm': {
                    'model_name': 'microsoft/layoutlmv3-base',
                },
                'preprocessing': {
                    'dpi': 150,
                    'max_image_size': 2048,
                    'convert_grayscale': False,
                    'enhance_contrast': True
                }
            }
        }
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Optimize image for processing (runs locally)"""
        config = self.visual_config.get('preprocessing', {})
        
        try:
            # Resize if too large
            max_size = config.get('max_image_size', 2048)
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Enhance contrast if enabled
            if config.get('enhance_contrast', True):
                from PIL import ImageEnhance
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.2)
        
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}")
            # Return original image if preprocessing fails
        
        return image
    
    def convert_pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """Convert PDF to images (runs locally)"""
        images, errors = self.pdf_converter.convert_pdf_to_images(pdf_path)
        
        if errors:
            for error in errors:
                logger.error(f"PDF conversion error: {error}")
        
        if images:
            return [self.preprocess_image(img) for img in images]
        else:
            logger.error(f"Failed to convert PDF: {pdf_path}")
            return []
    
    async def process_single_page(self, image: Image.Image, page_num: int) -> VisualParsingResult:
        """Process a single page through RunPod models"""
        start_time = time.time()
        
        # Convert PIL to numpy for OCR
        image_np = np.array(image)
        
        # Run all RunPod components in parallel
        async def safe_ocr_extraction():
            try:
                return await self.ocr.extract_text(image_np)
            except Exception as e:
                logger.error(f"RunPod OCR extraction failed: {e}")
                return "", 0.0
        
        async def safe_latex_extraction():
            try:
                return await self.nougat.extract_latex(image)
            except Exception as e:
                logger.error(f"RunPod LaTeX extraction failed: {e}")
                return [], 0.0
        
        async def safe_layout_extraction():
            try:
                return await self.layoutlm.extract_layout(image, "")
            except Exception as e:
                logger.error(f"RunPod Layout extraction failed: {e}")
                return {}, 0.0
        
        tasks = [
            safe_ocr_extraction(),
            safe_latex_extraction(), 
            safe_layout_extraction()
        ]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Unpack results with safe defaults
            text_content, text_confidence = results[0] if not isinstance(results[0], Exception) else ("", 0.0)
            latex_formulas, latex_confidence = results[1] if not isinstance(results[1], Exception) else ([], 0.0)
            layout_info, layout_confidence = results[2] if not isinstance(results[2], Exception) else ({}, 0.0)
            
            # Calculate costs (RunPod pricing)
            total_cost = (
                self.ocr.cost_per_page +
                self.nougat.cost_per_page +
                self.layoutlm.cost_per_page
            )
            
            processing_time = time.time() - start_time
            
            # Track costs
            self.cost_tracker.add_cost("visual_parsing_runpod", total_cost)
            
            return VisualParsingResult(
                text_content=text_content,
                latex_formulas=latex_formulas,
                tables=[],  # Tables would be extracted in a full implementation
                layout_info=layout_info,
                confidence_scores={
                    "text": text_confidence,
                    "latex": latex_confidence,
                    "layout": layout_confidence
                },
                processing_time=processing_time,
                cost_estimate=total_cost
            )
            
        except Exception as e:
            logger.error(f"RunPod page processing failed: {e}")
            return VisualParsingResult(
                text_content="",
                latex_formulas=[],
                tables=[],
                layout_info={},
                confidence_scores={},
                processing_time=time.time() - start_time,
                cost_estimate=0.0
            )
    
    async def process_document(self, document_path: str) -> List[VisualParsingResult]:
        """Process entire document with RunPod models"""
        logger.info(f"Processing document with RunPod: {document_path}")
        
        # Validate input (local operation)
        if not SystemRequirementsChecker.check_file_exists(document_path):
            logger.error(f"Document file not found: {document_path}")
            return []
        
        try:
            # Convert PDF to images (local operation)
            if document_path.lower().endswith('.pdf'):
                images = self.convert_pdf_to_images(document_path)
            else:
                # Handle single image
                try:
                    images = [Image.open(document_path)]
                except Exception as e:
                    logger.error(f"Failed to open image file: {e}")
                    return []
            
            if not images:
                logger.error("No images to process")
                return []
            
            # Process all pages with RunPod models
            tasks = [
                self.process_single_page(image, i)
                for i, image in enumerate(images)
            ]
            
            results = await asyncio.gather(*tasks)
            
            # Log summary
            total_cost = sum(result.cost_estimate for result in results)
            total_time = sum(result.processing_time for result in results)
            avg_confidence = np.mean([
                np.mean(list(result.confidence_scores.values()))
                for result in results
                if result.confidence_scores
            ]) if results else 0.0
            
            logger.info(f"Processed {len(results)} pages with RunPod")
            logger.info(f"Total cost: ${total_cost:.6f}")
            logger.info(f"Total time: {total_time:.2f}s")
            logger.info(f"Average confidence: {avg_confidence:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            return []
    
    def get_cost_report(self) -> Dict:
        """Get detailed cost breakdown"""
        return self.cost_tracker.get_report()
    
    def get_initialization_errors(self) -> List[str]:
        """Get all initialization errors for debugging"""
        errors = []
        if hasattr(self.ocr, 'initialization_errors'):
            errors.extend(self.ocr.initialization_errors)
        if hasattr(self.nougat, 'initialization_errors'):
            errors.extend(self.nougat.initialization_errors)
        return errors
    
    def cleanup(self):
        """Clean up resources"""
        self.executor.shutdown(wait=True)
        logger.info("Visual parsing stage cleanup completed") 