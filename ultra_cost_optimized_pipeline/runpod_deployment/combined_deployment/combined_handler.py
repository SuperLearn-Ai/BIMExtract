#!/usr/bin/env python3
"""
Combined RunPod Handler for Ultra-Cost-Optimized Pipeline
Serves both Qwen3 LLM and Visual Parsing on the same pod
"""

import runpod
import torch
import logging
import json
import base64
import io
import numpy as np
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
import paddleocr
import cv2

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model containers
qwen3_model = None
qwen3_tokenizer = None
ocr_model = None

def load_qwen3_model():
    """Load Qwen3 model globally"""
    global qwen3_model, qwen3_tokenizer
    if qwen3_model is None:
        logger.info("Loading Qwen3 model...")
        model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
        qwen3_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        qwen3_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            load_in_4bit=True
        )
        logger.info("✅ Qwen3 model loaded successfully")
    return qwen3_model, qwen3_tokenizer

def load_visual_models():
    """Load visual parsing models globally"""
    global ocr_model
    if ocr_model is None:
        logger.info("Loading PaddleOCR model...")
        ocr_model = paddleocr.PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        logger.info("✅ PaddleOCR model loaded successfully")
    return ocr_model

def process_qwen3_request(input_data):
    """Process Qwen3 LLM requests"""
    try:
        model, tokenizer = load_qwen3_model()
        
        text = input_data.get("text", "")
        thinking_mode = input_data.get("thinking_mode", False)
        max_tokens = input_data.get("max_tokens", 2048)
        temperature = input_data.get("temperature", 0.1)
        
        # Format prompt for thinking mode
        if thinking_mode:
            prompt = f"<|thinking|>\nLet me analyze this document chunk carefully:\n\n{text}\n</|thinking|>\n\nBased on my analysis:"
        else:
            prompt = text
        
        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        return {
            "response": response,
            "thinking_mode": thinking_mode,
            "token_count": len(outputs[0]) - inputs["input_ids"].shape[1],
            "service": "qwen3_llm"
        }
        
    except Exception as e:
        logger.error(f"Qwen3 processing error: {e}")
        return {"error": str(e), "service": "qwen3_llm"}

def process_visual_request(input_data):
    """Process visual parsing requests"""
    try:
        ocr_model = load_visual_models()
        
        image_data = input_data.get("image", "")
        parse_type = input_data.get("type", "ocr")  # ocr, nougat, layoutlm
        
        # Decode image
        if image_data.startswith("data:"):
            image_data = image_data.split(",")[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)
        
        # Process based on type
        if parse_type == "ocr":
            results = ocr_model.ocr(image_np, cls=True)
            
            # Format OCR results
            parsed_results = []
            for line in results:
                if line:  # Check if line is not None
                    for word_info in line:
                        box, (text, confidence) = word_info
                        parsed_results.append({
                            "text": text,
                            "confidence": confidence,
                            "bbox": box
                        })
            
            return {
                "results": parsed_results,
                "type": "ocr",
                "total_words": len(parsed_results),
                "service": "visual_parsing"
            }
        
        else:
            return {"error": f"Parse type '{parse_type}' not implemented", "service": "visual_parsing"}
            
    except Exception as e:
        logger.error(f"Visual processing error: {e}")
        return {"error": str(e), "service": "visual_parsing"}

def handler(job):
    """Main RunPod handler - routes requests to appropriate service"""
    try:
        input_data = job.get("input", {})
        service_type = input_data.get("service", "auto")
        
        # Auto-detect service type if not specified
        if service_type == "auto":
            if "image" in input_data:
                service_type = "visual_parsing"
            elif "text" in input_data:
                service_type = "qwen3_llm"
            else:
                return {"error": "Unable to determine service type. Please specify 'service' parameter."}
        
        # Route to appropriate service
        if service_type == "qwen3_llm":
            return process_qwen3_request(input_data)
        elif service_type == "visual_parsing":
            return process_visual_request(input_data)
        else:
            return {"error": f"Unknown service type: {service_type}"}
            
    except Exception as e:
        logger.error(f"Handler error: {e}")
        return {"error": str(e)}

# Health check endpoint
def health_check():
    """Health check for the combined service"""
    try:
        # Check if models can be loaded
        qwen3_status = "loaded" if qwen3_model is not None else "not_loaded"
        ocr_status = "loaded" if ocr_model is not None else "not_loaded"
        
        return {
            "status": "healthy",
            "services": {
                "qwen3_llm": qwen3_status,
                "visual_parsing": ocr_status
            },
            "gpu_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

# Start the RunPod serverless handler
if __name__ == "__main__":
    logger.info("🚀 Starting Combined Ultra-Pipeline Handler")
    logger.info("   🤖 Services: Qwen3 LLM + Visual Parsing")
    logger.info("   🎯 Cost Target: $0.15 per 1K pages")
    
    runpod.serverless.start({
        "handler": handler,
        "return_aggregate_stream": True
    }) 