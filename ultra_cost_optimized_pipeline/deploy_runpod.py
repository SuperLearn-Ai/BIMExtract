#!/usr/bin/env python3
"""
Simplified RunPod Deployment for Ultra-Cost-Optimized Pipeline
Consolidated deployment approach for current RunPod interface
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleRunPodDeployer:
    """Simplified RunPod deployment with essential configurations"""
    
    def __init__(self, api_key: Optional[str] = None, existing_pod_id: Optional[str] = None):
        self.api_key = api_key or os.getenv('RUNPOD_API_KEY')
        self.existing_pod_id = existing_pod_id
        if not self.api_key:
            raise ValueError("RUNPOD_API_KEY not found. Set via environment variable.")
    
    def get_deployment_configs(self) -> Dict:
        """Essential deployment configurations for pipeline models"""
        return {
            "qwen3_llm": {
                "name": "ultra-pipeline-qwen3",
                "template_id": "runpod-pytorch-2.1",
                "gpu_type": "NVIDIA RTX A5000",
                "container_disk_gb": 60,
                "docker_args": "--shm-size=16g",
                "env_vars": {
                    "MODEL_NAME": "Qwen/Qwen2.5-Coder-32B-Instruct",
                    "QUANTIZATION": "A3B",
                    "THINKING_MODE": "enabled",
                    "MAX_TOKENS": "8192",
                    "TEMPERATURE": "0.1"
                },
                "startup_script": """
# Install dependencies
pip install --no-cache-dir transformers>=4.45.0 torch>=2.1.0 accelerate>=0.25.0 bitsandbytes>=0.41.0
pip install --no-cache-dir vllm>=0.2.0 runpod>=1.5.0

# Download and cache model
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = 'Qwen/Qwen2.5-Coder-32B-Instruct'
print('Downloading Qwen3 model...')
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    trust_remote_code=True,
    load_in_4bit=True
)
print('Model downloaded and cached successfully!')
"

# Start handler
python handler.py
                """
            },
            "visual_parsing": {
                "name": "ultra-pipeline-visual",
                "template_id": "runpod-pytorch-2.1",
                "gpu_type": "NVIDIA RTX A5000",
                "container_disk_gb": 40,
                "env_vars": {
                    "VISUAL_MODE": "multi",
                    "OCR_LANG": "en",
                    "NOUGAT_MODEL": "0.1.0-base",
                    "LAYOUTLM_MODEL": "microsoft/layoutlm-base-uncased"
                },
                "startup_script": """
# Install visual parsing dependencies
pip install --no-cache-dir paddlepaddle-gpu>=2.5.2 paddleocr>=2.7.3
pip install --no-cache-dir nougat-ocr>=0.1.17 layoutparser[layoutmodels,tesseract,ocr]>=0.3.4
pip install --no-cache-dir transformers>=4.36.0 torch>=2.1.0 opencv-python>=4.8.0
pip install --no-cache-dir Pillow>=10.0.0 runpod>=1.5.0

# Initialize models
python -c "
import paddleocr
from nougat import predict
from transformers import LayoutLMTokenizer

print('Initializing PaddleOCR...')
ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

print('Downloading Nougat model...')
nougat_model = predict.download_model()

print('Downloading LayoutLM...')
layoutlm_tokenizer = LayoutLMTokenizer.from_pretrained('microsoft/layoutlm-base-uncased')

print('All visual models initialized!')
"

python handler.py
                """
            }
        }
    
    def generate_combined_handler_code(self) -> str:
        """Generate combined RunPod handler code for both services on one pod"""
        return '''
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
            prompt = f"<|thinking|>\\nLet me analyze this document chunk carefully:\\n\\n{text}\\n</|thinking|>\\n\\nBased on my analysis:"
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
        parse_type = input_data.get("type", "ocr")
        
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
                if line:
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

if __name__ == "__main__":
    logger.info("🚀 Starting Combined Ultra-Pipeline Handler")
    logger.info("   🤖 Services: Qwen3 LLM + Visual Parsing")
    logger.info("   🎯 Cost Target: $0.15 per 1K pages")
    
    runpod.serverless.start({
        "handler": handler,
        "return_aggregate_stream": True
    })
'''

    def generate_handler_code(self, model_type: str) -> str:
        """Generate RunPod handler code for each model type (legacy method)"""
        if model_type == "qwen3_llm":
            return '''
import runpod
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import logging

# Initialize model globally
model = None
tokenizer = None

def load_model():
    global model, tokenizer
    if model is None:
        model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            load_in_4bit=True
        )
    return model, tokenizer

def handler(job):
    """RunPod handler for Qwen3 LLM processing"""
    try:
        model, tokenizer = load_model()
        
        # Extract inputs
        input_data = job["input"]
        text = input_data.get("text", "")
        thinking_mode = input_data.get("thinking_mode", False)
        max_tokens = input_data.get("max_tokens", 2048)
        temperature = input_data.get("temperature", 0.1)
        
        # Format prompt for thinking mode
        if thinking_mode:
            prompt = f"<|thinking|>\\nLet me analyze this document chunk carefully:\\n\\n{text}\\n</|thinking|>\\n\\nBased on my analysis:"
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
            "token_count": len(outputs[0]) - inputs["input_ids"].shape[1]
        }
        
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
'''
        
        elif model_type == "visual_parsing":
            return '''
import runpod
import paddleocr
import cv2
import numpy as np
from PIL import Image
import base64
import io
import json

# Initialize models globally
ocr_model = None
nougat_model = None

def load_models():
    global ocr_model, nougat_model
    if ocr_model is None:
        ocr_model = paddleocr.PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    
    # Nougat initialization would go here if needed
    return ocr_model, nougat_model

def handler(job):
    """RunPod handler for visual parsing"""
    try:
        ocr_model, _ = load_models()
        
        # Extract inputs
        input_data = job["input"]
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
                "total_words": len(parsed_results)
            }
        
        else:
            return {"error": f"Parse type '{parse_type}' not implemented"}
            
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
'''
        
        return "# Handler code not available for this model type"
    
    def deploy_to_existing_pod(self, pod_id: str = "jeqrwyd0hbl40c"):
        """Deploy combined pipeline to existing RunPod pod"""
        output_dir = "combined_deployment"
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        logger.info(f"🚀 Deploying to existing pod: {pod_id}")
        
        # Create combined handler
        handler_code = self.generate_combined_handler_code()
        (output_path / "handler.py").write_text(handler_code)
        
        # Create combined requirements
        requirements = """transformers>=4.45.0
torch>=2.1.0
accelerate>=0.25.0
bitsandbytes>=0.41.0
paddlepaddle-gpu>=2.5.2
paddleocr>=2.7.3
opencv-python>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0
runpod>=1.5.0"""
        (output_path / "requirements.txt").write_text(requirements)
        
        # Create setup script
        setup_script = """#!/bin/bash
# Combined Setup Script for Ultra-Cost-Optimized Pipeline
set -e

echo "🚀 Setting up Ultra-Cost-Optimized Pipeline"
echo "   🤖 Services: Qwen3 LLM + Visual Parsing"
echo "   🎯 Target: $0.15 per 1K pages"

# Install dependencies
echo "📦 Installing dependencies..."
pip install --no-cache-dir -r requirements.txt

# Download models
echo "🤖 Pre-downloading models..."
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import paddleocr

# Download Qwen3
model_name = 'Qwen/Qwen2.5-Coder-32B-Instruct'
print('Downloading Qwen3...')
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map='auto', 
    trust_remote_code=True, load_in_4bit=True
)

# Initialize PaddleOCR
print('Initializing PaddleOCR...')
ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
print('✅ All models ready!')
"

echo "✅ Setup complete! Starting handler..."
python handler.py"""
        (output_path / "setup.sh").write_text(setup_script)
        
        # Create deployment instructions for existing pod
        instructions = f"""# Deploy to Existing RunPod Pod: {pod_id}

## Files Created
- handler.py: Combined Qwen3 + Visual Parsing handler
- requirements.txt: All dependencies
- setup.sh: Installation script

## Deployment Steps

### 1. Upload Files to Pod
```bash
# Upload to pod {pod_id}
runpodctl send {pod_id} {output_dir}/handler.py /workspace/
runpodctl send {pod_id} {output_dir}/requirements.txt /workspace/
runpodctl send {pod_id} {output_dir}/setup.sh /workspace/
```

### 2. Setup and Start Service
```bash
# Connect to pod and run setup
runpodctl exec {pod_id} "cd /workspace && chmod +x setup.sh && ./setup.sh"
```

### 3. Get Pod URL
The pod will be available at: https://{pod_id}-8000.proxy.runpod.net

### 4. Test Endpoints

**Qwen3 LLM Test:**
```bash
curl -X POST https://{pod_id}-8000.proxy.runpod.net/runsync \\
  -H "Content-Type: application/json" \\
  -d '{{"input": {{"service": "qwen3_llm", "text": "Analyze this document chunk", "thinking_mode": true}}}}'
```

**Visual Parsing Test:**
```bash
curl -X POST https://{pod_id}-8000.proxy.runpod.net/runsync \\
  -H "Content-Type: application/json" \\
  -d '{{"input": {{"service": "visual_parsing", "image": "base64_image_data", "type": "ocr"}}}}'
```

### 5. Update Local Configuration
Create config/runpod_endpoints.yaml:
```yaml
endpoints:
  combined_service: "https://{pod_id}-8000.proxy.runpod.net"
  
api_key: "your-runpod-api-key"
```
"""
        (output_path / "DEPLOYMENT_INSTRUCTIONS.md").write_text(instructions)
        
        logger.info(f"✅ Combined deployment files created in: {output_dir}/")
        logger.info(f"📖 Ready to deploy to pod: {pod_id}")
        
        return output_path, pod_id

    def create_deployment_files(self, output_dir: str = "runpod_deployment"):
        """Create deployment files for RunPod (legacy method for new pods)"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        configs = self.get_deployment_configs()
        
        # Create configuration files
        for model_name, config in configs.items():
            model_dir = output_path / model_name
            model_dir.mkdir(exist_ok=True)
            
            # Create handler.py
            handler_code = self.generate_handler_code(model_name)
            (model_dir / "handler.py").write_text(handler_code)
            
            # Create requirements.txt
            if model_name == "qwen3_llm":
                requirements = """
transformers>=4.45.0
torch>=2.1.0
accelerate>=0.25.0
bitsandbytes>=0.41.0
runpod>=1.5.0
"""
            else:
                requirements = """
paddlepaddle-gpu>=2.5.2
paddleocr>=2.7.3
opencv-python>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0
runpod>=1.5.0
"""
            (model_dir / "requirements.txt").write_text(requirements.strip())
            
            # Create config.json
            config_json = {
                "name": config["name"],
                "gpu_type": config["gpu_type"],
                "container_disk_gb": config["container_disk_gb"],
                "env_vars": config["env_vars"]
            }
            (model_dir / "config.json").write_text(json.dumps(config_json, indent=2))
            
            # Create startup script
            (model_dir / "startup.sh").write_text(config["startup_script"])
        
        # Create deployment instructions
        instructions = f"""
# RunPod Deployment Instructions

## 1. Prepare Files
Files created in: {output_dir}/

## 2. Deploy via RunPod Web Interface

### For Qwen3 LLM:
1. Go to RunPod.io → Create Pod
2. Select: NVIDIA RTX A5000 
3. Template: PyTorch 2.1
4. Container Disk: 60GB
5. Upload files from: {output_dir}/qwen3_llm/
6. Set environment variables from config.json
7. Deploy and note endpoint URL

### For Visual Parsing:
1. Go to RunPod.io → Create Pod  
2. Select: NVIDIA RTX A5000
3. Template: PyTorch 2.1
4. Container Disk: 40GB
5. Upload files from: {output_dir}/visual_parsing/
6. Set environment variables from config.json
7. Deploy and note endpoint URL

## 3. Update Pipeline Configuration
Create config/runpod_endpoints.yaml:

```yaml
endpoints:
  qwen3_llm: "https://your-qwen3-pod-id-runpod.io"
  visual_parsing: "https://your-visual-pod-id-runpod.io"
  
api_key: "your-runpod-api-key"
```

## 4. Test Deployment
Run: python test_runpod_deployment.py
"""
        
        (output_path / "DEPLOYMENT_INSTRUCTIONS.md").write_text(instructions)
        
        logger.info(f"✅ Deployment files created in: {output_dir}/")
        logger.info("📖 See DEPLOYMENT_INSTRUCTIONS.md for next steps")
        
        return output_path

def main():
    """Main deployment function"""
    import sys
    
    # Check if user wants to deploy to existing pod
    use_existing_pod = "--existing-pod" in sys.argv or len(sys.argv) > 1 and "jeqrwyd0hbl40c" in sys.argv
    
    try:
        deployer = SimpleRunPodDeployer()
        
        if use_existing_pod:
            # Deploy to existing "Superlearn Ai" pod
            deployment_path, pod_id = deployer.deploy_to_existing_pod("jeqrwyd0hbl40c")
            
            print(f"""
🚀 Deployment Ready for Existing Pod!

📁 Files created in: {deployment_path}
🎯 Target Pod: {pod_id} (Superlearn Ai)
📖 Instructions: {deployment_path}/DEPLOYMENT_INSTRUCTIONS.md

Next steps:
1. Upload files to pod: runpodctl send {pod_id} {deployment_path}/handler.py /workspace/
2. Upload requirements: runpodctl send {pod_id} {deployment_path}/requirements.txt /workspace/  
3. Upload setup script: runpodctl send {pod_id} {deployment_path}/setup.sh /workspace/
4. Run setup: runpodctl exec {pod_id} "cd /workspace && chmod +x setup.sh && ./setup.sh"
5. Access at: https://{pod_id}-8000.proxy.runpod.net

💡 Combined service handles both Qwen3 LLM and Visual Parsing!
            """)
        else:
            # Create files for new pods
            deployment_path = deployer.create_deployment_files()
            
            print(f"""
🚀 RunPod Deployment Files Ready!

📁 Files created in: {deployment_path}
📖 Instructions: {deployment_path}/DEPLOYMENT_INSTRUCTIONS.md

Next steps:
1. Review generated files
2. Deploy via RunPod web interface  
3. Update pipeline configuration
4. Test deployment

💡 Tip: Use --existing-pod to deploy to your Superlearn Ai pod instead.
            """)
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 