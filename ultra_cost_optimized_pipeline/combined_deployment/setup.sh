#!/bin/bash
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
python handler.py