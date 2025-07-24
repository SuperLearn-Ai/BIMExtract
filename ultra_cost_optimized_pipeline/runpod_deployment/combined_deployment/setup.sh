#!/bin/bash
# Combined Setup Script for Ultra-Cost-Optimized Pipeline on RunPod
set -e

echo "🚀 Setting up Ultra-Cost-Optimized Pipeline"
echo "   🤖 Services: Qwen3 LLM + Visual Parsing"
echo "   🎯 Target: $0.15 per 1K pages"

# Update system
apt-get update && apt-get install -y wget git

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --no-cache-dir -r requirements.txt

# Download and cache Qwen3 model
echo "🤖 Pre-downloading Qwen3 model..."
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = 'Qwen/Qwen2.5-Coder-32B-Instruct'
print('Downloading Qwen3 model and tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    trust_remote_code=True,
    load_in_4bit=True
)
print('✅ Qwen3 model cached successfully!')
"

# Initialize PaddleOCR (downloads models on first use)
echo "👁️ Initializing PaddleOCR..."
python -c "
import paddleocr
print('Initializing PaddleOCR...')
ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
print('✅ PaddleOCR initialized successfully!')
"

echo "✅ Setup complete! Starting handler..."

# Start the combined handler
python combined_handler.py 