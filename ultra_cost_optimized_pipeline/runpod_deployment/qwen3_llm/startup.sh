
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
                