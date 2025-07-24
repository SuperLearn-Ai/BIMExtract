
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
            "token_count": len(outputs[0]) - inputs["input_ids"].shape[1]
        }
        
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
