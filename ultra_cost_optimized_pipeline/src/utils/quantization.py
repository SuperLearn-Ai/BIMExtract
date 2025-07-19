"""
Quantization Manager for Ultra-Cost-Optimized Pipeline
Handles model quantization to reduce memory usage and costs
"""

import torch
import logging
from typing import Dict, Any, Optional, Tuple
from transformers import BitsAndBytesConfig, AutoModel, AutoTokenizer
from pathlib import Path
import psutil
import time


class QuantizationConfig:
    """Configuration for different quantization strategies"""
    
    @staticmethod
    def get_4bit_config(compute_dtype: str = "bfloat16") -> BitsAndBytesConfig:
        """4-bit quantization configuration"""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=getattr(torch, compute_dtype)
        )
    
    @staticmethod
    def get_8bit_config() -> BitsAndBytesConfig:
        """8-bit quantization configuration"""
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_has_fp16_weight=False
        )
    
    @staticmethod
    def get_mixed_precision_config() -> BitsAndBytesConfig:
        """Mixed precision configuration"""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="fp4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_enable_fp32_cpu_offload=True
        )


class MemoryMonitor:
    """Monitor memory usage during model operations"""
    
    def __init__(self):
        self.baseline_memory = self.get_current_memory()
        
    def get_current_memory(self) -> Dict[str, float]:
        """Get current memory usage"""
        memory_info = {
            "cpu_memory_gb": psutil.virtual_memory().used / (1024**3),
            "cpu_memory_percent": psutil.virtual_memory().percent
        }
        
        if torch.cuda.is_available():
            memory_info.update({
                "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                "gpu_memory_percent": (torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()) * 100 if torch.cuda.max_memory_allocated() > 0 else 0
            })
        
        return memory_info
    
    def get_memory_delta(self) -> Dict[str, float]:
        """Get memory usage change since baseline"""
        current = self.get_current_memory()
        delta = {}
        
        for key, value in current.items():
            if key in self.baseline_memory:
                delta[f"delta_{key}"] = value - self.baseline_memory[key]
        
        return delta
    
    def log_memory_usage(self, context: str = ""):
        """Log current memory usage"""
        memory_info = self.get_current_memory()
        delta = self.get_memory_delta()
        
        logging.info(f"Memory usage {context}:")
        for key, value in memory_info.items():
            unit = "%" if "percent" in key else "GB"
            logging.info(f"  {key}: {value:.2f}{unit}")
        
        if delta:
            logging.info("  Memory delta:")
            for key, value in delta.items():
                unit = "%" if "percent" in key else "GB"
                logging.info(f"    {key}: {value:+.2f}{unit}")


class QuantizationManager:
    """Manages model quantization and optimization"""
    
    def __init__(self):
        self.quantized_models = {}
        self.memory_monitor = MemoryMonitor()
        
        # Performance tracking
        self.quantization_stats = {
            "models_quantized": 0,
            "memory_saved_gb": 0.0,
            "inference_speedup": 0.0,
            "accuracy_retention": 0.0
        }
    
    def quantize_model(
        self,
        model_name: str,
        quantization_type: str = "4bit",
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.float16
    ) -> Tuple[Any, float]:
        """Quantize a model and return it with memory savings"""
        
        logging.info(f"Quantizing model: {model_name} with {quantization_type}")
        start_time = time.time()
        
        # Get baseline memory
        baseline_memory = self.memory_monitor.get_current_memory()
        
        # Choose quantization config
        if quantization_type == "4bit":
            quant_config = QuantizationConfig.get_4bit_config()
        elif quantization_type == "8bit":
            quant_config = QuantizationConfig.get_8bit_config()
        elif quantization_type == "mixed":
            quant_config = QuantizationConfig.get_mixed_precision_config()
        else:
            raise ValueError(f"Unsupported quantization type: {quantization_type}")
        
        try:
            # Load quantized model
            model = AutoModel.from_pretrained(
                model_name,
                quantization_config=quant_config,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Calculate memory savings
            post_memory = self.memory_monitor.get_current_memory()
            memory_saved = self._calculate_memory_savings(baseline_memory, post_memory)
            
            # Store quantized model
            model_key = f"{model_name}_{quantization_type}"
            self.quantized_models[model_key] = {
                "model": model,
                "quantization_type": quantization_type,
                "memory_saved_gb": memory_saved,
                "load_time": time.time() - start_time
            }
            
            # Update stats
            self.quantization_stats["models_quantized"] += 1
            self.quantization_stats["memory_saved_gb"] += memory_saved
            
            logging.info(f"Model quantized successfully. Memory saved: {memory_saved:.2f} GB")
            logging.info(f"Load time: {time.time() - start_time:.2f}s")
            
            return model, memory_saved
            
        except Exception as e:
            logging.error(f"Failed to quantize model {model_name}: {e}")
            raise
    
    def _calculate_memory_savings(self, baseline: Dict, current: Dict) -> float:
        """Calculate memory savings from quantization"""
        # Focus on GPU memory if available, otherwise CPU
        if "gpu_memory_allocated_gb" in baseline and "gpu_memory_allocated_gb" in current:
            return max(0, baseline["gpu_memory_allocated_gb"] - current["gpu_memory_allocated_gb"])
        else:
            return max(0, baseline["cpu_memory_gb"] - current["cpu_memory_gb"])
    
    def benchmark_quantized_model(
        self,
        model,
        tokenizer,
        test_inputs: list,
        original_model=None
    ) -> Dict[str, float]:
        """Benchmark quantized model performance"""
        logging.info("Benchmarking quantized model...")
        
        # Inference speed test
        start_time = time.time()
        
        with torch.no_grad():
            for test_input in test_inputs:
                inputs = tokenizer(test_input, return_tensors="pt", padding=True, truncation=True)
                if hasattr(model, 'generate'):
                    _ = model.generate(**inputs, max_length=50, do_sample=False)
                else:
                    _ = model(**inputs)
        
        inference_time = time.time() - start_time
        
        # Calculate metrics
        metrics = {
            "inference_time": inference_time,
            "inferences_per_second": len(test_inputs) / inference_time,
            "memory_usage_gb": self.memory_monitor.get_current_memory().get("gpu_memory_allocated_gb", 0)
        }
        
        # Compare with original model if provided
        if original_model is not None:
            original_start = time.time()
            with torch.no_grad():
                for test_input in test_inputs:
                    inputs = tokenizer(test_input, return_tensors="pt", padding=True, truncation=True)
                    if hasattr(original_model, 'generate'):
                        _ = original_model.generate(**inputs, max_length=50, do_sample=False)
                    else:
                        _ = original_model(**inputs)
            
            original_time = time.time() - original_start
            metrics["speedup_ratio"] = original_time / inference_time
            
            self.quantization_stats["inference_speedup"] = metrics["speedup_ratio"]
        
        logging.info(f"Benchmark results: {metrics}")
        return metrics
    
    def optimize_for_inference(self, model) -> Any:
        """Apply inference optimizations"""
        logging.info("Applying inference optimizations...")
        
        # Enable evaluation mode
        model.eval()
        
        # Compile model if using PyTorch 2.0+
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode="reduce-overhead")
                logging.info("Model compiled for faster inference")
            except Exception as e:
                logging.warning(f"Model compilation failed: {e}")
        
        # Enable memory efficient attention if available
        if hasattr(model, 'config') and hasattr(model.config, 'use_memory_efficient_attention'):
            model.config.use_memory_efficient_attention = True
        
        return model
    
    def apply_gradient_checkpointing(self, model, checkpoint_ratio: float = 0.5) -> Any:
        """Apply gradient checkpointing to reduce memory usage"""
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logging.info(f"Gradient checkpointing enabled with ratio: {checkpoint_ratio}")
        
        return model
    
    def get_model_size_info(self, model) -> Dict[str, Any]:
        """Get detailed model size information"""
        param_count = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate memory usage
        param_memory = 0
        for param in model.parameters():
            if param.dtype == torch.float32:
                param_memory += param.numel() * 4  # 4 bytes per float32
            elif param.dtype == torch.float16:
                param_memory += param.numel() * 2  # 2 bytes per float16
            elif param.dtype == torch.int8:
                param_memory += param.numel() * 1  # 1 byte per int8
            else:
                param_memory += param.numel() * 4  # Default to 4 bytes
        
        return {
            "total_parameters": param_count,
            "trainable_parameters": trainable_params,
            "parameter_memory_mb": param_memory / (1024 * 1024),
            "model_size_mb": param_memory / (1024 * 1024)
        }
    
    def cleanup_unused_models(self):
        """Clean up unused models to free memory"""
        for model_key, model_info in list(self.quantized_models.items()):
            # In a real implementation, you'd check if the model is actually unused
            # For now, we'll just demonstrate the cleanup pattern
            logging.info(f"Cleaning up model: {model_key}")
            
            # Clear model from memory
            del model_info["model"]
            del self.quantized_models[model_key]
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logging.info("Model cleanup completed")
    
    def get_quantization_report(self) -> Dict[str, Any]:
        """Get detailed quantization report"""
        current_memory = self.memory_monitor.get_current_memory()
        
        report = {
            "quantization_stats": self.quantization_stats,
            "current_memory_usage": current_memory,
            "active_models": len(self.quantized_models),
            "model_details": {}
        }
        
        # Add details for each quantized model
        for model_key, model_info in self.quantized_models.items():
            report["model_details"][model_key] = {
                "quantization_type": model_info["quantization_type"],
                "memory_saved_gb": model_info["memory_saved_gb"],
                "load_time": model_info["load_time"]
            }
        
        return report


# Utility functions
def estimate_quantization_savings(model_size_gb: float, quantization_type: str) -> Dict[str, float]:
    """Estimate memory savings from quantization"""
    savings_ratios = {
        "4bit": 0.75,  # 75% memory reduction
        "8bit": 0.50,  # 50% memory reduction
        "mixed": 0.60   # 60% memory reduction
    }
    
    ratio = savings_ratios.get(quantization_type, 0.0)
    memory_saved = model_size_gb * ratio
    
    return {
        "original_size_gb": model_size_gb,
        "quantized_size_gb": model_size_gb - memory_saved,
        "memory_saved_gb": memory_saved,
        "savings_percentage": ratio * 100
    }


def choose_optimal_quantization(
    model_size_gb: float,
    available_memory_gb: float,
    performance_priority: str = "balanced"
) -> str:
    """Choose optimal quantization strategy based on constraints"""
    
    if model_size_gb > available_memory_gb:
        return "4bit"  # Most aggressive quantization
    
    if performance_priority == "speed":
        return "8bit"  # Good balance of speed and memory
    elif performance_priority == "memory":
        return "4bit"  # Maximum memory savings
    else:  # balanced
        if model_size_gb > available_memory_gb * 0.7:
            return "4bit"
        else:
            return "8bit"


# Example usage
def main():
    """Example usage of quantization manager"""
    manager = QuantizationManager()
    
    # Example model quantization
    try:
        model, memory_saved = manager.quantize_model(
            "microsoft/DialoGPT-medium",
            quantization_type="4bit"
        )
        
        print(f"Model quantized successfully, saved {memory_saved:.2f} GB")
        
        # Get model size info
        size_info = manager.get_model_size_info(model)
        print(f"Model info: {size_info}")
        
        # Get quantization report
        report = manager.get_quantization_report()
        print(f"Quantization report: {report}")
        
    except Exception as e:
        print(f"Quantization failed: {e}")


if __name__ == "__main__":
    main()