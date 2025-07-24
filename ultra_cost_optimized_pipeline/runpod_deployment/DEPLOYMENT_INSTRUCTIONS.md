
# RunPod Deployment Instructions

## 1. Prepare Files
Files created in: runpod_deployment/

## 2. Deploy via RunPod Web Interface

### For Qwen3 LLM:
1. Go to RunPod.io → Create Pod
2. Select: NVIDIA RTX A5000 
3. Template: PyTorch 2.1
4. Container Disk: 60GB
5. Upload files from: runpod_deployment/qwen3_llm/
6. Set environment variables from config.json
7. Deploy and note endpoint URL

### For Visual Parsing:
1. Go to RunPod.io → Create Pod  
2. Select: NVIDIA RTX A5000
3. Template: PyTorch 2.1
4. Container Disk: 40GB
5. Upload files from: runpod_deployment/visual_parsing/
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
