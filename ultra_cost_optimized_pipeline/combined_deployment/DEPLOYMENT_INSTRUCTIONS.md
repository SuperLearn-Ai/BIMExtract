# Deploy to Existing RunPod Pod: jeqrwyd0hbl40c

## Files Created
- handler.py: Combined Qwen3 + Visual Parsing handler
- requirements.txt: All dependencies
- setup.sh: Installation script

## Deployment Steps

### 1. Upload Files to Pod
```bash
# Upload to pod jeqrwyd0hbl40c (updated SSH connection)
scp -P 52702 -i ~/.ssh/id_ed25519 combined_deployment/handler.py root@190.111.198.202:/workspace/
scp -P 52702 -i ~/.ssh/id_ed25519 combined_deployment/requirements.txt root@190.111.198.202:/workspace/
scp -P 52702 -i ~/.ssh/id_ed25519 combined_deployment/setup.sh root@190.111.198.202:/workspace/
```

### 2. Setup and Start Service
```bash
# Connect to pod and run setup (updated SSH connection)
ssh root@190.111.198.202 -p 52702 -i ~/.ssh/id_ed25519 "cd /workspace && chmod +x setup.sh && ./setup.sh"
```

### 3. Get Pod URL
The pod will be available at: https://jeqrwyd0hbl40c-8000.proxy.runpod.net

### 4. Test Endpoints

**Qwen3 LLM Test:**
```bash
curl -X POST https://jeqrwyd0hbl40c-8000.proxy.runpod.net/runsync \
  -H "Content-Type: application/json" \
  -d '{"input": {"service": "qwen3_llm", "text": "Analyze this document chunk", "thinking_mode": true}}'
```

**Visual Parsing Test:**
```bash
curl -X POST https://jeqrwyd0hbl40c-8000.proxy.runpod.net/runsync \
  -H "Content-Type: application/json" \
  -d '{"input": {"service": "visual_parsing", "image": "base64_image_data", "type": "ocr"}}'
```

### 5. Update Local Configuration
Create config/runpod_endpoints.yaml:
```yaml
endpoints:
  combined_service: "https://jeqrwyd0hbl40c-8000.proxy.runpod.net"
  
api_key: "your-runpod-api-key"
```
