
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
