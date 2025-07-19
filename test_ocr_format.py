#!/usr/bin/env python3
"""
Diagnostic test to understand PaddleOCR response format
"""

import logging
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_test_image():
    """Create a simple image with clear text"""
    # Create a white image
    width, height = 400, 100
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    # Add some clear black text
    text = "Hello World!"
    
    try:
        # Try to use a decent font
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)
    except:
        # Fallback to default font  
        font = ImageFont.load_default()
    
    # Draw text in black
    draw.text((50, 30), text, fill='black', font=font)
    
    return image

def analyze_ocr_response():
    """Analyze PaddleOCR response format"""
    try:
        from paddleocr import PaddleOCR
        
        logging.info("🚀 Initializing PaddleOCR...")
        ocr = PaddleOCR(use_textline_orientation=True, lang='en')
        
        # Create test image
        test_image = create_test_image()
        test_image.save('debug_ocr_image.png')
        logging.info("✅ Test image saved")
        
        # Convert to numpy array
        image_np = np.array(test_image)
        
        # Run OCR
        logging.info("🔍 Running OCR...")
        results = ocr.predict(image_np)
        
        # Analyze response format
        logging.info("📊 Analyzing response format:")
        logging.info(f"Type: {type(results)}")
        
        if isinstance(results, dict):
            logging.info("Response is a dictionary!")
            logging.info(f"Keys: {list(results.keys())}")
            for key, value in results.items():
                logging.info(f"Key '{key}': {type(value)} - {value}")
                
        elif isinstance(results, list):
            logging.info("Response is a list!")
            logging.info(f"Length: {len(results)}")
            for i, item in enumerate(results):
                logging.info(f"Item {i}: {type(item)} - {item}")
        else:
            logging.info(f"Unknown response type: {type(results)}")
            logging.info(f"Response: {results}")
        
        return results
            
    except Exception as e:
        logging.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    logging.info("🔍 Starting OCR Format Analysis")
    logging.info("=" * 50)
    
    results = analyze_ocr_response()
    
    if results:
        logging.info("🎉 Got OCR results! Check the format above.")
    else:
        logging.error("💥 No results obtained.")