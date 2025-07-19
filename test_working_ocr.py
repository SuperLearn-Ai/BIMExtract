#!/usr/bin/env python3
"""
Simple working OCR test with correct PaddleOCR API
"""

import logging
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_test_image():
    """Create a simple image with clear text"""
    # Create a white image
    width, height = 800, 200
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    # Add some clear black text
    text = "This is a simple test for OCR extraction."
    
    try:
        # Try to use a decent font
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
    
    # Draw text in black
    draw.text((50, 80), text, fill='black', font=font)
    
    return image

def test_paddleocr_correct_api():
    """Test PaddleOCR with correct API"""
    try:
        from paddleocr import PaddleOCR
        
        logging.info("🚀 Initializing PaddleOCR with correct API...")
        ocr = PaddleOCR(use_textline_orientation=True, lang='en')
        logging.info("✅ PaddleOCR initialized successfully")
        
        # Create test image
        logging.info("📸 Creating test image...")
        test_image = create_test_image()
        test_image.save('test_ocr_image.png')
        logging.info("✅ Test image saved as test_ocr_image.png")
        
        # Convert to numpy array
        image_np = np.array(test_image)
        
        # Run OCR with correct API
        logging.info("🔍 Running OCR on test image...")
        results = ocr.predict(image_np)
        
        if results and results[0]:
            logging.info("✅ OCR extraction successful!")
            extracted_text = ""
            for line in results[0]:
                if line:
                    text = line[1][0]  # Get the text
                    confidence = line[1][1]  # Get confidence
                    extracted_text += text + " "
                    logging.info(f"   Text: '{text}' (confidence: {confidence:.2f})")
            
            logging.info(f"📝 Full extracted text: '{extracted_text.strip()}'")
            return True, extracted_text.strip()
        else:
            logging.error("❌ No text found by OCR")
            return False, ""
            
    except Exception as e:
        logging.error(f"❌ OCR test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, ""

if __name__ == "__main__":
    logging.info("🔍 Starting Working OCR Test")
    logging.info("=" * 50)
    
    success, text = test_paddleocr_correct_api()
    
    if success:
        logging.info("🎉 SUCCESS! OCR is working correctly!")
        logging.info(f"📝 Extracted: '{text}'")
    else:
        logging.error("💥 OCR test failed.")