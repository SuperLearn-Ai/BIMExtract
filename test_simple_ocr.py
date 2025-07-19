#!/usr/bin/env python3
"""
Simple OCR test to verify PaddleOCR is working correctly
"""

import sys
import os
import logging
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add the pipeline path
pipeline_path = os.path.join(os.getcwd(), 'ultra_cost_optimized_pipeline', 'src')
sys.path.insert(0, pipeline_path)

def create_test_image():
    """Create a simple image with clear text"""
    # Create a white image
    width, height = 800, 200
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    # Add some clear black text
    text = "This is a simple test for OCR extraction. The text should be clearly readable."
    
    try:
        # Try to use a decent font
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
    
    # Draw text in black
    draw.text((50, 80), text, fill='black', font=font)
    
    return image

def test_paddleocr_directly():
    """Test PaddleOCR directly"""
    try:
        from paddleocr import PaddleOCR
        
        logging.info("🚀 Initializing PaddleOCR...")
        ocr = PaddleOCR(use_angle_cls=True, lang='en')
        logging.info("✅ PaddleOCR initialized successfully")
        
        # Create test image
        logging.info("📸 Creating test image...")
        test_image = create_test_image()
        test_image.save('test_ocr_image.png')
        logging.info("✅ Test image saved as test_ocr_image.png")
        
        # Convert to numpy array
        image_np = np.array(test_image)
        
        # Run OCR
        logging.info("🔍 Running OCR on test image...")
        results = ocr.ocr(image_np, cls=True)
        
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
        return False, ""

def test_pipeline_ocr():
    """Test OCR through the pipeline class"""
    try:
        from stage1_visual_parsing import QuantizedPaddleOCR
        
        logging.info("🚀 Testing pipeline OCR class...")
        
        # Initialize OCR with config
        config = {
            'use_angle_cls': True,
            'lang': 'en'
        }
        
        ocr_processor = QuantizedPaddleOCR(config)
        
        # Create test image
        test_image = create_test_image()
        image_np = np.array(test_image)
        
        # Extract text
        logging.info("🔍 Running pipeline OCR...")
        text, confidence = ocr_processor.extract_text(image_np)
        
        if text:
            logging.info(f"✅ Pipeline OCR successful!")
            logging.info(f"📝 Extracted text: '{text}'")
            logging.info(f"🎯 Confidence: {confidence:.2f}")
            return True, text
        else:
            logging.error("❌ Pipeline OCR returned empty text")
            return False, ""
            
    except Exception as e:
        logging.error(f"❌ Pipeline OCR test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, ""

if __name__ == "__main__":
    logging.info("🔍 Starting Simple OCR Test")
    logging.info("=" * 50)
    
    # Test 1: Direct PaddleOCR
    logging.info("\n📋 Test 1: Direct PaddleOCR")
    success1, text1 = test_paddleocr_directly()
    
    # Test 2: Pipeline OCR
    logging.info("\n📋 Test 2: Pipeline OCR Class")
    success2, text2 = test_pipeline_ocr()
    
    # Summary
    logging.info("\n📊 TEST SUMMARY")
    logging.info("=" * 30)
    logging.info(f"Direct PaddleOCR: {'✅ PASS' if success1 else '❌ FAIL'}")
    logging.info(f"Pipeline OCR: {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1 or success2:
        logging.info("🎉 OCR is working! Text extraction capability confirmed.")
    else:
        logging.error("💥 OCR tests failed. Need to investigate further.")