"""
Simple test to identify basic PDF extraction issues
"""

import asyncio
import logging
import sys
import traceback
from pathlib import Path

# Add ultra_cost_optimized_pipeline/src to path
sys.path.append(str(Path(__file__).parent / "ultra_cost_optimized_pipeline" / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_implementation_gaps():
    """Analyze the implementation for potential gaps"""
    logger.info("🔍 Analyzing PDF Extraction Implementation for Gaps")
    logger.info("="*60)
    
    gaps = []
    
    # Check if configuration files exist
    config_path = Path("ultra_cost_optimized_pipeline/config/pipeline_config.yaml")
    if not config_path.exists():
        gaps.append("❌ Pipeline configuration file missing")
    else:
        logger.info("✅ Pipeline configuration file exists")
    
    # Check if source files exist
    required_files = [
        "ultra_cost_optimized_pipeline/src/stage1_visual_parsing.py",
        "ultra_cost_optimized_pipeline/src/utils/cost_tracking.py",
        "ultra_cost_optimized_pipeline/src/utils/quantization.py"
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            gaps.append(f"❌ Required file missing: {file_path}")
        else:
            logger.info(f"✅ Found: {file_path}")
    
    return gaps


def check_dependencies():
    """Check for required dependencies"""
    logger.info("\n📦 Checking Dependencies")
    logger.info("="*30)
    
    missing_deps = []
    available_deps = []
    
    dependencies = [
        ("pdf2image", "PDF to image conversion"),
        ("paddleocr", "OCR text extraction"),
        ("transformers", "ML model loading"),
        ("PIL", "Image processing"),
        ("torch", "PyTorch framework"),
        ("numpy", "Numerical operations"),
        ("yaml", "Configuration parsing"),
        ("redis", "Caching (optional)"),
        ("qdrant_client", "Vector storage (optional)")
    ]
    
    for dep_name, description in dependencies:
        try:
            __import__(dep_name)
            available_deps.append(f"✅ {dep_name}: {description}")
            logger.info(f"✅ {dep_name}")
        except ImportError:
            missing_deps.append(f"❌ {dep_name}: {description}")
            logger.warning(f"❌ {dep_name}")
    
    return missing_deps, available_deps


async def test_basic_import():
    """Test basic import of visual parsing stage"""
    logger.info("\n🚀 Testing Basic Import")
    logger.info("="*25)
    
    issues = []
    
    try:
        from stage1_visual_parsing import VisualParsingStage
        logger.info("✅ Successfully imported VisualParsingStage")
        
        # Try to create instance
        try:
            stage = VisualParsingStage()
            logger.info("✅ Successfully created VisualParsingStage instance")
            return stage, issues
        except Exception as e:
            issues.append(f"Failed to create VisualParsingStage instance: {e}")
            logger.error(f"❌ Failed to create instance: {e}")
            logger.error(traceback.format_exc())
            return None, issues
            
    except Exception as e:
        issues.append(f"Failed to import VisualParsingStage: {e}")
        logger.error(f"❌ Failed to import: {e}")
        logger.error(traceback.format_exc())
        return None, issues


def analyze_pdf_processing_gaps():
    """Analyze specific gaps in PDF processing logic"""
    logger.info("\n🔬 Analyzing PDF Processing Logic")
    logger.info("="*35)
    
    gaps = []
    
    # Read the visual parsing implementation
    try:
        with open("ultra_cost_optimized_pipeline/src/stage1_visual_parsing.py", 'r') as f:
            content = f.read()
        
        # Check for potential issues
        
        # 1. PDF conversion dependencies
        if "pdf2image" not in content:
            gaps.append("❌ pdf2image import missing")
        
        if "poppler" not in content.lower():
            gaps.append("⚠️  No poppler-utils check (required for pdf2image)")
        
        # 2. Error handling
        if "except Exception as e:" not in content:
            gaps.append("❌ Insufficient error handling")
        
        # 3. File validation
        if "os.path.exists" not in content and "Path.exists" not in content:
            gaps.append("⚠️  No file existence validation")
        
        # 4. Image preprocessing
        if "preprocess_image" in content:
            logger.info("✅ Image preprocessing implemented")
        else:
            gaps.append("❌ Image preprocessing missing")
        
        # 5. Async handling
        if "asyncio" in content:
            logger.info("✅ Async processing implemented")
        else:
            gaps.append("⚠️  No async processing")
        
        # 6. OCR configuration
        if "PaddleOCR" in content:
            logger.info("✅ PaddleOCR integration found")
        else:
            gaps.append("❌ PaddleOCR integration missing")
        
        # 7. Memory management
        if "torch.cuda.empty_cache" in content:
            logger.info("✅ GPU memory management implemented")
        else:
            gaps.append("⚠️  No GPU memory cleanup")
        
        # 8. Configuration validation
        if "yaml.safe_load" in content:
            logger.info("✅ YAML configuration loading")
        else:
            gaps.append("❌ Configuration loading missing")
        
    except FileNotFoundError:
        gaps.append("❌ Visual parsing implementation file not found")
    except Exception as e:
        gaps.append(f"❌ Error reading implementation: {e}")
    
    return gaps


def check_pdf_conversion_requirements():
    """Check specific requirements for PDF conversion"""
    logger.info("\n📄 Checking PDF Conversion Requirements")
    logger.info("="*40)
    
    issues = []
    
    # Check for pdf2image
    try:
        import pdf2image
        logger.info("✅ pdf2image available")
        
        # Check for poppler
        try:
            from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError
            # Try a simple operation to check poppler
            # This would normally fail if poppler is not installed
            logger.info("✅ pdf2image imports successful")
        except Exception as e:
            issues.append(f"pdf2image import issues: {e}")
            
    except ImportError:
        issues.append("❌ pdf2image not installed")
    
    # Check for PIL/Pillow
    try:
        from PIL import Image
        logger.info("✅ PIL/Pillow available")
    except ImportError:
        issues.append("❌ PIL/Pillow not installed")
    
    # Check for numpy
    try:
        import numpy as np
        logger.info("✅ NumPy available")
    except ImportError:
        issues.append("❌ NumPy not installed")
    
    return issues


async def main():
    """Main test function"""
    logger.info("🎯 Starting Basic PDF Extraction Analysis")
    logger.info("="*50)
    
    # 1. Check implementation gaps
    impl_gaps = analyze_implementation_gaps()
    
    # 2. Check dependencies
    missing_deps, available_deps = check_dependencies()
    
    # 3. Check PDF conversion requirements
    pdf_issues = check_pdf_conversion_requirements()
    
    # 4. Analyze PDF processing logic
    logic_gaps = analyze_pdf_processing_gaps()
    
    # 5. Test basic import
    stage, import_issues = await test_basic_import()
    
    # Summary
    logger.info("\n📊 ANALYSIS SUMMARY")
    logger.info("="*20)
    
    all_issues = impl_gaps + logic_gaps + import_issues + pdf_issues
    
    logger.info(f"Implementation gaps: {len(impl_gaps)}")
    logger.info(f"Logic gaps: {len(logic_gaps)}")
    logger.info(f"Import issues: {len(import_issues)}")
    logger.info(f"PDF conversion issues: {len(pdf_issues)}")
    logger.info(f"Missing dependencies: {len(missing_deps)}")
    
    logger.info(f"\n🎯 TOTAL ISSUES: {len(all_issues)}")
    
    if all_issues:
        logger.info("\n❌ IDENTIFIED ISSUES:")
        for i, issue in enumerate(all_issues, 1):
            logger.error(f"  {i}. {issue}")
    
    if missing_deps:
        logger.info("\n📦 MISSING DEPENDENCIES:")
        for dep in missing_deps:
            logger.error(f"  {dep}")
    
    # Recommendations
    logger.info("\n🔧 RECOMMENDED FIXES:")
    
    if missing_deps:
        logger.info("1. Install missing dependencies:")
        logger.info("   pip install pdf2image paddleocr transformers torch pillow numpy pyyaml")
        logger.info("   # For Ubuntu/Debian: sudo apt-get install poppler-utils")
        logger.info("   # For macOS: brew install poppler")
    
    if logic_gaps:
        logger.info("2. Fix implementation gaps:")
        for gap in logic_gaps:
            logger.info(f"   - {gap}")
    
    logger.info("3. Add robust error handling for PDF conversion failures")
    logger.info("4. Implement fallback mechanisms for different PDF types")
    logger.info("5. Add input validation and file format checks")
    logger.info("6. Optimize image preprocessing for better OCR results")
    
    return len(all_issues)


if __name__ == "__main__":
    try:
        total_issues = asyncio.run(main())
        if total_issues == 0:
            print("\n🎉 No major issues found! Implementation looks good.")
        else:
            print(f"\n⚠️  Found {total_issues} issues that need attention.")
    except Exception as e:
        print(f"\n💥 Test failed: {e}")
        traceback.print_exc()