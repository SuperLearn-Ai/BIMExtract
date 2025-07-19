"""
Comprehensive test script to identify gaps in PDF extraction implementation
"""

import asyncio
import logging
import sys
import tempfile
import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import traceback

# Add ultra_cost_optimized_pipeline/src to path
sys.path.append(str(Path(__file__).parent / "ultra_cost_optimized_pipeline" / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_pdf_with_reportlab(filename: str, content_type: str = "academic"):
    """Create test PDF with various content types using ReportLab"""
    
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    if content_type == "academic":
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            alignment=1,  # Center
            spaceAfter=20
        )
        story.append(Paragraph("Academic Paper: Machine Learning in Natural Language Processing", title_style))
        story.append(Spacer(1, 12))
        
        # Abstract
        story.append(Paragraph("Abstract", styles['Heading2']))
        abstract_text = """
        This paper presents a comprehensive analysis of machine learning algorithms 
        and their applications in natural language processing. We demonstrate 
        significant improvements in accuracy and efficiency through novel approaches.
        Our methodology employs deep neural networks with transformer architectures.
        """
        story.append(Paragraph(abstract_text, styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Introduction with mathematical content
        story.append(Paragraph("1. Introduction", styles['Heading2']))
        intro_text = """
        Machine learning has revolutionized natural language processing. 
        The loss function can be defined as L = Σ(y_i - ŷ_i)². 
        Transformer models use attention mechanisms: Attention(Q,K,V) = softmax(QK^T/√d_k)V.
        """
        story.append(Paragraph(intro_text, styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Table
        story.append(Paragraph("2. Results", styles['Heading2']))
        table_data = [
            ['Model', 'Accuracy', 'F1-Score', 'Processing Time'],
            ['BERT', '92.5%', '91.2%', '2.3s'],
            ['GPT-3', '94.1%', '93.8%', '1.8s'],
            ['RoBERTa', '93.2%', '92.5%', '2.1s']
        ]
        
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(table)
        story.append(Spacer(1, 12))
        
        # Conclusion
        story.append(Paragraph("3. Conclusion", styles['Heading2']))
        conclusion_text = """
        Our experiments demonstrate significant improvements in both accuracy and efficiency.
        The proposed approach achieves state-of-the-art results on benchmark datasets.
        Future work will explore multi-modal architectures and cross-lingual applications.
        """
        story.append(Paragraph(conclusion_text, styles['Normal']))
        
    elif content_type == "simple_text":
        story.append(Paragraph("Simple Document", styles['Heading1']))
        story.append(Paragraph("This is a simple text document with minimal formatting.", styles['Normal']))
        
    elif content_type == "complex_layout":
        # Multi-column layout simulation
        story.append(Paragraph("Complex Layout Document", styles['Heading1']))
        
        # Create table to simulate columns
        col_data = [
            ['Column 1 Content: Lorem ipsum dolor sit amet, consectetur adipiscing elit.', 
             'Column 2 Content: Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.']
        ]
        col_table = Table(col_data, colWidths=[3*inch, 3*inch])
        col_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('FONTSIZE', (0, 0), (-1, -1), 10)
        ]))
        story.append(col_table)
    
    try:
        doc.build(story)
        logger.info(f"Created test PDF: {filename}")
        return True
    except Exception as e:
        logger.error(f"Failed to create PDF {filename}: {e}")
        return False


def create_test_image_pdf(filename: str):
    """Create a PDF with embedded images and figures"""
    try:
        # Create a simple figure using matplotlib
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create a simple plot
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        ax.plot(x, y, 'bo-', label='Linear Function')
        ax.set_xlabel('X Values')
        ax.set_ylabel('Y Values')
        ax.set_title('Sample Mathematical Plot')
        ax.legend()
        ax.grid(True)
        
        # Add some geometric shapes
        rect = patches.Rectangle((2, 3), 2, 3, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
        # Save as PDF
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        plt.close()
        
        logger.info(f"Created image-based PDF: {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create image PDF {filename}: {e}")
        return False


def create_problematic_pdf(filename: str):
    """Create a PDF that might cause extraction issues"""
    try:
        from reportlab.pdfgen import canvas
        
        c = canvas.Canvas(filename, pagesize=letter)
        width, height = letter
        
        # Rotated text
        c.saveState()
        c.translate(100, 400)
        c.rotate(45)
        c.drawString(0, 0, "Rotated text that might be hard to extract")
        c.restoreState()
        
        # Overlapping text
        c.drawString(100, 300, "Overlapping text line 1")
        c.drawString(105, 295, "Overlapping text line 2")
        
        # Very small text
        c.setFont("Helvetica", 6)
        c.drawString(100, 250, "Very small text that might be missed by OCR")
        
        # Text with special characters
        c.setFont("Helvetica", 12)
        c.drawString(100, 200, "Special chars: α β γ δ ε ∑ ∫ ∂ ∇ ∞ ≈ ≠ ≤ ≥")
        
        # Mathematical equations (as text)
        c.drawString(100, 150, "E = mc² ∫f(x)dx Σx² √(a²+b²)")
        
        c.save()
        
        logger.info(f"Created problematic PDF: {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create problematic PDF {filename}: {e}")
        return False


async def test_visual_parsing_stage():
    """Test the visual parsing stage with various PDF types"""
    try:
        from stage1_visual_parsing import VisualParsingStage
        
        # Initialize visual parsing stage
        stage = VisualParsingStage()
        logger.info("Visual parsing stage initialized successfully")
        
        return stage
        
    except Exception as e:
        logger.error(f"Failed to initialize visual parsing stage: {e}")
        logger.error(traceback.format_exc())
        return None


async def test_pdf_extraction(stage, pdf_path: str):
    """Test PDF extraction and identify issues"""
    logger.info(f"\n{'='*50}")
    logger.info(f"Testing PDF: {pdf_path}")
    logger.info(f"{'='*50}")
    
    issues = []
    
    try:
        # Test PDF to image conversion
        if hasattr(stage, 'convert_pdf_to_images'):
            images = stage.convert_pdf_to_images(pdf_path)
            if not images:
                issues.append("PDF to image conversion failed - no images returned")
            else:
                logger.info(f"✅ PDF converted to {len(images)} images")
        else:
            issues.append("convert_pdf_to_images method not found")
            
        # Test full document processing
        results = await stage.process_document(pdf_path)
        
        if not results:
            issues.append("Document processing returned no results")
        else:
            logger.info(f"✅ Document processed, got {len(results)} page results")
            
            # Analyze each page result
            for i, result in enumerate(results):
                logger.info(f"\nPage {i+1} Analysis:")
                logger.info(f"  Text length: {len(result.text_content)}")
                logger.info(f"  LaTeX formulas: {len(result.latex_formulas)}")
                logger.info(f"  Confidence scores: {result.confidence_scores}")
                logger.info(f"  Processing time: {result.processing_time:.2f}s")
                logger.info(f"  Cost: ${result.cost_estimate:.6f}")
                
                # Check for potential issues
                if len(result.text_content) < 10:
                    issues.append(f"Page {i+1}: Very little text extracted ({len(result.text_content)} chars)")
                
                if result.confidence_scores.get('text', 0) < 0.5:
                    issues.append(f"Page {i+1}: Low text extraction confidence ({result.confidence_scores.get('text', 0):.2f})")
                
                # Print sample of extracted text
                if result.text_content:
                    sample_text = result.text_content[:200] + "..." if len(result.text_content) > 200 else result.text_content
                    logger.info(f"  Sample text: {repr(sample_text)}")
                else:
                    issues.append(f"Page {i+1}: No text extracted")
        
    except Exception as e:
        issues.append(f"Processing failed with exception: {e}")
        logger.error(f"Exception during processing: {e}")
        logger.error(traceback.format_exc())
    
    return issues


async def run_comprehensive_test():
    """Run comprehensive PDF extraction test"""
    logger.info("🔍 Starting Comprehensive PDF Extraction Analysis")
    logger.info("="*60)
    
    # Create docs directory
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    
    # Check for required dependencies
    missing_deps = []
    
    try:
        import pdf2image
    except ImportError:
        missing_deps.append("pdf2image")
    
    try:
        import paddleocr
    except ImportError:
        missing_deps.append("paddleocr")
    
    try:
        from transformers import AutoModel
    except ImportError:
        missing_deps.append("transformers")
    
    try:
        import reportlab
    except ImportError:
        missing_deps.append("reportlab")
    
    if missing_deps:
        logger.error(f"❌ Missing dependencies: {missing_deps}")
        logger.error("Install with: pip install pdf2image paddleocr transformers reportlab matplotlib")
        return
    
    # Create test PDFs
    test_pdfs = {}
    
    logger.info("📄 Creating test PDFs...")
    
    # Academic paper PDF
    academic_pdf = docs_dir / "academic_paper.pdf"
    if create_test_pdf_with_reportlab(str(academic_pdf), "academic"):
        test_pdfs["academic"] = academic_pdf
    
    # Simple text PDF
    simple_pdf = docs_dir / "simple_text.pdf"
    if create_test_pdf_with_reportlab(str(simple_pdf), "simple_text"):
        test_pdfs["simple"] = simple_pdf
    
    # Complex layout PDF
    complex_pdf = docs_dir / "complex_layout.pdf"
    if create_test_pdf_with_reportlab(str(complex_pdf), "complex_layout"):
        test_pdfs["complex"] = complex_pdf
    
    # Image-based PDF
    image_pdf = docs_dir / "image_content.pdf"
    if create_test_image_pdf(str(image_pdf)):
        test_pdfs["image"] = image_pdf
    
    # Problematic PDF
    problem_pdf = docs_dir / "problematic.pdf"
    if create_problematic_pdf(str(problem_pdf)):
        test_pdfs["problematic"] = problem_pdf
    
    logger.info(f"✅ Created {len(test_pdfs)} test PDFs")
    
    # Test visual parsing stage
    logger.info("\n🚀 Testing Visual Parsing Stage...")
    stage = await test_visual_parsing_stage()
    
    if not stage:
        logger.error("❌ Cannot proceed - visual parsing stage failed to initialize")
        return
    
    # Test each PDF
    all_issues = {}
    
    for pdf_type, pdf_path in test_pdfs.items():
        issues = await test_pdf_extraction(stage, str(pdf_path))
        all_issues[pdf_type] = issues
    
    # Summary of issues
    logger.info("\n📊 COMPREHENSIVE ANALYSIS SUMMARY")
    logger.info("="*50)
    
    total_issues = 0
    for pdf_type, issues in all_issues.items():
        logger.info(f"\n{pdf_type.upper()} PDF:")
        if issues:
            for issue in issues:
                logger.error(f"  ❌ {issue}")
                total_issues += 1
        else:
            logger.info(f"  ✅ No issues detected")
    
    logger.info(f"\n🎯 TOTAL ISSUES FOUND: {total_issues}")
    
    if total_issues > 0:
        logger.info("\n🔧 RECOMMENDED FIXES:")
        logger.info("1. Check pdf2image poppler installation")
        logger.info("2. Verify PaddleOCR model downloads")
        logger.info("3. Add better error handling for PDF conversion")
        logger.info("4. Implement fallback mechanisms for failed extractions")
        logger.info("5. Add image preprocessing optimizations")
        logger.info("6. Improve confidence score calculations")
    
    # Cleanup
    if hasattr(stage, 'cleanup'):
        stage.cleanup()
    
    return all_issues


if __name__ == "__main__":
    asyncio.run(run_comprehensive_test())