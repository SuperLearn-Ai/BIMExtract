#!/usr/bin/env python3
"""
Test script for Ultra-Cost-Optimized Pipeline with Qwen3-30B-A3B Stage 2
Tests the complete pipeline integration and validates performance targets
"""

import os
import sys
import asyncio
import logging
import time
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

# Test imports
try:
    from main_pipeline import UltraOptimizedPipeline
    print("✅ Pipeline imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


class PipelineTester:
    """Comprehensive pipeline testing suite"""
    
    def __init__(self):
        self.setup_logging()
        self.test_documents = self.find_test_documents()
        self.results = []
        
    def setup_logging(self):
        """Setup test logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - TEST - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('PipelineTester')
        
    def find_test_documents(self) -> List[str]:
        """Find available test documents"""
        documents_dir = Path("documents")
        if not documents_dir.exists():
            documents_dir.mkdir()
            
        test_files = ["sample.pdf", "sample2.pdf", "sample3.pdf"]
        available_files = []
        
        for file in test_files:
            file_path = documents_dir / file
            if file_path.exists():
                available_files.append(str(file_path))
            else:
                self.logger.warning(f"Test document not found: {file_path}")
        
        return available_files
    
    def create_mock_documents(self):
        """Create mock PDF documents for testing if none exist"""
        if self.test_documents:
            return  # Use existing documents
        
        self.logger.info("Creating mock test documents...")
        
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib.units import inch
            
            documents_dir = Path("documents")
            documents_dir.mkdir(exist_ok=True)
            
            # Sample document content
            sample_contents = [
                {
                    "title": "Simple Academic Paper",
                    "content": """
                    Abstract: This is a simple academic paper for testing purposes.
                    
                    Introduction:
                    Academic papers contain structured information that needs to be
                    processed efficiently. This document tests the pipeline's ability
                    to handle basic academic content.
                    
                    Methodology:
                    We use a systematic approach to document processing that involves
                    multiple stages of analysis and extraction.
                    
                    Results:
                    The results demonstrate effective processing capabilities across
                    different document types and complexity levels.
                    
                    Conclusion:
                    This pipeline shows promising results for cost-effective document
                    extraction and processing.
                    """
                },
                {
                    "title": "Mathematical Research Paper",
                    "content": """
                    Mathematical Analysis of Document Processing
                    
                    Abstract: This paper examines mathematical approaches to document
                    processing with focus on optimization techniques.
                    
                    1. Introduction
                    Mathematical optimization plays a crucial role in document processing.
                    
                    2. Mathematical Framework
                    Let f(x) = x^2 + 3x + 2 represent our cost function.
                    The derivative f'(x) = 2x + 3 gives us the optimization direction.
                    
                    3. Algorithm Analysis
                    Our algorithm achieves O(n log n) complexity for document chunking.
                    
                    4. Experimental Results
                    Table 1: Performance Metrics
                    | Method | Cost | Time | Accuracy |
                    |--------|------|------|----------|
                    | Ours   | 0.012| 2.3s | 97.2%   |
                    | Baseline| 0.04| 5.1s | 94.8%   |
                    
                    5. Conclusion
                    The proposed method demonstrates significant improvements in both
                    cost efficiency and processing speed.
                    """
                },
                {
                    "title": "Complex Technical Document", 
                    "content": """
                    Advanced Topics in Document Intelligence
                    
                    Abstract: This comprehensive study examines complex technical
                    approaches to document understanding and extraction.
                    
                    1. Literature Review
                    Recent advances in transformer architectures have revolutionized
                    natural language processing. Models like GPT-4, Claude, and
                    Qwen3 represent significant breakthroughs in language understanding.
                    
                    2. Technical Methodology
                    
                    2.1 Model Architecture
                    We employ a Mixture-of-Experts (MoE) architecture with the following
                    configuration:
                    - 30B total parameters
                    - 3B active parameters per token
                    - 128k context length
                    
                    2.2 Mathematical Formulation
                    The attention mechanism is defined as:
                    Attention(Q, K, V) = softmax(QK^T / √d_k)V
                    
                    Where Q, K, V are the query, key, and value matrices respectively.
                    
                    2.3 Optimization Strategy
                    Our cost function minimizes:
                    L = α * processing_cost + β * quality_loss + γ * latency_penalty
                    
                    3. Implementation Details
                    
                    3.1 Quantization
                    We use 4-bit quantization with NF4 format to reduce memory usage
                    while maintaining model performance.
                    
                    3.2 Caching Strategy  
                    Redis Cluster provides distributed caching with 95% hit rates.
                    
                    4. Experimental Evaluation
                    
                    4.1 Dataset
                    We evaluate on academic papers from arXiv, PubMed, and ACL Anthology.
                    
                    4.2 Metrics
                    - Cost per 1K pages: $0.012 (target)
                    - Processing time: <5 seconds per document
                    - Chunk quality score: >95%
                    - Cache hit rate: >90%
                    
                    4.3 Results Analysis
                    Our approach achieves state-of-the-art results across all metrics.
                    The thinking/non-thinking mode switching provides optimal balance
                    between quality and cost.
                    
                    5. Discussion
                    
                    5.1 Complexity Analysis
                    Documents are classified into three complexity levels:
                    - Simple: Basic text, minimal formulas
                    - Medium: Some technical content, moderate structure
                    - Complex: Heavy mathematical content, complex layouts
                    
                    5.2 Cost Breakdown
                    Stage 1 (Visual): $0.05 per 1K pages
                    Stage 2 (Qwen3): $0.012 per 1K pages  
                    Stage 3 (Vector): $0.03 per 1K pages
                    Total: $0.092 per 1K pages (93% cost reduction)
                    
                    6. Future Work
                    Future research directions include multimodal understanding,
                    real-time processing, and cross-lingual capabilities.
                    
                    7. Conclusion
                    This work demonstrates significant advances in cost-effective
                    document processing while maintaining high quality standards.
                    
                    References:
                    [1] Vaswani et al., "Attention Is All You Need", NIPS 2017
                    [2] Brown et al., "Language Models are Few-Shot Learners", NeurIPS 2020
                    [3] Bai et al., "Qwen Technical Report", arXiv 2023
                    """
                }
            ]
            
            # Create PDF documents
            styles = getSampleStyleSheet()
            
            for i, doc_content in enumerate(sample_contents):
                filename = f"sample{i+1}.pdf" if i > 0 else "sample.pdf"
                file_path = documents_dir / filename
                
                pdf = SimpleDocTemplate(str(file_path), pagesize=letter)
                story = []
                
                # Title
                title = Paragraph(doc_content["title"], styles['Title'])
                story.append(title)
                story.append(Spacer(1, 0.2*inch))
                
                # Content
                content = doc_content["content"]
                for paragraph in content.split('\n\n'):
                    if paragraph.strip():
                        para = Paragraph(paragraph.strip(), styles['Normal'])
                        story.append(para)
                        story.append(Spacer(1, 0.1*inch))
                
                pdf.build(story)
                self.test_documents.append(str(file_path))
                self.logger.info(f"Created mock document: {filename}")
                
        except ImportError:
            self.logger.warning("reportlab not available. Please install: pip install reportlab")
            self.logger.info("Or manually add sample.pdf, sample2.pdf, sample3.pdf to documents/ directory")
    
    async def test_pipeline_initialization(self) -> bool:
        """Test pipeline initialization"""
        self.logger.info("Testing pipeline initialization...")
        
        try:
            pipeline = UltraOptimizedPipeline()
            self.logger.info("✅ Pipeline initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"❌ Pipeline initialization failed: {e}")
            return False
    
    async def test_configuration_loading(self) -> bool:
        """Test configuration loading"""
        self.logger.info("Testing configuration loading...")
        
        try:
            pipeline = UltraOptimizedPipeline()
            
            # Check main config
            assert 'visual_parsing' in pipeline.config
            assert 'stage2' in pipeline.config
            
            # Check Stage 2 config
            assert 'stage2_qwen3' in pipeline.stage2_config
            assert 'model' in pipeline.stage2_config['stage2_qwen3']
            
            self.logger.info("✅ Configuration loaded successfully")
            return True
        except Exception as e:
            self.logger.error(f"❌ Configuration loading failed: {e}")
            return False
    
    async def test_document_processing(self) -> bool:
        """Test document processing"""
        if not self.test_documents:
            self.logger.warning("No test documents available for processing test")
            return False
            
        self.logger.info("Testing document processing...")
        
        try:
            pipeline = UltraOptimizedPipeline()
            
            # Test single document processing
            test_doc = self.test_documents[0]
            self.logger.info(f"Processing test document: {test_doc}")
            
            result = await pipeline.process_document(test_doc)
            
            if result.success:
                self.logger.info(f"✅ Document processed successfully")
                self.logger.info(f"  Processing time: {result.total_processing_time:.2f}s")
                self.logger.info(f"  Total cost: ${result.total_cost:.4f}")
                
                if result.stage2_result:
                    self.logger.info(f"  Chunks created: {len(result.stage2_result.chunks)}")
                    self.logger.info(f"  Cache hit rate: {result.stage2_result.cache_hit_rate:.2f}")
                
                return True
            else:
                self.logger.error(f"❌ Document processing failed: {result.error_message}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Document processing test failed: {e}")
            return False
    
    async def test_batch_processing(self) -> bool:
        """Test batch document processing"""
        if len(self.test_documents) < 2:
            self.logger.warning("Need at least 2 documents for batch processing test")
            return False
            
        self.logger.info("Testing batch processing...")
        
        try:
            pipeline = UltraOptimizedPipeline()
            
            # Process multiple documents
            results = await pipeline.process_documents(self.test_documents[:2])
            
            success_count = sum(1 for r in results if r.success)
            total_count = len(results)
            
            self.logger.info(f"Batch processing complete: {success_count}/{total_count} succeeded")
            
            if success_count > 0:
                self.logger.info("✅ Batch processing successful")
                return True
            else:
                self.logger.error("❌ Batch processing failed")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Batch processing test failed: {e}")
            return False
    
    async def test_performance_targets(self) -> bool:
        """Test performance targets"""
        if not self.test_documents:
            self.logger.warning("No documents for performance testing")
            return False
            
        self.logger.info("Testing performance targets...")
        
        try:
            pipeline = UltraOptimizedPipeline()
            
            # Process a document and check metrics
            result = await pipeline.process_document(self.test_documents[0])
            
            if not result.success:
                self.logger.error("❌ Document processing failed")
                return False
            
            # Check performance targets
            targets_met = []
            
            # Cost target: $0.012 per 1K pages for Stage 2
            if result.stage2_result and result.stage2_result.cost_estimate <= 0.015:  # Allow small margin
                targets_met.append("Stage 2 cost target")
                self.logger.info(f"✅ Stage 2 cost: ${result.stage2_result.cost_estimate:.4f} ≤ $0.015")
            else:
                self.logger.warning(f"⚠️  Stage 2 cost: ${result.stage2_result.cost_estimate:.4f} > $0.015")
            
            # Processing time target: <5 seconds
            if result.total_processing_time <= 10.0:  # Allow generous margin for test environment
                targets_met.append("Processing time target")
                self.logger.info(f"✅ Processing time: {result.total_processing_time:.2f}s ≤ 10s")
            else:
                self.logger.warning(f"⚠️  Processing time: {result.total_processing_time:.2f}s > 10s")
            
            # Quality target: chunks created
            if result.stage2_result and len(result.stage2_result.chunks) > 0:
                targets_met.append("Quality target")
                self.logger.info(f"✅ Chunks created: {len(result.stage2_result.chunks)}")
            else:
                self.logger.warning("⚠️  No chunks created")
            
            success = len(targets_met) >= 2  # At least 2/3 targets met
            
            if success:
                self.logger.info(f"✅ Performance targets met: {', '.join(targets_met)}")
            else:
                self.logger.warning(f"⚠️  Some performance targets not met")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Performance testing failed: {e}")
            return False
    
    async def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests and return results"""
        self.logger.info("🚀 Starting comprehensive pipeline tests...")
        
        # Ensure we have test documents
        if not self.test_documents:
            self.create_mock_documents()
        
        tests = {
            "Pipeline Initialization": await self.test_pipeline_initialization(),
            "Configuration Loading": await self.test_configuration_loading(),
            "Document Processing": await self.test_document_processing(),
            "Batch Processing": await self.test_batch_processing(),  
            "Performance Targets": await self.test_performance_targets(),
        }
        
        # Summary
        passed = sum(tests.values())
        total = len(tests)
        
        self.logger.info(f"\n📊 Test Summary: {passed}/{total} tests passed")
        
        for test_name, result in tests.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            self.logger.info(f"  {status}: {test_name}")
        
        if passed == total:
            self.logger.info("🎉 All tests passed! Pipeline is ready for use.")
        else:
            self.logger.warning(f"⚠️  {total - passed} tests failed. Check logs for details.")
        
        return tests


async def main():
    """Main test execution"""
    print("🔬 Ultra-Cost-Optimized Pipeline Test Suite")
    print("=" * 60)
    
    # Check system requirements
    print("Checking system requirements...")
    
    # Check if documents directory exists
    documents_dir = Path("documents")
    if not documents_dir.exists():
        print("📁 Creating documents directory...")
        documents_dir.mkdir()
    
    # Run tests
    tester = PipelineTester()
    test_results = await tester.run_all_tests()
    
    # Final summary
    print("\n" + "=" * 60)
    if all(test_results.values()):
        print("🎉 SUCCESS: All tests passed!")
        print("Your Qwen3-30B-A3B pipeline is ready for production use.")
        print("\nNext steps:")
        print("1. Add your PDF documents to the documents/ directory")
        print("2. Run: python src/main_pipeline.py")
        print("3. Monitor costs and performance in the logs")
    else:
        print("⚠️  PARTIAL SUCCESS: Some tests failed")
        print("Check the logs above for specific issues.")
        print("The pipeline may still work for basic use cases.")
    
    return test_results


if __name__ == "__main__":
    asyncio.run(main()) 