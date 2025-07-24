#!/usr/bin/env python3
"""
Comprehensive Test Suite for Ultra-Cost-Optimized Pipeline
Consolidated testing for all pipeline modes: Local, LangChain, and RunPod
"""

import os
import sys
import asyncio
import logging
import time
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pytest

# Add src directory to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

# Mock classes for testing without heavy dependencies
class MockPaddleOCR:
    def __init__(self, *args, **kwargs):
        pass
    
    def ocr(self, image, cls=True):
        # Mock OCR results
        return [[[(["100", "200", "300", "400"], ("Sample text content", 0.95)),
                  (["150", "250", "350", "450"], ("More sample text", 0.87))],]]

class MockCostTracker:
    def __init__(self, *args, **kwargs):
        self.total_cost = 0.0
        self.operations = {}
    
    def start_operation(self, operation_name: str) -> str:
        op_id = f"op_{len(self.operations)}"
        self.operations[op_id] = {"name": operation_name, "start_time": time.time()}
        return op_id
    
    def end_operation(self, op_id: str) -> float:
        if op_id in self.operations:
            cost = 0.001  # Mock cost
            self.total_cost += cost
            return cost
        return 0.0
    
    def add_cost(self, stage: str, cost: float, **kwargs):
        self.total_cost += cost

# Mock results for testing
@dataclass
class MockVisualParsingResult:
    text_content: str
    latex_formulas: List[str] = None
    tables: List[Dict] = None
    layout_info: Dict = None
    confidence_scores: Dict[str, float] = None
    processing_time: float = 0.0
    cost_estimate: float = 0.0

@dataclass
class MockQwen3Result:
    chunks: List[Dict]
    processing_mode: str
    thinking_used: bool
    total_tokens: int
    cost: float
    processing_time: float

class ComprehensivePipelineTester:
    """Unified testing class for all pipeline modes"""
    
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
        docs_dir = Path("docs")
        if not docs_dir.exists():
            docs_dir.mkdir()
            
        test_files = ["sample.pdf", "sample2.pdf", "sample3.pdf", "academic_paper.pdf"]
        available_files = []
        
        for file in test_files:
            file_path = docs_dir / file
            if file_path.exists():
                available_files.append(str(file_path))
        
        return available_files
    
    def create_mock_document(self, content: str, filename: str) -> str:
        """Create a mock PDF document for testing"""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph
            from reportlab.lib.styles import getSampleStyleSheet
            
            temp_dir = Path("temp")
            temp_dir.mkdir(exist_ok=True)
            
            doc_path = temp_dir / filename
            doc = SimpleDocTemplate(str(doc_path), pagesize=letter)
            styles = getSampleStyleSheet()
            
            story = [Paragraph(content, styles['Normal'])]
            doc.build(story)
            
            return str(doc_path)
            
        except ImportError:
            # Fallback: create a text file for testing
            temp_dir = Path("temp")
            temp_dir.mkdir(exist_ok=True)
            
            text_path = temp_dir / f"{filename}.txt"
            text_path.write_text(content)
            return str(text_path)
    
    async def test_local_pipeline(self) -> Dict[str, Any]:
        """Test traditional local pipeline"""
        self.logger.info("🧪 Testing Local Pipeline Mode")
        
        try:
            from pipeline.main_pipeline import UltraOptimizedPipeline
            
            pipeline = UltraOptimizedPipeline()
            self.logger.info("✅ Local pipeline initialized")
            
            # Test with mock document if no real documents available
            if not self.test_documents:
                test_doc = self.create_mock_document(
                    "This is a test academic paper with complex formulas and tables.",
                    "test_local.pdf"
                )
                test_docs = [test_doc]
            else:
                test_docs = self.test_documents[:1]  # Use first document
            
            start_time = time.time()
            result = await pipeline.process_document(test_docs[0])
            processing_time = time.time() - start_time
            
            return {
                "mode": "local",
                "status": "success" if result else "failed",
                "processing_time": processing_time,
                "document": test_docs[0],
                "result": result
            }
            
        except Exception as e:
            self.logger.error(f"❌ Local pipeline test failed: {e}")
            return {
                "mode": "local",
                "status": "failed",
                "error": str(e)
            }
    
    async def test_langchain_orchestration(self) -> Dict[str, Any]:
        """Test LangChain orchestrated pipeline"""
        self.logger.info("🔗 Testing LangChain Orchestration Mode")
        
        try:
            from pipeline.pipeline_orchestrator import AdvancedPipelineOrchestrator
            
            orchestrator = AdvancedPipelineOrchestrator()
            self.logger.info("✅ LangChain orchestrator initialized")
            self.logger.info("   📋 Sequential chains: Visual → Qwen3 → Qdrant")
            self.logger.info("   🧠 Memory: Conversation buffer enabled")
            self.logger.info("   📈 Callbacks: GPU monitoring + cost tracking")
            
            # Test with mock document
            if not self.test_documents:
                test_doc = self.create_mock_document(
                    "Academic research paper discussing machine learning optimization techniques.",
                    "test_langchain.pdf"
                )
            else:
                test_doc = self.test_documents[0]
            
            start_time = time.time()
            result = await orchestrator.run({"document_path": test_doc})
            processing_time = time.time() - start_time
            
            return {
                "mode": "langchain",
                "status": "success" if result else "failed",
                "processing_time": processing_time,
                "document": test_doc,
                "result": result
            }
            
        except Exception as e:
            self.logger.error(f"❌ LangChain orchestration test failed: {e}")
            return {
                "mode": "langchain",
                "status": "failed",
                "error": str(e)
            }
    
    async def test_stage2_isolation(self) -> Dict[str, Any]:
        """Test Stage 2 (Qwen3) in isolation with mocks"""
        self.logger.info("🤖 Testing Stage 2 (Qwen3) Isolation")
        
        try:
            # Mock heavy dependencies
            sys.modules['paddleocr'] = type(sys)('paddleocr')
            sys.modules['paddleocr'].PaddleOCR = MockPaddleOCR
            
            from pipeline.stage2_qwen3_optimized import QwenStage2
            
            stage2 = QwenStage2()
            self.logger.info("✅ Stage 2 initialized with mocked dependencies")
            
            # Create mock visual parsing result
            mock_visual_result = MockVisualParsingResult(
                text_content="This is sample text extracted from a document. It contains technical information about machine learning algorithms and optimization techniques.",
                latex_formulas=["\\alpha = \\beta + \\gamma", "E = mc^2"],
                tables=[{"header": ["Method", "Accuracy"], "rows": [["SVM", "0.95"], ["RF", "0.92"]]}],
                confidence_scores={"ocr": 0.95, "layout": 0.87}
            )
            
            start_time = time.time()
            result = await stage2.process_visual_result(mock_visual_result)
            processing_time = time.time() - start_time
            
            return {
                "mode": "stage2_isolation",
                "status": "success" if result else "failed",
                "processing_time": processing_time,
                "thinking_mode": getattr(result, 'thinking_used', False) if result else False,
                "chunks_generated": len(getattr(result, 'chunks', [])) if result else 0
            }
            
        except Exception as e:
            self.logger.error(f"❌ Stage 2 isolation test failed: {e}")
            return {
                "mode": "stage2_isolation",
                "status": "failed",
                "error": str(e)
            }
    
    async def test_runpod_mode(self) -> Dict[str, Any]:
        """Test RunPod mode (if configured)"""
        self.logger.info("☁️ Testing RunPod Mode")
        
        try:
            # Check if RunPod configuration exists
            runpod_config_path = Path("config/runpod_endpoints.yaml")
            if not runpod_config_path.exists():
                self.logger.warning("⚠️ RunPod configuration not found, skipping RunPod tests")
                return {
                    "mode": "runpod",
                    "status": "skipped",
                    "reason": "No RunPod configuration found"
                }
            
            from pipeline.runpod_langchain_api import RunPodLangChainPipeline
            
            pipeline = RunPodLangChainPipeline()
            self.logger.info("✅ RunPod pipeline initialized")
            
            # Test basic connectivity
            # This would typically test the RunPod endpoints
            return {
                "mode": "runpod",
                "status": "configured",
                "note": "RunPod endpoints configured but not tested (requires active pods)"
            }
            
        except Exception as e:
            self.logger.error(f"❌ RunPod mode test failed: {e}")
            return {
                "mode": "runpod",
                "status": "failed",
                "error": str(e)
            }
    
    def run_performance_tests(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance of test results"""
        self.logger.info("📊 Running Performance Analysis")
        
        successful_results = [r for r in results if r.get("status") == "success"]
        
        if not successful_results:
            return {"status": "no_successful_tests"}
        
        avg_processing_time = sum(r.get("processing_time", 0) for r in successful_results) / len(successful_results)
        
        performance_analysis = {
            "total_tests": len(results),
            "successful_tests": len(successful_results),
            "success_rate": len(successful_results) / len(results) * 100,
            "average_processing_time": avg_processing_time,
            "fastest_mode": min(successful_results, key=lambda x: x.get("processing_time", float('inf'))).get("mode"),
            "performance_targets": {
                "target_cost_per_1k_pages": 0.15,
                "target_processing_time": 60.0,  # seconds per document
                "target_success_rate": 95.0      # percentage
            }
        }
        
        # Performance evaluation
        meets_time_target = avg_processing_time <= performance_analysis["performance_targets"]["target_processing_time"]
        meets_success_target = performance_analysis["success_rate"] >= performance_analysis["performance_targets"]["target_success_rate"]
        
        performance_analysis["performance_grade"] = "PASS" if (meets_time_target and meets_success_target) else "NEEDS_IMPROVEMENT"
        
        return performance_analysis
    
    def display_results(self, results: List[Dict[str, Any]], performance: Dict[str, Any]):
        """Display comprehensive test results"""
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE PIPELINE TEST RESULTS")
        print("="*80)
        
        for result in results:
            mode = result.get("mode", "unknown").upper()
            status = result.get("status", "unknown")
            
            if status == "success":
                time_info = f"({result.get('processing_time', 0):.2f}s)"
                print(f"✅ {mode:<20} {time_info}")
                
                # Additional info for specific modes
                if result.get("chunks_generated"):
                    print(f"   📄 Chunks generated: {result['chunks_generated']}")
                if result.get("thinking_mode"):
                    print(f"   🧠 Thinking mode: {result['thinking_mode']}")
                    
            elif status == "skipped":
                reason = result.get("reason", "Unknown reason")
                print(f"⏭️  {mode:<20} (Skipped: {reason})")
                
            else:
                error = result.get("error", "Unknown error")
                print(f"❌ {mode:<20} (Error: {error})")
        
        print(f"\n📈 PERFORMANCE ANALYSIS")
        print(f"   Success Rate: {performance.get('success_rate', 0):.1f}%")
        print(f"   Average Time: {performance.get('average_processing_time', 0):.2f}s")
        print(f"   Fastest Mode: {performance.get('fastest_mode', 'N/A')}")
        print(f"   Overall Grade: {performance.get('performance_grade', 'N/A')}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS")
        if performance.get('success_rate', 0) < 95:
            print("   • Investigate failed test modes for stability issues")
        if performance.get('average_processing_time', 0) > 60:
            print("   • Consider optimization for processing time")
        print("   • Run tests with actual documents for production validation")
    
    async def run_all_tests(self):
        """Run all comprehensive tests"""
        print("🚀 Starting Comprehensive Pipeline Tests")
        print("Testing all modes: Local, LangChain, Stage2 Isolation, RunPod")
        print("="*80)
        
        test_methods = [
            self.test_local_pipeline,
            self.test_langchain_orchestration, 
            self.test_stage2_isolation,
            self.test_runpod_mode
        ]
        
        results = []
        for test_method in test_methods:
            try:
                result = await test_method()
                results.append(result)
            except Exception as e:
                self.logger.error(f"Test method {test_method.__name__} failed: {e}")
                results.append({
                    "mode": test_method.__name__.replace("test_", ""),
                    "status": "failed",
                    "error": str(e)
                })
        
        # Performance analysis
        performance = self.run_performance_tests(results)
        
        # Display results
        self.display_results(results, performance)
        
        return results, performance

async def main():
    """Main test runner"""
    tester = ComprehensivePipelineTester()
    results, performance = await tester.run_all_tests()
    
    # Return exit code based on results
    failed_tests = [r for r in results if r.get("status") == "failed"]
    return 1 if failed_tests else 0

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"💥 Test suite failed: {e}")
        sys.exit(1) 