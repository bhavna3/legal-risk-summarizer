#!/usr/bin/env python3
"""
Test script for the Legal Contract Analyzer

This script demonstrates the functionality of the legal analyzer
by processing a sample contract.
"""

import logging
from pathlib import Path

from src.analysis import LegalAnalyzer
from src.config import UI_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_legal_analyzer():
    """Test the legal analyzer with a sample contract."""
    
    print("=" * 60)
    print("🧪 Testing Legal Contract Analyzer")
    print("=" * 60)
    
    try:
        # Initialize analyzer
        print("📋 Initializing Legal Analyzer...")
        analyzer = LegalAnalyzer(use_gpu=False)  # Use CPU for testing
        print("✅ Analyzer initialized successfully")
        
        # Test with sample contract
        sample_contract_path = "data/sample_contract.txt"
        
        if not Path(sample_contract_path).exists():
            print(f"❌ Sample contract not found at {sample_contract_path}")
            return
        
        print(f"\n📄 Analyzing sample contract: {sample_contract_path}")
        print("-" * 40)
        
        # Analyze document
        analysis_results = analyzer.analyze_document(sample_contract_path)
        
        # Display results
        print("\n📊 ANALYSIS RESULTS")
        print("=" * 40)
        
        # Document info
        doc_info = analysis_results["document_info"]
        print(f"📋 Document: {doc_info['filename']}")
        print(f"📏 Size: {doc_info['file_size']}")
        print(f"⏱️  Processing time: {analysis_results['analysis_metadata']['processing_time']:.2f} seconds")
        
        # Risk assessment
        risk_level = analysis_results["risks"]["overall_risk"]
        print(f"\n⚠️  RISK ASSESSMENT")
        print(f"Overall Risk Level: {risk_level}")
        
        risk_distribution = analysis_results["risks"]["risk_distribution"]
        print(f"Risk Distribution:")
        for level, count in risk_distribution.items():
            print(f"  - {level}: {count} clauses")
        
        # Clause analysis
        clause_stats = analysis_results["clauses"]["clause_statistics"]
        print(f"\n📋 CLAUSE ANALYSIS")
        print(f"Total Clauses: {clause_stats['total_clauses']}")
        print(f"Clause Types Found: {clause_stats['clause_types_found']}")
        print(f"Average Confidence: {clause_stats['average_confidence']:.2f}")
        
        # Top clauses by type
        clause_distribution = analysis_results["clauses"]["clause_distribution"]
        print(f"\n📊 CLAUSE DISTRIBUTION")
        for clause_type, count in sorted(clause_distribution.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                print(f"  - {clause_type}: {count}")
        
        # Executive summary
        print(f"\n📝 EXECUTIVE SUMMARY")
        print("-" * 30)
        executive_summary = analysis_results["summaries"]["executive_summary"]
        print(executive_summary)
        
        # Action items
        print(f"\n🎯 RECOMMENDED ACTIONS")
        print("-" * 30)
        action_items = analysis_results["summaries"]["action_items"]
        for item in action_items:
            print(f"  • {item}")
        
        # Specific risks
        specific_risks = analysis_results["risks"]["specific_risks"]
        if specific_risks:
            print(f"\n🚨 SPECIFIC RISK FACTORS")
            print("-" * 30)
            for risk in specific_risks[:5]:  # Show top 5
                print(f"  • {risk['risk_type']} ({risk['severity']} severity)")
                print(f"    Indicator: {risk['indicator']}")
        
        print("\n✅ Analysis completed successfully!")
        
        # Generate report
        print(f"\n📄 GENERATING ANALYSIS REPORT...")
        report = analyzer.get_analysis_report(analysis_results)
        print("✅ Report generated successfully")
        
        return analysis_results
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        logger.error(f"Test failed: {str(e)}")
        return None


def test_individual_components():
    """Test individual components of the analyzer."""
    
    print("\n" + "=" * 60)
    print("🔧 Testing Individual Components")
    print("=" * 60)
    
    try:
        from src.preprocessing import DocumentProcessor, TextCleaner, ClauseExtractor
        
        # Test document processor
        print("\n📄 Testing Document Processor...")
        processor = DocumentProcessor()
        sample_path = "data/sample_contract.txt"
        
        if Path(sample_path).exists():
            result = processor.preprocess_document(sample_path)
            print(f"✅ Document processed: {result['num_chunks']} chunks, {result['total_length']} characters")
        
        # Test text cleaner
        print("\n🧹 Testing Text Cleaner...")
        cleaner = TextCleaner()
        sample_text = "This is a SAMPLE legal document with CONFIDENTIAL information."
        cleaned_text = cleaner.clean_legal_text(sample_text)
        print(f"✅ Text cleaned: '{cleaned_text}'")
        
        # Test clause extractor
        print("\n📋 Testing Clause Extractor...")
        extractor = ClauseExtractor()
        clauses = extractor.extract_clauses(sample_text)
        print(f"✅ Clauses extracted: {len([c for c in clauses.values() if c])} types found")
        
        print("\n✅ All component tests passed!")
        
    except Exception as e:
        print(f"❌ Component test failed: {str(e)}")
        logger.error(f"Component test failed: {str(e)}")


def main():
    """Main test function."""
    
    print("🚀 Starting Legal Contract Analyzer Tests")
    print("=" * 60)
    
    # Test individual components
    test_individual_components()
    
    # Test full analyzer
    results = test_legal_analyzer()
    
    if results:
        print("\n🎉 All tests completed successfully!")
        print("\n💡 To run the web application:")
        print("   streamlit run streamlit_app.py")
    else:
        print("\n❌ Some tests failed. Please check the logs.")


if __name__ == "__main__":
    main() 