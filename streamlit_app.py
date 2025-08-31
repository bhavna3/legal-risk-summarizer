"""
AI-Powered Legal Contract Analyzer - Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import tempfile
import os
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our modules
from src.analysis import LegalAnalyzer
from src.config import UI_CONFIG, RISK_LEVELS

# Page configuration
st.set_page_config(
    page_title=UI_CONFIG["page_title"],
    page_icon=UI_CONFIG["page_icon"],
    layout=UI_CONFIG["layout"],
    initial_sidebar_state=UI_CONFIG["initial_sidebar_state"]
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
    .risk-medium {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
    }
    .risk-low {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">⚖️ AI-Powered Legal Contract Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Analyze legal documents with AI-powered clause classification, risk detection, and simplified summaries
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Analysis Options")
        
        # Model options
        st.subheader("Model Settings")
        use_gpu = st.checkbox("Use GPU (if available)", value=True)
        
        # Analysis options
        st.subheader("Analysis Options")
        include_rag = st.checkbox("Include RAG Analysis", value=True)
        generate_summaries = st.checkbox("Generate Simplified Summaries", value=True)
        
        # About section
        st.header("ℹ️ About")
        st.markdown("""
        This AI-powered legal contract analyzer uses:
        - **HuggingFace Transformers** for clause classification
        - **LoRA fine-tuning** for domain adaptation
        - **LangChain RAG** for enhanced understanding
        - **CUAD dataset** for training
        """)
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Document Upload", "📊 Analysis Results", "⚠️ Risk Analysis", "🔍 Detailed Analysis"])
    
    with tab1:
        document_upload_tab()
    
    with tab2:
        analysis_results_tab()
    
    with tab3:
        risk_analysis_tab()
    
    with tab4:
        detailed_analysis_tab()

def document_upload_tab():
    """Document upload and processing tab."""
    st.header("📄 Upload Legal Document")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a legal document (PDF, DOCX, TXT)",
        type=['pdf', 'docx', 'txt'],
        help="Upload a legal contract or agreement for analysis"
    )
    
    if uploaded_file is not None:
        # Display file info
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.1f} KB",
            "File type": uploaded_file.type
        }
        
        st.subheader("📋 File Information")
        for key, value in file_details.items():
            st.write(f"**{key}:** {value}")
        
        # Process button
        if st.button("🚀 Analyze Document", type="primary"):
            with st.spinner("Analyzing document... This may take a few minutes."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    # Initialize analyzer
                    analyzer = LegalAnalyzer(use_gpu=st.session_state.get('use_gpu', True))
                    
                    # Analyze document
                    analysis_results = analyzer.analyze_document(tmp_file_path)
                    
                    # Store results in session state
                    st.session_state['analysis_results'] = analysis_results
                    st.session_state['document_processed'] = True
                    
                    # Clean up temporary file
                    os.unlink(tmp_file_path)
                    
                    st.success("✅ Document analysis completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error analyzing document: {str(e)}")
                    logger.error(f"Analysis error: {str(e)}")

def analysis_results_tab():
    """Display analysis results tab."""
    st.header("📊 Analysis Results")
    
    if not st.session_state.get('document_processed', False):
        st.info("👆 Please upload and analyze a document first.")
        return
    
    analysis_results = st.session_state.get('analysis_results')
    if not analysis_results:
        st.error("No analysis results found.")
        return
    
    # Executive Summary
    st.subheader("📋 Executive Summary")
    executive_summary = analysis_results['summaries']['executive_summary']
    st.markdown(executive_summary)
    
    # Risk Assessment
    st.subheader("⚠️ Risk Assessment")
    risk_level = analysis_results['risks']['overall_risk']
    risk_color = RISK_LEVELS[risk_level]['color']
    
    # Risk level indicator
    risk_class = f"risk-{risk_level.lower()}"
    st.markdown(f"""
    <div class="{risk_class}">
        <h4>Overall Risk Level: {risk_level}</h4>
        <p>{analysis_results['summaries']['risk_summary']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Metrics
    st.subheader("📈 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Clauses",
            analysis_results['analysis_metadata']['total_clauses']
        )
    
    with col2:
        st.metric(
            "High Risk Clauses",
            len(analysis_results['risks']['high_risk_clauses'])
        )
    
    with col3:
        st.metric(
            "Processing Time",
            f"{analysis_results['analysis_metadata']['processing_time']:.1f}s"
        )
    
    with col4:
        st.metric(
            "Confidence Score",
            f"{analysis_results['analysis_metadata']['confidence_score']:.2f}"
        )
    
    # Action Items
    st.subheader("🎯 Recommended Actions")
    action_items = analysis_results['summaries']['action_items']
    for item in action_items:
        st.write(f"• {item}")
    
    # Export Results
    st.subheader("📥 Export Results")
    if st.button("📊 Export Analysis to CSV"):
        csv_data = create_analysis_csv(analysis_results)
        st.download_button(
            label="💾 Download CSV Report",
            data=csv_data,
            file_name=f"legal_analysis_{int(time.time())}.csv",
            mime="text/csv"
        )

def create_analysis_csv(analysis_results):
    """Create CSV data for analysis results without duplicates."""
    import io
    
    # Prepare data for CSV
    csv_data = []
    
    # Add overall analysis info
    csv_data.append({
        'Section': 'Overall Analysis',
        'Type': 'Document Info',
        'Value': f"Risk Level: {analysis_results['risks']['overall_risk']}",
        'Details': f"Total Clauses: {analysis_results['analysis_metadata']['total_clauses']}"
    })
    
    # Add high-risk clauses (only once)
    for clause in analysis_results['risks']['high_risk_clauses']:
        csv_data.append({
            'Section': 'High Risk Clauses',
            'Type': clause.get('type', 'Unknown'),
            'Value': 'HIGH',
            'Details': clause.get('text', '')[:200] + '...' if len(clause.get('text', '')) > 200 else clause.get('text', '')
        })
    
    # Add medium-risk clauses (only once)
    for clause in analysis_results['risks']['medium_risk_clauses']:
        csv_data.append({
            'Section': 'Medium Risk Clauses',
            'Type': clause.get('type', 'Unknown'),
            'Value': 'MEDIUM',
            'Details': clause.get('text', '')[:200] + '...' if len(clause.get('text', '')) > 200 else clause.get('text', '')
        })
    
    # Add recommendations
    for i, item in enumerate(analysis_results['summaries']['action_items']):
        csv_data.append({
            'Section': 'Recommendations',
            'Type': f'Action {i+1}',
            'Value': 'RECOMMENDATION',
            'Details': item
        })
    
    # Convert to CSV
    df = pd.DataFrame(csv_data)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()

def risk_analysis_tab():
    """Display simplified risk analysis tab."""
    st.header("⚠️ Risk Analysis")
    
    if not st.session_state.get('document_processed', False):
        st.info("👆 Please upload and analyze a document first.")
        return
    
    analysis_results = st.session_state.get('analysis_results')
    if not analysis_results:
        st.error("No analysis results found.")
        return
    
    # Overall Risk Level
    st.subheader("🎯 Overall Risk Assessment")
    overall_risk = analysis_results['risks']['overall_risk']
    risk_color = RISK_LEVELS[overall_risk]['color']
    
    st.markdown(f"""
    <div style="text-align: center; padding: 20px; background-color: {risk_color}20; border-radius: 10px; border-left: 5px solid {risk_color};">
        <h2 style="color: {risk_color}; margin: 0;">{overall_risk} RISK</h2>
        <p style="margin: 5px 0 0 0; color: #666;">Overall Contract Risk Level</p>
    </div>
    """, unsafe_allow_html=True)
    
    # High Risk Clauses
    high_risk_clauses = analysis_results['risks']['high_risk_clauses']
    if high_risk_clauses:
        st.subheader("🚨 High Risk Clauses")
        for clause in high_risk_clauses:
            with st.expander(f"🔴 {clause.get('type', 'Unknown')} - {clause.get('text', '')[:100]}..."):
                st.write(f"**Clause Type:** {clause.get('type', 'Unknown')}")
                st.write(f"**Risk Level:** {clause.get('risk_level', 'HIGH')}")
                st.write(f"**Confidence:** {clause.get('risk_confidence', 0):.2f}")
                st.write(f"**Text:** {clause.get('text', '')}")
    
    # Medium Risk Clauses
    medium_risk_clauses = analysis_results['risks']['medium_risk_clauses']
    if medium_risk_clauses:
        st.subheader("⚠️ Medium Risk Clauses")
        for clause in medium_risk_clauses:
            with st.expander(f"🟡 {clause.get('type', 'Unknown')} - {clause.get('text', '')[:100]}..."):
                st.write(f"**Clause Type:** {clause.get('type', 'Unknown')}")
                st.write(f"**Risk Level:** {clause.get('risk_level', 'MEDIUM')}")
                st.write(f"**Confidence:** {clause.get('risk_confidence', 0):.2f}")
                st.write(f"**Text:** {clause.get('text', '')}")
    
    # Risk Summary
    st.subheader("📋 Risk Summary")
    risk_distribution = analysis_results['risks']['risk_distribution']
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("High Risk", risk_distribution.get('HIGH', 0), delta=None)
    with col2:
        st.metric("Medium Risk", risk_distribution.get('MEDIUM', 0), delta=None)
    with col3:
        st.metric("Low Risk", risk_distribution.get('LOW', 0), delta=None)

def detailed_analysis_tab():
    """Display detailed analysis tab."""
    st.header("🔍 Detailed Analysis")
    
    if not st.session_state.get('document_processed', False):
        st.info("👆 Please upload and analyze a document first.")
        return
    
    analysis_results = st.session_state.get('analysis_results')
    if not analysis_results:
        st.error("No analysis results found.")
        return
    
    # Detailed Clause Analysis
    st.subheader("📋 Detailed Clause Analysis")
    
    classified_clauses = analysis_results['clauses']['classified_clauses']
    if classified_clauses:
        # Create DataFrame for detailed view
        clause_data = []
        for clause in classified_clauses:
            clause_data.append({
                'Type': clause.get('type', 'Unknown'),
                'Predicted Type': clause.get('predicted_type', 'Unknown'),
                'Confidence': f"{clause.get('confidence', 0):.2f}",
                'Length': clause.get('length', 0),
                'Text Preview': clause.get('text', '')[:100] + '...' if len(clause.get('text', '')) > 100 else clause.get('text', '')
            })
        
        df_clauses = pd.DataFrame(clause_data)
        st.dataframe(df_clauses, use_container_width=True)
    
    # Specific Risks
    st.subheader("🚨 Specific Risk Factors")
    specific_risks = analysis_results['risks'].get('specific_risks', [])
    
    if specific_risks:
        risk_data = []
        for risk in specific_risks:
            risk_data.append({
                'Risk Type': risk.get('risk_type', 'Unknown'),
                'Severity': risk.get('severity', 'Unknown'),
                'Indicator': risk.get('indicator', 'Unknown'),
                'Context': risk.get('context', '')[:100] + '...' if len(risk.get('context', '')) > 100 else risk.get('context', '')
            })
        
        df_risks = pd.DataFrame(risk_data)
        st.dataframe(df_risks, use_container_width=True)
    
    # RAG Analysis
    if 'rag_analysis' in analysis_results:
        st.subheader("🧠 AI-Enhanced Insights")
        rag_summary = analysis_results['rag_analysis'].get('overall_summary', '')
        if rag_summary:
            st.markdown(rag_summary)
    
    # Raw Analysis Data (expandable)
    with st.expander("🔧 Raw Analysis Data"):
        st.json(analysis_results)

if __name__ == "__main__":
    main() 