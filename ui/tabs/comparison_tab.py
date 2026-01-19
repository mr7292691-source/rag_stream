"""
Tab 3: Flow Comparison
Compare Zero-shot vs RAG extraction methods
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime

from analysis.flow_comparison import zero_shot_extraction, rag_extraction, compare_outputs
from core.retrieval import retrieve
from ui.components import render_comparison_table
from ui.styles import get_metric_color


def render(settings: dict) -> None:
    """
    Render the Flow Comparison tab.
    
    Args:
        settings: Dict of sidebar settings
    """
    st.markdown("### ⚖️ Flow Comparison: Zero-Shot vs RAG")
    st.markdown("Compare extraction quality between direct LLM extraction and RAG-based extraction against a master output.")
    
    if "index" not in st.session_state:
        st.warning("⚠️ Please build an index first from the sidebar.")
        return
    
    # Master Output Section
    master_output = _render_master_input()
    
    if master_output:
        st.session_state.comparison_master = master_output
        
        with st.expander("📋 View Master Output Fields"):
            master_df = pd.DataFrame(master_output)
            st.dataframe(master_df[["field_name", "value"]], use_container_width=True, height=200)
    
    st.markdown("---")
    
    # Custom Prompt Section
    custom_prompt = _render_custom_prompt_input()
    if custom_prompt:
        st.session_state.custom_zs_prompt = custom_prompt
    elif "custom_zs_prompt" in st.session_state:
        del st.session_state.custom_zs_prompt
    
    st.markdown("---")
    st.markdown("#### 🚀 Step 3: Run Comparison")
    
    if "comparison_master" in st.session_state:
        _render_comparison_buttons(settings)
    else:
        st.info("👆 Please provide a master output first to enable comparison.")
    
    # Display Results
    if "comparison_result" in st.session_state and st.session_state.comparison_result:
        _display_comparison_results(settings)


def _render_master_input():
    """Render master output input section."""
    st.markdown("#### 📋 Step 1: Define Master Output (Ground Truth)")
    
    master_input_method = st.radio(
        "How would you like to provide the master output?",
        ["Upload JSON", "Paste JSON", "Use Analysis Results"],
        horizontal=True
    )
    
    master_output = None
    
    if master_input_method == "Upload JSON":
        uploaded_master = st.file_uploader("Upload master output JSON", type=["json"], key="master_upload")
        if uploaded_master:
            try:
                master_output = json.load(uploaded_master)
                if isinstance(master_output, dict):
                    master_output = [{"field_name": k, "value": v} for k, v in master_output.items()]
                st.success(f"✅ Loaded {len(master_output)} fields from master output")
            except Exception as e:
                st.error(f"Error parsing JSON: {e}")
                
    elif master_input_method == "Paste JSON":
        master_json_text = st.text_area(
            "Paste master output JSON",
            placeholder='[{"field_name": "Invoice Date", "value": "2024-01-15"}, ...]',
            height=150
        )
        if master_json_text:
            try:
                master_output = json.loads(master_json_text)
                if isinstance(master_output, dict):
                    master_output = [{"field_name": k, "value": v} for k, v in master_output.items()]
                st.success(f"✅ Parsed {len(master_output)} fields")
            except Exception as e:
                st.error(f"Invalid JSON: {e}")
                
    else:  # Use Analysis Results
        if "analysis_results" in st.session_state and st.session_state.analysis_results:
            master_output = st.session_state.analysis_results
            st.success(f"✅ Using {len(master_output)} fields from Document Analysis")
        else:
            st.warning("⚠️ No analysis results available. Run Document Analysis first or use another method.")
    
    return master_output


def _render_custom_prompt_input():
    """Render custom prompt input section."""
    st.markdown("#### 📝 Step 2: Custom Prompt for Zero-Shot (Optional)")
    st.caption("Upload or paste a custom prompt for Zero-Shot extraction. Use {FIELDS} and {DOCUMENT} as placeholders.")
    
    prompt_input_method = st.radio(
        "Custom prompt input method:",
        ["Use Default Prompt", "Upload Prompt File", "Paste Custom Prompt"],
        horizontal=True,
        key="prompt_method"
    )
    
    custom_prompt = None
    
    if prompt_input_method == "Upload Prompt File":
        uploaded_prompt = st.file_uploader("Upload prompt (.txt)", type=["txt"], key="prompt_upload")
        if uploaded_prompt:
            custom_prompt = uploaded_prompt.read().decode("utf-8")
            st.success(f"✅ Loaded custom prompt ({len(custom_prompt)} characters)")
            with st.expander("📄 Preview Uploaded Prompt"):
                st.text(custom_prompt[:500] + "..." if len(custom_prompt) > 500 else custom_prompt)
                
    elif prompt_input_method == "Paste Custom Prompt":
        default_template = """You are a document field extraction expert. Extract the following fields and provide confidence scores.

FIELDS TO EXTRACT:
{FIELDS}

DOCUMENT:
{DOCUMENT}

INSTRUCTIONS:
1. Extract exact values for each field from the document
2. For EACH field, provide the extracted value, confidence (0-100), and reasoning
3. Return as JSON format

Return ONLY a valid JSON object."""
        
        custom_prompt = st.text_area(
            "Enter your custom prompt",
            value=default_template,
            height=200,
            help="Use {FIELDS} for field list and {DOCUMENT} for document text"
        )
        if custom_prompt:
            st.info(f"📝 Using custom prompt ({len(custom_prompt)} characters)")
    else:
        st.info("📋 Using default built-in prompt for Zero-Shot extraction")
    
    return custom_prompt


def _render_comparison_buttons(settings: dict):
    """Render comparison action buttons."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        run_zeroshot_btn = st.button("🎯 Run Zero-Shot Only", use_container_width=True)
    with col2:
        run_rag_btn = st.button("🔍 Run RAG Only", use_container_width=True)
    with col3:
        run_both_btn = st.button("⚖️ Run Both & Compare", type="primary", use_container_width=True)
    
    api_delay = settings.get("api_delay", 2.0)
    top_k = settings.get("top_k", 5)
    
    # Run Zero-Shot
    if run_zeroshot_btn or run_both_btn:
        with st.spinner("🎯 Running Zero-Shot extraction..."):
            try:
                fields = st.session_state.comparison_master
                custom_prompt = st.session_state.get("custom_zs_prompt", None)
                
                if custom_prompt:
                    st.info("📝 Using custom prompt for Zero-Shot extraction")
                
                zs_results, zs_metrics = zero_shot_extraction(
                    st.session_state.client,
                    st.session_state.pdf_text,
                    fields,
                    delay_seconds=api_delay,
                    custom_prompt=custom_prompt
                )
                
                if zs_results:
                    st.session_state.zs_results = zs_results
                    st.session_state.zs_metrics = zs_metrics
                    st.success(f"✅ Zero-Shot complete! Time: {zs_metrics.get('total_time', 0)}s, Tokens: {zs_metrics.get('total_tokens', 0)}")
                else:
                    st.error(f"❌ Zero-Shot failed: {zs_metrics.get('error', 'Unknown error')}")
            except Exception as e:
                st.error(f"❌ Zero-Shot extraction error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # Run RAG
    if run_rag_btn or run_both_btn:
        with st.spinner("🔍 Running RAG extraction..."):
            try:
                fields = st.session_state.comparison_master
                
                # Add queries if missing
                for f in fields:
                    if "query" not in f:
                        f["query"] = f"What is the {f['field_name']}?"
                
                def retriever(query):
                    return retrieve(
                        st.session_state.client,
                        st.session_state.index,
                        st.session_state.chunks,
                        query,
                        top_k
                    )
                
                rag_results, rag_metrics = rag_extraction(
                    st.session_state.client,
                    fields,
                    retriever,
                    delay_seconds=api_delay
                )
                
                if rag_results:
                    st.session_state.rag_results = rag_results
                    st.session_state.rag_metrics = rag_metrics
                    st.success(f"✅ RAG complete! Time: {rag_metrics.get('total_time', 0)}s, Tokens: {rag_metrics.get('total_tokens', 0)}")
                else:
                    st.error(f"❌ RAG failed: {rag_metrics.get('error', 'Unknown error')}")
            except Exception as e:
                st.error(f"❌ RAG extraction error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # Compare if both available
    if run_both_btn or (run_zeroshot_btn and "rag_results" in st.session_state) or (run_rag_btn and "zs_results" in st.session_state):
        if "zs_results" in st.session_state and "rag_results" in st.session_state:
            try:
                with st.spinner("⚖️ Generating comparison report..."):
                    comparison = compare_outputs(
                        st.session_state.comparison_master,
                        st.session_state.zs_results,
                        st.session_state.rag_results,
                        st.session_state.pdf_text
                    )
                    st.session_state.comparison_result = comparison
                    st.success("✅ Comparison report generated!")
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Comparison generation error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


def _display_comparison_results(settings: dict):
    """Display comparison results."""
    st.markdown("---")
    st.markdown("### 📊 Comparison Results")
    
    comparison = st.session_state.comparison_result
    zs_summary = comparison["zero_shot_summary"]
    rag_summary = comparison["rag_summary"]
    zs_metrics = st.session_state.get("zs_metrics", {})
    rag_metrics = st.session_state.get("rag_metrics", {})
    
    # Summary Metrics Dashboard
    st.markdown("#### 📈 Summary Metrics")
    
    col1, col2, col3 = st.columns([1, 2, 2])
    
    with col1:
        st.markdown("**Metric**")
        st.markdown("---")
        for label in ["🎯 **Accuracy**", "📊 **Field Coverage**", "⚠️ **Hallucination**", 
                      "🤖 **LLM Tokens**", "🔢 **Embedding Tokens**", "📊 **Total Tokens**",
                      "⏱️ **Total Time**", "📞 **API Calls**"]:
            st.markdown(label)
    
    with col2:
        st.markdown("**Zero-Shot**")
        st.markdown("---")
        acc_color = get_metric_color(zs_summary["accuracy"], (50, 70))
        st.markdown(f"{acc_color} {zs_summary['accuracy']}%")
        st.markdown(f"{zs_summary['field_coverage']}%")
        halluc_color = get_metric_color(100 - zs_summary["avg_hallucination"], (60, 80))
        st.markdown(f"{halluc_color} {zs_summary['avg_hallucination']}%")
        st.markdown(f"{zs_metrics.get('llm_total_tokens', 'N/A')} (in: {zs_metrics.get('llm_input_tokens', 0)}, out: {zs_metrics.get('llm_output_tokens', 0)})")
        st.markdown(f"{zs_metrics.get('embedding_tokens', 0)}")
        st.markdown(f"**{zs_metrics.get('total_tokens', 'N/A')}**")
        st.markdown(f"{zs_metrics.get('total_time', 'N/A')}s")
        st.markdown(f"{zs_metrics.get('api_calls', 'N/A')} (LLM: {zs_metrics.get('llm_calls', 0)}, Embed: {zs_metrics.get('embedding_calls', 0)})")
    
    with col3:
        st.markdown("**RAG**")
        st.markdown("---")
        acc_color = get_metric_color(rag_summary["accuracy"], (50, 70))
        st.markdown(f"{acc_color} {rag_summary['accuracy']}%")
        st.markdown(f"{rag_summary['field_coverage']}%")
        halluc_color = get_metric_color(100 - rag_summary["avg_hallucination"], (60, 80))
        st.markdown(f"{halluc_color} {rag_summary['avg_hallucination']}%")
        st.markdown(f"{rag_metrics.get('llm_total_tokens', 'N/A')} (in: {rag_metrics.get('llm_input_tokens', 0)}, out: {rag_metrics.get('llm_output_tokens', 0)})")
        st.markdown(f"{rag_metrics.get('embedding_tokens', 0)}")
        st.markdown(f"**{rag_metrics.get('total_tokens', 'N/A')}**")
        st.markdown(f"{rag_metrics.get('total_time', 'N/A')}s")
        st.markdown(f"{rag_metrics.get('api_calls', 'N/A')} (LLM: {rag_metrics.get('llm_calls', 0)}, Embed: {rag_metrics.get('embedding_calls', 0)})")
    
    # Winner Determination
    st.markdown("---")
    zs_score = zs_summary["accuracy"] - zs_summary["avg_hallucination"]
    rag_score = rag_summary["accuracy"] - rag_summary["avg_hallucination"]
    
    if rag_score > zs_score:
        st.success(f"🏆 **Winner: RAG** (Score: {rag_score:.1f} vs {zs_score:.1f})")
        st.caption("RAG provides better accuracy with lower hallucination")
    elif zs_score > rag_score:
        st.success(f"🏆 **Winner: Zero-Shot** (Score: {zs_score:.1f} vs {rag_score:.1f})")
        st.caption("Zero-Shot provides better accuracy with lower hallucination")
    else:
        st.info("🤝 **Tie** - Both methods performed equally")
    
    # Field-Level Comparison Table
    st.markdown("---")
    st.markdown("#### 📋 Field-Level Comparison")
    field_df = render_comparison_table(comparison["fields"])
    
    # Confidence Reasoning Section
    st.markdown("---")
    st.markdown("#### 💡 Confidence Reasoning (LLM Explanations)")
    st.caption("Click on each field to see why the LLM chose this value and confidence level")
    
    zs_results = st.session_state.get("zs_results", [])
    rag_results = st.session_state.get("rag_results", [])
    
    for field in comparison["fields"]:
        field_name = field["field_name"]
        
        zs_result = next((r for r in zs_results if r["field_name"] == field_name), {})
        rag_result = next((r for r in rag_results if r["field_name"] == field_name), {})
        
        with st.expander(f"🔍 {field_name} - ZS: {field['zero_shot_confidence']}% | RAG: {field['rag_confidence']}%"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🎯 Zero-Shot**")
                st.markdown(f"**Value:** `{field['zero_shot_value']}`")
                st.markdown(f"**Confidence:** {field['zero_shot_confidence']}%")
                st.info(f"**Reasoning:** {zs_result.get('confidence_reason', 'No reasoning available')}")
            
            with col2:
                st.markdown("**🔍 RAG**")
                st.markdown(f"**Value:** `{field['rag_value']}`")
                st.markdown(f"**Confidence:** {field['rag_confidence']}%")
                st.info(f"**Reasoning:** {rag_result.get('confidence_reason', 'No reasoning available')}")
    
    # Visual Charts
    st.markdown("---")
    st.markdown("#### 📊 Visual Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Accuracy Comparison**")
        chart_data = pd.DataFrame({
            "Method": ["Zero-Shot", "RAG"],
            "Accuracy": [zs_summary["accuracy"], rag_summary["accuracy"]]
        })
        st.bar_chart(chart_data.set_index("Method"))
    
    with col2:
        st.markdown("**Hallucination Comparison**")
        chart_data = pd.DataFrame({
            "Method": ["Zero-Shot", "RAG"],
            "Hallucination": [zs_summary["avg_hallucination"], rag_summary["avg_hallucination"]]
        })
        st.bar_chart(chart_data.set_index("Method"))
    
    # Export Options
    st.markdown("---")
    st.markdown("#### 📥 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        zs_export = {
            "type": "zero_shot",
            "metrics": zs_metrics,
            "summary": zs_summary,
            "results": zs_results,
            "timestamp": datetime.now().isoformat()
        }
        st.download_button(
            "📥 output.json (Zero-Shot)",
            data=json.dumps(zs_export, indent=2),
            file_name="output.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        rag_export = {
            "type": "rag",
            "metrics": rag_metrics,
            "summary": rag_summary,
            "results": rag_results,
            "timestamp": datetime.now().isoformat()
        }
        st.download_button(
            "📥 rag.json (RAG)",
            data=json.dumps(rag_export, indent=2),
            file_name="rag.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col3:
        full_report = {
            "comparison": comparison,
            "zero_shot": zs_export,
            "rag": rag_export,
            "winner": "RAG" if rag_score > zs_score else "Zero-Shot" if zs_score > rag_score else "Tie",
            "timestamp": datetime.now().isoformat()
        }
        st.download_button(
            "📥 comparison_report.json",
            data=json.dumps(full_report, indent=2),
            file_name=f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    st.download_button(
        "📥 Download Field Comparison (CSV)",
        data=field_df.to_csv(index=False),
        file_name=f"field_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )
