import streamlit as st
import pandas as pd
import numpy as np
from src.data_processing import (load_genomic_data, get_sequence_stats,
                               process_sequences, load_sample_data,
                               export_results)
from src.visualizations import (plot_kmer_distribution, plot_sequence_length_distribution,
                              plot_sequence_similarity_matrix, plot_sequence_clusters,
                              plot_feature_importance, plot_sequence_composition)
from src.ai_analysis import analyze_sequences

# Page configuration
st.set_page_config(
    page_title="Biomarker Discovery Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# Title and description
st.title("🧬 Viral Disease Biomarker Discovery Platform")
st.markdown("""
This platform uses machine learning to analyze viral genomic sequences and identify potential biomarkers.
Upload your FASTA file or use sample data to get started.
""")

# Sidebar configuration
with st.sidebar:
    st.header("Analysis Parameters")
    k_mer_size = st.slider("K-mer Size", 4, 8, 6,
                          help="Size of k-mers for sequence analysis")

# Main content
tab1, tab2, tab3 = st.tabs([
    "📊 Data Analysis",
    "🧬 AI Analysis",
    "💾 Export Results"
])

with tab1:
    st.header("Data Upload and Processing")
    data_option = st.radio(
        "Choose data source",
        ["Upload FASTA file", "Use sample data"]
    )

    sequences_df = None
    if data_option == "Upload FASTA file":
        uploaded_file = st.file_uploader("Upload FASTA file", type=['fasta', 'fa'])
        if uploaded_file:
            sequences_df = load_genomic_data(uploaded_file)
    else:
        sequences_df = load_sample_data()
        st.success("Sample data loaded successfully!")

    if sequences_df is not None:
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        stats = get_sequence_stats(sequences_df)

        with col1:
            st.metric("Total Sequences", stats['Total Sequences'])
        with col2:
            st.metric("Average Length", stats['Average Length'])
        with col3:
            st.metric("GC Content (%)", stats['GC Content (%)'])

        # Sequence analysis
        st.subheader("Sequence Analysis")
        col1, col2 = st.columns(2)

        with col1:
            fig = plot_sequence_length_distribution(sequences_df)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = plot_sequence_composition(sequences_df)
            st.plotly_chart(fig, use_container_width=True)

        # K-mer analysis
        st.subheader("K-mer Analysis")
        kmer_freq = process_sequences(sequences_df, k=k_mer_size)
        fig = plot_kmer_distribution(kmer_freq)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    if sequences_df is not None:
        st.header("Advanced Sequence Analysis")

        if st.button("Run Analysis"):
            with st.spinner("Analyzing sequences..."):
                try:
                    analysis_results = analyze_sequences(sequences_df)
                    st.session_state.analysis_results = analysis_results

                    if analysis_results is not None:
                        # Plot similarity matrix
                        st.subheader("Sequence Similarity Analysis")
                        st.info("""
                        This visualization shows how similar the sequences are to each other.
                        Brighter colors indicate higher similarity between sequences.
                        """)
                        fig = plot_sequence_similarity_matrix(analysis_results['similarity_matrix'])
                        st.plotly_chart(fig, use_container_width=True)

                        # Plot clusters
                        st.subheader("Sequence Clustering")
                        st.info("""
                        Sequences are grouped into clusters based on their features.
                        Points of the same color belong to the same cluster.
                        """)
                        fig = plot_sequence_clusters(
                            analysis_results['reduced_embeddings'],
                            analysis_results['clusters']
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Show feature importance
                        st.subheader("Feature Analysis")
                        fig = plot_feature_importance(analysis_results['feature_importance'])
                        st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.info("Try with a smaller dataset or check your sequence format.")

with tab3:
    if sequences_df is not None and st.session_state.analysis_results is not None:
        st.header("Export Results")
        if st.button("Export Analysis Results"):
            results = {
                'sequences': sequences_df.to_dict(orient='records'),
                'analysis': {
                    'clusters': st.session_state.analysis_results['clusters'].tolist(),
                    'feature_importance': st.session_state.analysis_results['feature_importance']
                }
            }

            st.download_button(
                "Download Results (JSON)",
                data=pd.json_normalize(results).to_json(),
                file_name="biomarker_analysis_results.json",
                mime="application/json"
            )

# Footer
st.markdown("---")
st.markdown("""
Built with Streamlit • Scikit-learn • Biopython  
Created by ABHISHEK S R
""")