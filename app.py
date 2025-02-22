import streamlit as st
import pandas as pd
import numpy as np
from src.data_processing import (load_genomic_data, get_sequence_stats,
                              process_sequences, load_sample_data,
                              export_results)
from src.model import (build_cnn_model, train_model_with_progress,
                     prepare_sequence_data, get_feature_importance)
from src.evaluate import evaluate_model, plot_roc_curve, cross_validate
from src.visualizations import (plot_kmer_distribution, plot_sequence_length_distribution,
                             plot_training_history, plot_feature_importance,
                             plot_confusion_matrix, plot_sequence_composition,
                             plot_sequence_similarity_matrix, plot_sequence_clusters)
from src.ai_analysis import analyze_sequences

# Page configuration
st.set_page_config(
    page_title="Biomarker Discovery Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #2ecc71;
    }
    .stButton>button {
        background-color: #2ecc71;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #27ae60;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# Title and description
st.title("🧬 Viral Disease Biomarker Discovery Platform")
st.markdown("""
This platform uses machine learning and deep learning to analyze viral genomic sequences and identify potential biomarkers.
Upload your FASTA file or use sample data to get started.
""")

# Sidebar configuration
with st.sidebar:
    st.header("Analysis Parameters")
    k_mer_size = st.slider("K-mer Size", 4, 8, 6,
                          help="Size of k-mers for sequence analysis")
    epochs = st.slider("Training Epochs", 5, 50, 10,
                      help="Number of training epochs")
    learning_rate = st.select_slider(
        "Learning Rate",
        options=[0.1, 0.01, 0.001, 0.0001],
        value=0.001,
        format_func=lambda x: f"{x:.4f}",
        help="Learning rate for model training"
    )

# Main content
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Analysis",
    "🧬 AI Analysis",
    "🧪 Model Training",
    "📈 Evaluation",
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
    if sequences_df is not None:
        st.header("Model Training")

        if st.button("Train Model"):
            with st.spinner("Preparing data and training model..."):
                try:
                    # Prepare data
                    X, y = prepare_sequence_data(sequences_df, kmer_freq, k=k_mer_size)

                    # Split data
                    from sklearn.model_selection import train_test_split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )

                    # Build and train model
                    input_shape = (X.shape[1], 1)
                    model = build_cnn_model(input_shape, learning_rate)

                    # Store model and data in session state
                    st.session_state.model = model
                    st.session_state.processed_data = {
                        'X_train': X_train,
                        'X_test': X_test,
                        'y_train': y_train,
                        'y_test': y_test,
                        'kmer_freq': kmer_freq
                    }

                    # Train model with progress tracking
                    history = train_model_with_progress(
                        model, X_train, y_train,
                        X_test, y_test, epochs
                    )
                    st.session_state.training_history = history

                    st.success("Model training completed!")

                except Exception as e:
                    st.error(f"Error during model training: {str(e)}")
                    st.info("Try adjusting the model parameters or using a smaller dataset.")

with tab4:
    if st.session_state.model is not None and st.session_state.processed_data is not None:
        st.header("Model Evaluation")

        try:
            # Training history
            st.subheader("Training History")
            fig = plot_training_history(st.session_state.training_history)
            st.plotly_chart(fig, use_container_width=True)

            # Model metrics
            metrics, fpr, tpr, roc_auc = evaluate_model(
                st.session_state.model,
                st.session_state.processed_data['X_test'],
                st.session_state.processed_data['y_test']
            )

            # Display metrics
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Classification Metrics")
                metrics_df = pd.DataFrame({
                    'Metric': ['Precision', 'Recall', 'F1-Score'],
                    'Value': [
                        metrics['precision'],
                        metrics['recall'],
                        metrics['f1-score']
                    ]
                })
                st.table(metrics_df)

            with col2:
                st.subheader("ROC Curve")
                fig = plot_roc_curve(fpr, tpr, roc_auc)
                st.plotly_chart(fig, use_container_width=True)

            # Feature importance
            st.subheader("Feature Importance Analysis")
            feature_importance = get_feature_importance(
                st.session_state.model,
                list(st.session_state.processed_data['kmer_freq'].keys())
            )
            fig = plot_feature_importance(feature_importance)
            st.plotly_chart(fig, use_container_width=True)

            # Confusion matrix
            st.subheader("Confusion Matrix")
            predictions = st.session_state.model.predict(
                st.session_state.processed_data['X_test']
            )
            fig = plot_confusion_matrix(
                st.session_state.processed_data['y_test'],
                predictions
            )
            st.plotly_chart(fig, use_container_width=True)

            # Cross-validation
            st.subheader("Cross-validation Results")
            cv_mean, cv_std = cross_validate(
                st.session_state.model,
                st.session_state.processed_data['X_test'],
                st.session_state.processed_data['y_test']
            )
            st.metric("Cross-validation Accuracy",
                     f"{cv_mean:.3f} ± {cv_std:.3f}")

        except Exception as e:
            st.error(f"Error during model evaluation: {str(e)}")
            st.info("Try retraining the model with different parameters.")

with tab5:
    if sequences_df is not None and (st.session_state.model is not None or st.session_state.analysis_results is not None):
        st.header("Export Results")
        if st.button("Export Analysis Results"):
            try:
                results = {
                    'sequences': sequences_df.to_dict(orient='records'),
                    'analysis': {}
                }

                # Add AI analysis results if available
                if st.session_state.analysis_results is not None:
                    results['analysis'].update({
                        'clusters': st.session_state.analysis_results['clusters'].tolist(),
                        'feature_importance': st.session_state.analysis_results['feature_importance']
                    })

                # Add model results if available
                if st.session_state.model is not None:
                    results['model_metrics'] = metrics
                    results['cross_validation'] = {
                        'mean_accuracy': float(cv_mean),
                        'std_accuracy': float(cv_std)
                    }

                st.download_button(
                    "Download Results (JSON)",
                    data=pd.json_normalize(results).to_json(),
                    file_name="biomarker_analysis_results.json",
                    mime="application/json"
                )

            except Exception as e:
                st.error(f"Error exporting results: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
Built with Streamlit • TensorFlow • Scikit-learn • Biopython  
Created by ABHISHEK S R
""")