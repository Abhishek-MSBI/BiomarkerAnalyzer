import streamlit as st
import pandas as pd
from io import StringIO
import numpy as np
import json
from src.data_processing import (load_genomic_data, get_sequence_stats,
                               process_sequences, load_sample_data,
                               export_results)
from src.model import build_cnn_model, train_model_with_progress, prepare_sequence_data
from src.evaluate import evaluate_model, plot_roc_curve, cross_validate
from src.visualizations import (plot_kmer_distribution, plot_sequence_length_distribution,
                              plot_training_history, plot_feature_importance)

st.set_page_config(page_title="Biomarker Discovery Platform", layout="wide")

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = None

# Title and description
st.title("🧬 Viral Disease Biomarker Discovery Platform")
st.markdown("""
This platform uses deep learning to analyze viral genomic sequences and identify potential biomarkers.
Upload your FASTA file or use sample data to get started.
""")

# Sidebar configuration
st.sidebar.header("Model Parameters")
k_mer_size = st.sidebar.slider("K-mer Size", 4, 8, 6)
epochs = st.sidebar.slider("Training Epochs", 5, 50, 10)
learning_rate = st.sidebar.select_slider(
    "Learning Rate",
    options=[0.1, 0.01, 0.001, 0.0001],
    value=0.001,
    format_func=lambda x: f"{x:.4f}"
)

# Data loading section
st.header("1. Data Upload and Processing")
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
    # Display basic statistics
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sequence Statistics")
        stats = get_sequence_stats(sequences_df)
        for key, value in stats.items():
            st.metric(key, value)

    with col2:
        st.subheader("Sequence Length Distribution")
        fig = plot_sequence_length_distribution(sequences_df)
        st.plotly_chart(fig, use_container_width=True)

    # Process k-mers
    st.header("2. K-mer Analysis")
    kmer_freq = process_sequences(sequences_df, k=k_mer_size)

    st.subheader("K-mer Distribution")
    fig = plot_kmer_distribution(kmer_freq)
    st.plotly_chart(fig, use_container_width=True)

    # Model training
    st.header("3. Model Training")

    if st.button("Train Model", key="train_model"):
        with st.spinner("Preparing data and training model..."):
            # Prepare data
            X = prepare_sequence_data(kmer_freq, k=k_mer_size)
            y = np.random.randint(0, 2, size=len(sequences_df))  # Dummy labels for demonstration

            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            # Build and train model
            st.session_state.model = build_cnn_model((X.shape[1], 1), learning_rate)

            st.subheader("Training Progress")
            st.session_state.training_history = train_model_with_progress(
                st.session_state.model, X_train, y_train,
                X_test, y_test, epochs
            )

            # Store processed data
            st.session_state.processed_data = {
                'X_test': X_test,
                'y_test': y_test,
                'kmer_freq': kmer_freq
            }

            st.success("Model training completed!")

    # Model evaluation
    if st.session_state.model is not None and st.session_state.processed_data is not None:
        st.header("4. Model Evaluation")

        # Training history plot
        st.subheader("Training History")
        fig = plot_training_history(st.session_state.training_history)
        st.plotly_chart(fig, use_container_width=True)

        # Model metrics
        metrics, fpr, tpr, roc_auc = evaluate_model(
            st.session_state.model,
            st.session_state.processed_data['X_test'],
            st.session_state.processed_data['y_test']
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Classification Metrics")
            metrics_df = pd.DataFrame({
                'Metric': ['Precision', 'Recall', 'F1-Score'],
                'Value': [
                    metrics['1']['precision'],
                    metrics['1']['recall'],
                    metrics['1']['f1-score']
                ]
            })
            st.table(metrics_df)

        with col2:
            st.subheader("ROC Curve")
            fig = plot_roc_curve(fpr, tpr, roc_auc)
            st.plotly_chart(fig, use_container_width=True)

        # Feature importance
        st.subheader("Feature Importance Analysis")
        feature_names = list(st.session_state.processed_data['kmer_freq'].keys())
        fig = plot_feature_importance(st.session_state.model, feature_names)
        st.plotly_chart(fig, use_container_width=True)

        # Cross-validation
        st.subheader("Cross-validation Results")
        cv_mean, cv_std = cross_validate(
            st.session_state.model,
            st.session_state.processed_data['X_test'],
            st.session_state.processed_data['y_test']
        )
        st.write(f"Mean CV Accuracy: {cv_mean:.3f} ± {cv_std:.3f}")

        # Export results
        st.header("5. Export Results")
        if st.button("Export Analysis Results"):
            results = export_results(
                sequences_df,
                metrics,
                st.session_state.processed_data['kmer_freq']
            )
            st.download_button(
                "Download Results (JSON)",
                results,
                "biomarker_analysis_results.json",
                "application/json"
            )

else:
    st.info("Please upload a FASTA file or use sample data to begin analysis")

# Footer
st.markdown("---")
st.markdown("""
Built with Streamlit • TensorFlow • Biopython  
Created by ABHISHEK S R
""")